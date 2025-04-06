/**
 * Distributed Lighting Control System
 *
 * This system implements an intelligent distributed lighting controller with:
 * - Illuminance sensing and filtering
 * - PID-based feedback control
 * - CAN bus communication between nodes
 * - Neighbor state coordination for energy efficiency
 * - Multiple operating modes (off, unoccupied, occupied)
 * - Serial command interface for configuration and monitoring
 * - Data logging and streaming capabilities
 */
#include <Arduino.h>
#include <math.h>
#include "pico/multicore.h"

#include "Globals.h"
#include "CANComm.h"
#include "CommandInterface.h"
#include "DataLogger.h"
#include "LEDDriver.h"
#include "Metrics.h"
#include "PIController.h"
#include "SensorManager.h"

//=============================================================================
// INTER-CORE MESSAGING
//=============================================================================

// Message types for inter-core communication
#define MSG_SEND_SENSOR 1
#define MSG_SEND_CONTROL 2
#define MSG_UPDATE_STATE 3

// Message structure for inter-core communication
struct CoreMessage {
  uint8_t msgType;  // Message type identifier
  uint8_t dataType; // Type of data being sent
  float value;      // Value to send
  uint8_t nodeId;   // Target or source node ID
};

// Queues for communication between cores
queue_t core0to1queue;
queue_t core1to0queue;

//=============================================================================
// GLOBAL SYSTEM STATE
//=============================================================================

// System state structures (see Globals.h for definitions)
critical_section_t commStateLock;
DeviceConfig deviceConfig;
SensorState sensorState;
ControlState controlState;
CommState commState;

//=============================================================================
// HARDWARE CONFIGURATION
//=============================================================================

// Pin assignments
const int LED_PIN = 15; // PWM output for LED driver

// PID controller configuration
const float Kp = 20;    // Initial proportional gain (will be replaced by ledGain)
const float Ki = 400;   // Integral gain
const float DT = 0.01;  // Sampling period (seconds)
const float BETA = 0.8; // Setpoint weighting factor (0.0-1.0)
PIController pid(Kp, Ki, BETA, DT);

//=============================================================================
// SYSTEM SETTINGS
//=============================================================================

// Illuminance setpoints for different states
const float SETPOINT_OFF = 0.0;
const float SETPOINT_UNOCCUPIED = 5.0;
const float SETPOINT_OCCUPIED = 15.0;

// CAN monitoring flag
bool canMonitorEnabled = false;

//=============================================================================
// NEIGHBOR MANAGEMENT
//=============================================================================

// Maximum number of neighbors to track
const int MAX_NEIGHBORS = 5;

// Structure to store neighbor information
struct NeighborInfo {
  uint8_t nodeId;           // CAN node ID
  float lastLux;            // Last reported illuminance
  float lastDuty;           // Last reported duty cycle
  LuminaireState state;     // Current operating state
  unsigned long lastUpdate; // Last update timestamp
  bool isActive;            // Is node currently active
};

// Array of neighbor information
NeighborInfo neighbors[MAX_NEIGHBORS];

//=============================================================================
// SYSTEM UTILITY FUNCTIONS
//=============================================================================

/**
 * Change the luminaire operating state
 * Updates system state and broadcasts change to network
 *
 * @param newState New state to set (OFF, UNOCCUPIED, OCCUPIED)
 */
void changeState(LuminaireState newState)
{
  // Don't do anything if state is unchanged
  critical_section_enter_blocking(&commStateLock);
  if (newState == controlState.luminaireState)
  {
    critical_section_exit(&commStateLock); 
    return;
  }

  controlState.luminaireState = newState;

  // Update setpoint based on new state
  switch (controlState.luminaireState)
  {
  case STATE_OFF:
    controlState.setpointLux = SETPOINT_OFF;
    controlState.feedbackControl = false;
    break;

  case STATE_UNOCCUPIED:
    controlState.setpointLux = SETPOINT_UNOCCUPIED;
    controlState.feedbackControl = true;
    break;

  case STATE_OCCUPIED:
    controlState.setpointLux = SETPOINT_OCCUPIED;
    controlState.feedbackControl = true;
    break;
  }
  critical_section_exit(&commStateLock);

  // Reset PID controller to avoid integral windup during transitions
  pid.reset();

  // Send to CAN bus using core0
  CoreMessage msg;
  msg.msgType = MSG_SEND_CONTROL;
  msg.nodeId = CAN_ADDR_BROADCAST;
  msg.dataType = CAN_CTRL_STATE_CHANGE;
  msg.value = (float)controlState.luminaireState;
  queue_add_blocking(&core1to0queue, &msg);
}

/**
 * Update stored neighbor information when receiving CAN messages
 *
 * @param nodeId Source node ID
 * @param sensorType Type of sensor data (0=lux, 1=duty, 2=state)
 * @param value Sensor reading value
 */
void updateNeighborInfo(uint8_t nodeId, uint8_t sensorType, float value)
{
  int emptySlot = -1;

  // Find existing neighbor or empty slot
  for (int i = 0; i < MAX_NEIGHBORS; i++)
  {
    if (neighbors[i].isActive && neighbors[i].nodeId == nodeId)
    {
      // Update existing neighbor
      if (sensorType == 0)
        neighbors[i].lastLux = value;
      else if (sensorType == 1)
        neighbors[i].lastDuty = value;
      else if (sensorType == 2)
        neighbors[i].state = (LuminaireState)((int)value);

      neighbors[i].lastUpdate = millis();
      return;
    }

    if (!neighbors[i].isActive && emptySlot < 0)
    {
      emptySlot = i;
    }
  }

  // Add as new neighbor if slot available
  if (emptySlot >= 0)
  {
    neighbors[emptySlot].nodeId = nodeId;
    neighbors[emptySlot].isActive = true;
    neighbors[emptySlot].lastUpdate = millis();

    if (sensorType == 0)
      neighbors[emptySlot].lastLux = value;
    else if (sensorType == 1)
      neighbors[emptySlot].lastDuty = value;
    else if (sensorType == 2)
      neighbors[emptySlot].state = (LuminaireState)((int)value);
  }
}

/**
 * Get estimated power consumption based on current duty cycle
 *
 * @return Estimated power in watts
 */
float getPowerConsumption()
{
  critical_section_enter_blocking(&commStateLock);
  float duty = controlState.dutyCycle;
  critical_section_exit(&commStateLock);

  return duty * MAX_POWER_WATTS;
}

/**
 * Get elapsed time since boot
 *
 * @return Time in seconds
 */
unsigned long getElapsedTime()
{
  return millis() / 1000;
}

/**
 * Apply device configuration settings
 * Updates global variables and configures controller
 *
 * @param config Configuration to apply
 */
void applyDeviceConfig(const DeviceConfig &config)
{
  deviceConfig = config; // Store the configuration

  // Configure PID controller
  pid.setGains(config.pidKp, config.pidKi);
  pid.setWeighting(config.pidBeta);

  Serial.print("Configured as device #");
  Serial.print(config.nodeId);
  Serial.print(", Kp=");
  Serial.print(config.pidKp);
  Serial.print(", Ki=");
  Serial.print(config.pidKi);
  Serial.print(", Beta=");
  Serial.print(config.pidBeta);
  Serial.println();
}

/**
 * Configure device based on hardware ID
 * Sets node ID and PID parameters based on unique board ID
 */
void configureDeviceFromID()
{
  // Read hardware ID
  pico_unique_board_id_t board_id;
  pico_get_unique_board_id(&board_id);
  
  // Generate unique node ID from the last byte (1-63)
  uint8_t nodeId = board_id.id[7] & 0x3F; // Use last 6 bits for node ID (0-63)

  // Save existing calibration values
  float existingGain = deviceConfig.ledGain;
  float existingOffset = deviceConfig.calibrationOffset;

  // Start with default configuration
  DeviceConfig config;
  
  config.nodeId = nodeId;
  
  // Apply device-type specific PID parameters
  switch (config.nodeId)
  {
  case 33: // First device type
    config.pidKp = 20.0;
    config.pidKi = 400.0;
    config.pidBeta = 0.8;
    break;

  case 40: // Second device type
    config.pidKp = 25.0;
    config.pidKi = 350.0;
    config.pidBeta = 0.75;
    break;

  case 52: // Third device type
    config.pidKp = 22.0;
    config.pidKi = 380.0;
    config.pidBeta = 0.82;
    break;
  }

  // Preserve calibration values
  config.ledGain = existingGain;
  config.calibrationOffset = existingOffset;

  // Apply the configuration
  applyDeviceConfig(config);
}

//=============================================================================
// MAIN PROGRAM
//=============================================================================

/**
 * Arduino setup function
 * Initializes hardware and subsystems
 */
void setup()
{
  Serial.begin(115200);
  delay(2000); // Allow time for serial connection to establish

  // Configure hardware
  analogReadResolution(12);
  analogWriteFreq(30000);
  analogWriteRange(4095); // Use the constant value directly

  // Initialize subsystems
  initLEDDriver(LED_PIN);
  initStorage();

  // Initialize synchronization primitives
  critical_section_init(&commStateLock);
  queue_init(&core0to1queue, sizeof(CoreMessage), 10);
  queue_init(&core1to0queue, sizeof(CoreMessage), 10);

  // Set initial state values
  sensorState.filterEnabled = true;
  sensorState.lastFilteredLux = -1.0;

  controlState.setpointLux = SETPOINT_OFF;
  controlState.luminaireState = STATE_OFF;
  controlState.feedbackControl = true;
  controlState.antiWindup = true;
  controlState.dutyCycle = 0.0; 
  commState.streamingEnabled = false;
  commState.isCalibrationMaster = false;
  commState.calibrationInProgress = false;
  commState.calibrationStep = 0;
  commState.calLastStepTime = 0;
  
  // Calibrate the system
  deviceConfig.ledGain = 0;
  Serial.println();

  // Configure device based on unique ID
  configureDeviceFromID();

  initCANComm();
  initPendingQueries();

  // Launch core 0 for CAN communication
  multicore_launch_core1(core1_main);

  Serial.println("Distributed Control System with CAN-BUS initialized");
}

/**
 * Arduino main loop
 * Processes sensor readings, controls, and communication
 */
void loop()
{
  // (A) Process incoming serial commands
  processSerialCommands();

  // (B) Handle streaming
  handleStreaming();
  handleRemoteStreamRequests();

  // (C) Read sensor data safely
  float lux = readLux();

  // Update calibration state machine if calibration is in progress
  critical_section_enter_blocking(&commStateLock);
  bool calibInProgress = commState.calibrationInProgress;
  critical_section_exit(&commStateLock);
  
  if (calibInProgress) {
    updateCalibrationState();
  }

  critical_section_enter_blocking(&commStateLock);
  sensorState.filteredLux = lux;
  LuminaireState state = controlState.luminaireState;
  bool feedback = controlState.feedbackControl;
  float setpoint = controlState.setpointLux;
  critical_section_exit(&commStateLock);

  if (state == STATE_OFF)
  {
    setLEDDutyCycle(0.0);
  }
  else if (feedback)
  {
    float u = pid.compute(setpoint, lux);
    setLEDPWMValue((int)u);
  }
  else
  {
    critical_section_enter_blocking(&commStateLock);
    float duty = controlState.dutyCycle;
    critical_section_exit(&commStateLock);
    setLEDDutyCycle(duty);
  }

  // Update duty cycle after control action
  critical_section_enter_blocking(&commStateLock);
  controlState.dutyCycle = getLEDDutyCycle();
  critical_section_exit(&commStateLock);

  // Log data
  logData(millis(), lux, controlState.dutyCycle);

  // Wait for next control cycle
  delay((int)(pid.getSamplingTime() * 1000));
}

/**
 * Core 0: CAN Communication processing function
 * Runs on the second core and handles all CAN-related operations
 */
void core1_main()
{
  // Add these variables at the top of core1_main
  unsigned long lastHeartbeatTime = 0;
  const unsigned long HEARTBEAT_INTERVAL = 3000; // Send heartbeat every 3 seconds
  
  // Also declare other timing variables at the top
  unsigned long lastStatusUpdateTime = 0;
  
  while (true)
  {
    // Process incoming CAN messages
    canCommLoop();
    
    // Handle periodic heartbeat transmission
    unsigned long currentTime = millis();
    
    if (currentTime - lastHeartbeatTime >= HEARTBEAT_INTERVAL) {
      sendHeartbeat();
      lastHeartbeatTime = currentTime;
      
      // While we're at it, also update node status (mark inactive nodes)
      updateNodeStatus();
    }
    
    // Process outgoing messages from Core 0
    CoreMessage msg;
    if (queue_try_remove(&core1to0queue, &msg))
    {
      bool sendResult = false;
      switch (msg.msgType)
      {
        case MSG_SEND_SENSOR:
          sendResult = sendSensorReading(msg.nodeId, msg.dataType, msg.value);
          break;
          
        case MSG_SEND_CONTROL:
          sendResult = sendControlCommand(msg.nodeId, msg.dataType, msg.value);
          break;
          
        case MSG_UPDATE_STATE:
          // Handle state updates - this might update local state based on Core 0 request
          critical_section_enter_blocking(&commStateLock);
          switch(msg.dataType) {
            case CAN_CTRL_STATE_CHANGE:
              controlState.luminaireState = (LuminaireState)(int)msg.value;
              break;
            // Add other state update types as needed
          }
          critical_section_exit(&commStateLock);
          sendResult = true;
          break;
      }
      
      // Could report back to Core 0 if needed
    }
  
    
    // Update node status every second
    if (currentTime - lastStatusUpdateTime >= 1000) {
      updateNodeStatus();
      lastStatusUpdateTime = currentTime;
    }
    
    // Brief delay to prevent core hogging
    sleep_ms(1);
  }
}