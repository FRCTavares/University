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
#include <map>
#include "pico/multicore.h"

#include "Globals.h"
#include "CANComm.h"
#include "CommandInterface.h"
#include "DataLogger.h"
#include "LEDDriver.h"
#include "Metrics.h"
#include "PIController.h"
#include "SensorManager.h"
#include "Calibration.h"
#include "ADMM.h"

//=============================================================================
// INTER-CORE MESSAGING
//=============================================================================

// Message types for inter-core communication
#define MSG_SEND_SENSOR 1
#define MSG_SEND_CONTROL 2
#define MSG_UPDATE_STATE 3

// Constants for consensus protocol
#define CONSENSUS_TIMEOUT_MS 2000 // Timeout for waiting on proposals

// Message structure for inter-core communication
struct CoreMessage
{
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
const float SETPOINT_UNOCCUPIED = 10.0;
const float SETPOINT_OCCUPIED = 20.0;

// CAN monitoring flag
bool canMonitorEnabled = false;

//=============================================================================
// NEIGHBOR MANAGEMENT
//=============================================================================

NeighborInfo neighbors[MAX_NEIGHBORS];

// System stabilization detection variables
const float STABILITY_LUX_THRESHOLD = 0.5f;   // Maximum lux deviation to be considered stable
const float STABILITY_DUTY_THRESHOLD = 0.01f; // Maximum duty cycle change to be considered stable
const unsigned long STABILITY_TIME_MS = 2000; // Time system must maintain stability (ms)
bool systemStable = false;                    // Current stability state
unsigned long lastStabilityChange = 0;        // When stability state last changed
bool stabilizationReported = false;           // Whether we've reported this stable state

// =============================================================================
// ADMM CONSENSUS PROTOCOL VARIABLES
// =============================================================================

// These variables track the ADMM consensus state
static bool is_running_admm_consensus = false;
static bool is_first_admm_iteration = false;
static int admm_consensus_iteration = 0;
static int max_admm_iterations = 10;
static std::map<std::pair<int, uint8_t>, float> d_received_values;
enum admm_consensus_stage_t
{
  ADMM_CONSENSUS_ITERATION,
  ADMM_WAIT_FOR_MESSAGES
};
static admm_consensus_stage_t admm_consensus_stage;
const float maxGain = 30.0f; // Maximum expected gain value

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
    // Only set duty cycle to 0 if we're actually transitioning to OFF state
    controlState.dutyCycle = 0.0;
    break;

  case STATE_UNOCCUPIED:
    controlState.setpointLux = controlState.unoccupiedLuxBound; // Use configurable value
    controlState.feedbackControl = true;
    break;

  case STATE_OCCUPIED:
    controlState.setpointLux = controlState.occupiedLuxBound; // Use configurable value
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

//===========================================================================================================================================================
// Distributed Coordinated Control (Greedy)
//===========================================================================================================================================================

float computeFeedforward();
float getConsensusFeedforwardTerm();
void checkAndReportStability();

// Add after other utility functions
/**
 * Compute the greedy feedforward duty cycle
 * Calculates duty cycle based on: d_ff = (r_i - o_i - sum_of_neighbor_contributions) / k_ii
 *
 * @return Feedforward duty cycle [0,1]
 */
float computeFeedforward()
{
  // Get our node's index in the calibration matrix
  int ourIndex = -1;
  float setpoint = 0.0f;
  float ownGain = 0.0f;
  float ownDuty = 0.0f; // Add declaration for own duty cycle
  float measuredLux = readLux();

  // Read necessary values under lock
  critical_section_enter_blocking(&commStateLock);

  // Get our index in the calibration matrix
  for (int i = 0; i < commState.calibMatrix.numNodes; i++)
  {
    if (commState.calibMatrix.nodeIds[i] == deviceConfig.nodeId)
    {
      ourIndex = i;
      break;
    }
  }

  // If we couldn't find our index, return 0
  if (ourIndex < 0)
  {
    critical_section_exit(&commStateLock);
    return 0.0f;
  }

  // Get setpoint and external light
  setpoint = controlState.setpointLux;

  // Get our own gain (k_ii) and current duty cycle
  ownGain = commState.calibMatrix.gains[ourIndex][ourIndex];
  ownDuty = controlState.dutyCycle;

  // Calculate the total LED contribution (including our own)
  float totalLEDContribution = ownGain * ownDuty; // Start with our own contribution

  // Add neighbor contributions to the total
  for (int i = 0; i < MAX_NEIGHBORS; i++)
  {
    if (neighbors[i].isActive)
    {
      // Find this neighbor's index in the calibration matrix
      int neighborIndex = -1;
      for (int j = 0; j < commState.calibMatrix.numNodes; j++)
      {
        if (commState.calibMatrix.nodeIds[j] == neighbors[i].nodeId)
        {
          neighborIndex = j;
          break;
        }
      }

      // If we found this neighbor in the matrix, add their contribution
      if (neighborIndex >= 0 && neighborIndex != ourIndex)
      {
        totalLEDContribution += commState.calibMatrix.gains[ourIndex][neighborIndex] *
                                neighbors[i].lastDuty;
      }
    }
  }

  critical_section_exit(&commStateLock);

  // Dynamically estimate current external light
  // (measured light - light from all LEDs = must be external light)
  float estimatedExternalLight = measuredLux - totalLEDContribution;

  // Ensure we don't get negative external light estimates due to noise or timing issues
  if (estimatedExternalLight < 0)
    estimatedExternalLight = 0;

  // Calculate the feedforward duty cycle with real-time external light estimation
  // d_ff = (r_i - o_i - sum_of_neighbor_contributions) / k_ii
  float d_ff = 0.0f;

  // Avoid division by zero
  if (ownGain > 0.01f)
  {
    // (desired lux - external lux - neighbor contributions) / own gain
    float neighborContributions = totalLEDContribution - (ownGain * ownDuty);
    d_ff = (setpoint - estimatedExternalLight - neighborContributions) / ownGain;

    // Clamp to valid range [0,1]
    d_ff = constrain(d_ff, 0.0f, 1.0f);
  }

  return d_ff;
}

/**
 * Get the consensus feedforward term for the current control cycle
 * Periodically computes a new feedforward term and broadcasts our duty cycle
 *
 * @return Feedforward term [0,1] or -1 if not available
 */
float getConsensusFeedforwardTerm()
{
  static unsigned long lastFeedforwardTime = 0;
  const unsigned long FEEDFORWARD_INTERVAL_MS = 1000; // Compute feedforward every 1 second

  unsigned long currentTime = millis();

  // Check if it's time to compute a new feedforward term
  if (currentTime - lastFeedforwardTime >= FEEDFORWARD_INTERVAL_MS)
  {
    lastFeedforwardTime = currentTime;

    // Compute the feedforward term
    float d_ff = computeFeedforward();

    // Broadcast our current duty cycle to neighbors
    critical_section_enter_blocking(&commStateLock);
    float currentDuty = controlState.dutyCycle;
    critical_section_exit(&commStateLock);
    sendSensorReading(CAN_ADDR_BROADCAST, 1, currentDuty);

    return d_ff;
  }

  // Return -1 to indicate no new feedforward term
  return -1.0f;
}

/**
 * Check and report system stability based on lux and duty cycle changes
 * Called after each control update to detect stabilization
 */
void checkAndReportStability()
{
  static float lastLux = 0.0f;
  static float lastDuty = 0.0f;
  unsigned long currentTime = millis();
  float currentLux = readLux();

  critical_section_enter_blocking(&commStateLock);
  float currentDuty = controlState.dutyCycle;
  critical_section_exit(&commStateLock);

  // Check if the system is stable based on lux and duty changes
  bool isStableNow = (fabs(currentLux - lastLux) < STABILITY_LUX_THRESHOLD) &&
                     (fabs(currentDuty - lastDuty) < STABILITY_DUTY_THRESHOLD);

  // Update our tracked values
  lastLux = currentLux;
  lastDuty = currentDuty;

  // If stability state changed, update the time
  if (isStableNow != systemStable)
  {
    systemStable = isStableNow;
    lastStabilityChange = currentTime;
    stabilizationReported = false;
  }

  // If system has been stable for the required time and we haven't reported it
  if (systemStable && !stabilizationReported &&
      (currentTime - lastStabilityChange > STABILITY_TIME_MS))
  {
    Serial.println("System stabilized!");
    stabilizationReported = true;
  }
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
  analogWriteRange(4095);

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
  controlState.feedbackControl = false;
  controlState.antiWindup = true;
  controlState.dutyCycle = 0.0;
  commState.streamingEnabled = false;
  commState.isCalibrationMaster = false;
  commState.calibrationInProgress = false;
  commState.calibrationStep = 0;
  commState.calLastStepTime = 0;
  controlState.systemAwake = false;
  controlState.systemReady = false;
  controlState.discoveryStartTime = 0;
  // Initialize with default setpoint values
  controlState.occupiedLuxBound = SETPOINT_OCCUPIED;
  controlState.unoccupiedLuxBound = SETPOINT_UNOCCUPIED;

  commState.useDefaultGains = true; // Default to real calibration
  controlState.usingSeq = false;      // Start with sequential control by default

  // Initialize the consensus-related fields
  controlState.pendingDutyCycle = 0.0;
  controlState.consensusRound = 0;
  controlState.consensusReached = false;
  controlState.lastProposalTime = 0;
  controlState.iterationCounter = 0; // Initialize the iteration counter

  // Initialize ADMM-related state
  controlState.usingADMM = false; // Start with sequential control by default
  controlState.rho = 0.07;
  controlState.cost = 1.0;

  // Initialize neighbor proposal flags
  for (int i = 0; i < MAX_NEIGHBORS; i++)
  {
    neighbors[i].pendingDuty = 0.0;
    neighbors[i].hasProposal = false;
  }

  commState.readingIndex = 0;
  commState.measurementsStable = false;
  for (int i = 0; i < 5; i++)
  {
    commState.previousReadings[i] = 0.0f;
  }

  // Calibrate the system
  deviceConfig.ledGain = 0;
  Serial.println();

  // Configure device based on unique ID
  configureDeviceFromID();

  initCANComm();
  initPendingQueries();

  // Launch core 0 for CAN communication
  multicore_launch_core1(core1_main);

  controlState.standbyMode = true; // Start in standby mode by default
  Serial.println("System starting in standby mode. Send 'wakeup' command to activate.");
}

/**
 * Arduino main loop
 * Processes sensor readings, controls, and communication
 */
void loop()
{
  // ===============================================================
  // SYSTEM STATE MANAGEMENT
  // ===============================================================

  // Get current system state
  critical_section_enter_blocking(&commStateLock);
  bool inStandby = controlState.standbyMode;
  bool inWakeUpDiscovery = controlState.systemAwake && !controlState.systemReady;
  bool isSystemReady = controlState.systemReady;
  unsigned long discoveryStartTime = controlState.discoveryStartTime;
  bool feedbackEnabled = controlState.feedbackControl;
  LuminaireState currentState = controlState.luminaireState;
  critical_section_exit(&commStateLock);

  float lux = readLux();

  // Get current duty cycle
  critical_section_enter_blocking(&commStateLock);
  float currentDuty = controlState.dutyCycle;
  float setpoint = controlState.setpointLux;
  critical_section_exit(&commStateLock);

  // Log data point
  logData(millis(), lux, currentDuty);

  // Process serial commands in all states (required for standby wake-up)
  processSerialCommands();

  // Handle standby mode - minimal processing
  if (inStandby)
  {
    delay(100); // Reduced CPU usage while in standby
    return;     // Skip the rest of the loop
  }

  // Handle wake-up discovery phase
  if (inWakeUpDiscovery)
  {
    // Check if discovery period has ended (5 seconds)
    if (millis() - discoveryStartTime > 5000)
    {
      Serial.println("\nNode discovery phase complete");
      displayDiscoveredNodes();

      // Transition to calibration phase
      Serial.println("Starting system calibration as coordinated...");
      critical_section_enter_blocking(&commStateLock);
      controlState.systemAwake = false;
      controlState.systemReady = false; // Will be set true after calibration
      critical_section_exit(&commStateLock);

      // Trigger calibration
      startCalibration();
    }

    // Continue with minimal processing during discovery
    sendHeartbeat();
    delay(50);
    return;
  }

  // ===============================================================
  // INPUT PROCESSING
  // ===============================================================

  // Read sensor data - do this early to have fresh data for all functions

  // Update filtered lux value in the sensor state
  critical_section_enter_blocking(&commStateLock);
  sensorState.filteredLux = lux;
  critical_section_exit(&commStateLock);

  // ===============================================================
  // CALIBRATION MANAGEMENT
  // ===============================================================

  // Check calibration status
  critical_section_enter_blocking(&commStateLock);
  bool calibInProgress = commState.calibrationInProgress;
  bool wasCalibrating = commState.wasCalibrating;
  critical_section_exit(&commStateLock);

  // Handle active calibration process
  if (calibInProgress)
  {
    updateCalibrationState();
    // Skip control during calibration
    logData(millis(), lux, 0.0f);
    delay((int)(pid.getSamplingTime() * 1000));
    return;
  }

  // Handle post-calibration initialization
  if (wasCalibrating)
  {
    handlePostCalibrationSetup();
    return;
  }

  // ===============================================================
  // COORDINATED CONTROL SYSTEM EXECUTION
  // ===============================================================

  // Get current system state safely
  critical_section_enter_blocking(&commStateLock);
  currentDuty = controlState.dutyCycle;
  setpoint = controlState.setpointLux;
  bool usingADMM = controlState.usingADMM;
  bool usingSeq = controlState.usingSeq;
  critical_section_exit(&commStateLock);

  // Branch based on control method
  if (feedbackEnabled && !inStandby && currentState != STATE_OFF)
  {
    float admmDuty = 0.0f;
    float u_pwm = 0.0f;

    updateExternalIlluminanceForControl();

    if (usingADMM)
    {
      // Run ADMM optimization
      bool admmUpdated = updateADMMConsensus();

      // Even if no update this cycle, use the current ADMM result as feedforward
      // Get ADMM calculated duty cycle
      admmDuty = controlToOutputDuty(localNode.u[localNode.index]);

      // Use ADMM as a feedforward term with PID for error correction
      u_pwm = pid.computeWithFeedforward(setpoint, lux, admmDuty);

      // Convert to duty cycle and apply limits
      float finalDuty = u_pwm / 4095.0f;
      finalDuty = constrain(finalDuty, 0.0f, 1.0f);

      // Apply the corrected duty cycle
      setLEDDutyCycle(finalDuty);

      // Log status periodically
      static unsigned long lastLogTime = 0;
      if (millis() - lastLogTime > 5000)
      {
        Serial.print("ADMM duty: ");
        Serial.print(admmDuty);
        Serial.print(", PID corrected: ");
        Serial.println(finalDuty);
        lastLogTime = millis();
      }
    }
    else if (usingSeq)
    {
      // Track the last valid consensus feedforward term for reuse
      static float lastValidFeedforward = 0.0f;

      // Try to get a new feedforward term
      float consensusFeedforward = getConsensusFeedforwardTerm();

      // Update our saved value if we got a valid new one
      if (consensusFeedforward >= 0.0f)
      {
        lastValidFeedforward = consensusFeedforward;
      }

      // Always run PID with the most recent valid feedforward term
      float u_pwm = pid.computeWithFeedforward(setpoint, lux, lastValidFeedforward);
      float finalDuty = u_pwm / 4095.0f;

      // Constrain to valid range
      finalDuty = constrain(finalDuty, 0.0f, 1.0f);

      // Apply the duty cycle to the LED
      setLEDDutyCycle(finalDuty);

      // Update the system state
      critical_section_enter_blocking(&commStateLock);
      controlState.dutyCycle = finalDuty;
      critical_section_exit(&commStateLock);

      // Check system stability after control update
      checkAndReportStability();
    }
    else if (feedbackEnabled && !usingADMM && !usingSeq)
    {
      // Run normal PID control without ADMM or sequential control
      float u = pid.compute(setpoint, lux);
      setLEDPWMValue((int)u);
    }
    else
    {
      // Direct duty cycle control in manual mode
      setLEDDutyCycle(controlState.dutyCycle);
    }
  }

  // Handle streaming requests
  handleStreaming();
  handleRemoteStreamRequests();

  // ===============================================================
  // DATA LOGGING AND TIMING
  // ===============================================================

  // Log data for monitoring and metrics
  logData(millis(), lux, controlState.dutyCycle);

  // Wait for next control cycle
  delay((int)(pid.getSamplingTime() * 1000));
}

/**
 * Handle post-calibration system setup
 * Initializes control state after calibration is complete
 */
void handlePostCalibrationSetup()
{
  // Reset calibration flags and set system as ready
  critical_section_enter_blocking(&commStateLock);
  commState.wasCalibrating = false;
  controlState.systemReady = true;
  critical_section_exit(&commStateLock);

  // Set initial state to UNOCCUPIED
  changeState(STATE_UNOCCUPIED);

  // Calculate and apply initial duty cycle
  float initialDuty = SETPOINT_UNOCCUPIED / deviceConfig.ledGain;
  initialDuty = constrain(initialDuty, 0.1f, 0.7f);
  setLEDDutyCycle(initialDuty);

  // Reset PID controller to avoid transients
  pid.reset();

  // Update control state with initial settings
  critical_section_enter_blocking(&commStateLock);
  controlState.dutyCycle = initialDuty;
  critical_section_exit(&commStateLock);

  // Broadcast state change and initial duty cycle to network
  sendControlCommand(CAN_ADDR_BROADCAST, CAN_CTRL_STATE_CHANGE, (float)STATE_UNOCCUPIED);
  sendSensorReading(CAN_ADDR_BROADCAST, 1, initialDuty);

  Serial.println("Calibration complete!");

  // Add a short delay to separate the calibration message from help output
  delay(500);

  // Print welcome message and available commands
  Serial.println("\n===== System Ready =====");
  Serial.println("The following commands are now available:");

  // Print the full help menu
  printHelp();
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

    if (currentTime - lastHeartbeatTime >= HEARTBEAT_INTERVAL)
    {
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
        switch (msg.dataType)
        {
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
    if (currentTime - lastStatusUpdateTime >= 1000)
    {
      updateNodeStatus();
      lastStatusUpdateTime = currentTime;
    }

    // Brief delay to prevent core hogging
    sleep_ms(1);
  }
}