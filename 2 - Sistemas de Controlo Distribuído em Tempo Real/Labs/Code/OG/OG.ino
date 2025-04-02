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
#include "PIController.h"
#include "CommandInterface.h"
#include "CANComm.h"
#include "pico/multicore.h"
#include "Globals.h"


#define MSG_SEND_SENSOR 1
#define MSG_SEND_CONTROL 2
#define MSG_UPDATE_STATE 3

critical_section_t commStateLock;
DeviceConfig deviceConfig;
SensorState sensorState;
ControlState controlState;
CommState commState;

struct CoreMessage
{
  uint8_t msgType;  // Message type identifier
  uint8_t dataType; // Type of data being sent
  float value;      // Value to send
  uint8_t nodeId;   // Target or source node ID
};

queue_t core0to1queue;
queue_t core1to0queue;

//=============================================================================
// HARDWARE CONFIGURATION
//=============================================================================

//-----------------------------------------------------------------------------
// Sensor Configuration
//-----------------------------------------------------------------------------

// LDR Calibration parameters (for lux conversion)
const float R10 = 225000.0;       // LDR resistance at ~10 lux (ohms)
const float LDR_M = -1.0;         // Slope of log-log resistance vs. illuminance
float LDR_B = log10(R10) - LDR_M; // Y-intercept for log-log conversion

//-----------------------------------------------------------------------------
// Pin Assignments
//-----------------------------------------------------------------------------
const int LED_PIN = 15; // PWM output for LED driver
const int LDR_PIN = A0; // Analog input for light sensor

//-----------------------------------------------------------------------------
// PWM Configuration
//-----------------------------------------------------------------------------
const int PWM_MAX = 4095; // Maximum PWM value (12-bit)
const int PWM_MIN = 0;    // Minimum PWM value (off)

//-----------------------------------------------------------------------------
// Measurement Filter Configuration
//-----------------------------------------------------------------------------
const int NUM_SAMPLES = 10;          // Samples for averaging
const float OUTLIER_THRESHOLD = 2.0; // Standard deviations for outlier detection
const float ALPHA = 0.3;             // EMA filter coefficient (0=slow, 1=fast)

//-----------------------------------------------------------------------------
// PID Controller Parameters
//-----------------------------------------------------------------------------
const float Kp = 20;    // Initial proportional gain (will be replaced by ledGain)
const float Ki = 400;   // Integral gain
const float DT = 0.01;  // Sampling period (seconds)
const float BETA = 0.8; // Setpoint weighting factor (0.0-1.0)
PIController pid(Kp, Ki, BETA, DT); // Create pid controller with initial values
//-----------------------------------------------------------------------------
// Power Consumption Model
//-----------------------------------------------------------------------------
const float MAX_POWER_WATTS = 0.08755; // Maximum power consumption at 100% duty

//=============================================================================
// GLOBAL SYSTEM STATE
//=============================================================================

// Illuminance setpoints for different states
const float SETPOINT_OFF = 0.0;        // Off state target (lux)
const float SETPOINT_UNOCCUPIED = 5.0; // Unoccupied state target (lux)
const float SETPOINT_OCCUPIED = 15.0;  // Occupied state target (lux)

//-----------------------------------------------------------------------------
// Flicker Comparison Variables
//-----------------------------------------------------------------------------
float flickerWithFilter = 0.0;    // Accumulated flicker with filtering
float flickerWithoutFilter = 0.0; // Accumulated flicker without filtering

// Previous duty cycle values for flicker calculation
float prevDutyWithFilter1 = 0.0, prevDutyWithFilter2 = 0.0;
float prevDutyWithoutFilter1 = 0.0, prevDutyWithoutFilter2 = 0.0;
bool flickerInitialized = false; // Track if we've initialized flicker tracking

//-----------------------------------------------------------------------------
// External Light Tracking
//-----------------------------------------------------------------------------
const float EXT_LUX_ALPHA = 0.05;                   // Slow-moving average coefficient
const float EXTERNAL_LIGHT_CHANGE_THRESHOLD = 1.0f; // Minimum lux change to trigger adaptation

//-----------------------------------------------------------------------------
// CAN Communication State
//-----------------------------------------------------------------------------
bool canMonitorEnabled = false;         // Display received messages
uint8_t nodeID = 0;                     // Node ID (set during initialization)
unsigned long lastCANSend = 0;          // Last transmission timestamp

//-----------------------------------------------------------------------------
// Neighbor Management
//-----------------------------------------------------------------------------
struct NeighborInfo
{
  uint8_t nodeId;           // CAN node ID
  float lastLux;            // Last reported illuminance
  float lastDuty;           // Last reported duty cycle
  LuminaireState state;     // Current operating state
  unsigned long lastUpdate; // Last update timestamp
  bool isActive;            // Is node currently active
};

//-----------------------------------------------------------------------------
// Network Discovery and Calibration
//-----------------------------------------------------------------------------

// Network discovery and calibration variables
NetworkState currentNetworkState = NET_STATE_BOOT;
bool discoveredNodes[64] = {false};      // Tracks which nodes have been discovered
uint8_t discoveredCount = 0;             // Count of discovered nodes
bool readyNodes[64] = {false};           // Tracks which nodes are ready for calibration
uint8_t readyCount = 0;                  // Count of ready nodes
uint8_t calibrationNodeSequence = 0;     // Current node in calibration sequence
uint8_t calibrationStep = 0;             // Current step in calibration process
unsigned long lastNetworkActionTime = 0; // Timestamp of last network action
bool isCalibrationComplete = false;      // Flag indicating if calibration is complete

// Light coupling measurement data
float selfGain = 0.0;                    // Luminaire's effect on its own sensor
float crossGains[64] = {0};              // Effect of other luminaires on this sensor

const int MAX_NEIGHBORS = 5;           // Maximum tracked neighbors
NeighborInfo neighbors[MAX_NEIGHBORS]; // Neighbor state array

//-----------------------------------------------------------------------------
// Controller Object
//-----------------------------------------------------------------------------
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
//=============================================================================
// STATE MANAGEMENT SUBSYSTEM
//=============================================================================

/**
 * Read and process illuminance with multi-stage filtering:
 * 1. Multiple samples to reduce noise
 * 2. Statistical outlier rejection
 * 3. EMA filtering for temporal smoothing
 * 4. Calibration offset application
 *
 * @return Processed illuminance value in lux
 */
float readLux()
{
  bool isFilterEnabled;
  float lastFiltered;

  critical_section_enter_blocking(&commStateLock);
  isFilterEnabled = sensorState.filterEnabled;
  lastFiltered = sensorState.lastFilteredLux;
  critical_section_exit(&commStateLock);

  float samples[NUM_SAMPLES];
  float sum = 0.0;
  float count = 0.0;

  // Always take at least one sample to update rawLux
  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    // Read the ADC value from the analog pin
    int adcValue = analogRead(LDR_PIN);
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;

    // Skip invalid readings
    if (voltage <= 0.0)
    {
      continue;
    }

    // Calculate resistance of LDR using voltage divider formula
    float resistance = FIXED_RESISTOR * (VCC / voltage - 1.0);

    // Convert resistance to LUX using calibration formula
    float logR = log10(resistance);
    float logLux = (logR - LDR_B) / LDR_M;
    float luxValue = pow(10, logLux);

    samples[i] = luxValue;
    sum += luxValue;
    count++;
  }

  if (count == 0)
    return 0.0; // No valid readings

  // Store the raw lux (average of all samples without further filtering)
  critical_section_enter_blocking(&commStateLock);
  sensorState.rawLux = sum / count;
  critical_section_exit(&commStateLock);

  // If filtering is disabled, return the raw value immediately
  if (!isFilterEnabled)
  {
    return sensorState.rawLux;
  }

  // 2. Calculate mean and standard deviation
  float mean = sum / count;
  float variance = 0.0;

  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    if (samples[i] > 0)
    { // Only consider valid samples
      variance += sq(samples[i] - mean);
    }
  }
  float stdDev = sqrt(variance);

  // 3. Filter outliers and recalculate mean
  float filteredSum = 0.0;
  float filteredCount = 0.0;

  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    if (samples[i] > 0 && abs(samples[i] - mean) <= OUTLIER_THRESHOLD * stdDev)
    {
      filteredSum += samples[i];
      filteredCount++;
    }
  }

  float filteredMean = (filteredCount > 0) ? filteredSum / filteredCount : mean;

  // 4. Apply exponential moving average (EMA) filter for temporal smoothing
  if (lastFiltered < 0)
  {
    lastFiltered = filteredMean; // First valid reading
  }
  else
  {
    lastFiltered = ALPHA * filteredMean + (1.0 - ALPHA) * lastFiltered;
  }

  // 5. Apply calibration offset and safety bounds check
  float calibratedLux = lastFiltered+ deviceConfig.calibrationOffset;
  if (calibratedLux < 0.0)
    calibratedLux = 0.0;

  // Store in sensorState with proper synchronization
  critical_section_enter_blocking(&commStateLock);
  sensorState.rawLux = sum / count;
  sensorState.lastFilteredLux = calibratedLux;
  critical_section_exit(&commStateLock);

  return calibratedLux;
}

/**
 * Calibrate LUX sensor using a reference measurement
 *
 * @param knownLux Reference illuminance from trusted external meter
 */
void calibrateLuxSensor(float knownLux)
{
  float measuredLux = 0.0;
  const int CAL_SAMPLES = 10;

  for (int i = 0; i < CAL_SAMPLES; i++)
  {
    // Use a special raw reading to avoid existing calibration
    int adcValue = analogRead(LDR_PIN);
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
    if (voltage <= 0.0)
      continue;

    float resistance = FIXED_RESISTOR * (VCC / voltage - 1.0);
    float logR = log10(resistance);
    float logLux = (logR - LDR_B) / LDR_M;
    critical_section_enter_blocking(&commStateLock);
    sensorState.rawLux = pow(10, logLux);
    critical_section_exit(&commStateLock);

    measuredLux += sensorState.rawLux;
    delay(50); // Short delay between readings
  }
  measuredLux /= CAL_SAMPLES;

  // Calculate the offset needed
  deviceConfig.calibrationOffset = knownLux - measuredLux;

  Serial.print("Sensor calibrated: offset = ");
  Serial.println(deviceConfig.calibrationOffset);
}

/**
 * Get raw voltage at LDR sensing pin
 *
 * @return Voltage at LDR pin (0-VCC)
 */
float getVoltageAtLDR()
{
  int adcValue = analogRead(LDR_PIN);
  return (adcValue / MY_ADC_RESOLUTION) * VCC;
}

/**
 * Get estimated external illuminance (without LED contribution)
 * Calculates background illuminance by subtracting LED contribution from total
 *
 * @return Estimated external illuminance in lux
 */
float getExternalIlluminance()
{
  float totalLux;
  float ledContribution;
  float currentDuty;

  // Access shared data safely
  critical_section_enter_blocking(&commStateLock);
  totalLux = sensorState.filteredLux;   // Current measured illuminance
  currentDuty = controlState.dutyCycle; // Current LED duty cycle
  float ledGain = deviceConfig.ledGain; // LED contribution factor
  critical_section_exit(&commStateLock);

  // Calculate LED's contribution using duty cycle and gain
  ledContribution = currentDuty * ledGain;

  // Subtract LED contribution from total measured illuminance
  float externalLux = totalLux - ledContribution;

  // Ensure we don't return negative values
  if (externalLux < 0.0f)
    externalLux = 0.0f;

  return externalLux;
}

/**
 * Adapt control system to external light changes
 * Uses a feedforward approach to assist the PID controller
 */
void adaptToExternalLight()
{
  static unsigned long lastAdaptTime = 0;
  static float previousExternal = -1.0;

  // Only check every 5 seconds to avoid rapid adjustments
  if (millis() - lastAdaptTime < 5000)
  {
    return;
  }
  lastAdaptTime = millis();

  // Get current external illuminance
  float externalLux = getExternalIlluminance();

  critical_section_enter_blocking(&commStateLock);
  bool feedback = controlState.feedbackControl;
  critical_section_exit(&commStateLock);
  if (previousExternal < 0 || !feedback)
  {
    previousExternal = externalLux;
    return;
  }

  // If external light has changed significantly (>1 lux)
  if (abs(externalLux - previousExternal) > EXTERNAL_LIGHT_CHANGE_THRESHOLD)
  {
    // Calculate how much of our setpoint is satisfied by external light
    critical_section_enter_blocking(&commStateLock);
    float externalContribution = min(externalLux, controlState.setpointLux);
    float requiredFromLED = max(0.0f, controlState.setpointLux - externalContribution);
    critical_section_exit(&commStateLock);

    // Pre-adjust duty cycle based on external light (feedforward control)
    float estimatedDuty = requiredFromLED / 30.0; // Assuming 30 lux at full power
    estimatedDuty = constrain(estimatedDuty, 0.0, 1.0);

    // Apply a small adjustment to help PID converge faster
    float currentDuty = getLEDDutyCycle();
    float newDuty = currentDuty * 0.7 + estimatedDuty * 0.3; // Gradual adjustment

    setLEDDutyCycle(newDuty);

    previousExternal = externalLux;
  }
}

//=============================================================================
// NEIGHBOR COORDINATION SUBSYSTEM
//=============================================================================

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
 * Calculate light contribution from neighboring luminaires
 *
 * @return Estimated illuminance contribution from neighbors (lux)
 */
float getNeighborsContribution()
{
  float totalContribution = 0.0;
  unsigned long currentTime = millis();
  const unsigned long NEIGHBOR_TIMEOUT = 10000; // 10 seconds timeout

  for (int i = 0; i < MAX_NEIGHBORS; i++)
  {
    if (neighbors[i].isActive)
    {
      // Mark inactive if too old
      if (currentTime - neighbors[i].lastUpdate > NEIGHBOR_TIMEOUT)
      {
        neighbors[i].isActive = false;
        continue;
      }

      // Skip neighbors that are off
      if (neighbors[i].state == STATE_OFF)
        continue;

      // Simple light contribution model - would need calibration in real deployment
      float contribution = neighbors[i].lastDuty * 3.0; // Each neighbor at 100% adds ~3 lux
      totalContribution += contribution;
    }
  }

  return totalContribution;
}

/**
 * Coordinate illuminance with neighbors to optimize energy usage
 * Adjusts PID target to account for light from neighboring nodes
 */
void coordinateWithNeighbors()
{
  // Calculate total neighbor light contribution
  float neighborContribution = getNeighborsContribution();

  if (neighborContribution > 0.5)
  { // Only adjust if contribution is significant
    // Adjust our target to account for light from neighbors
    critical_section_enter_blocking(&commStateLock);
    float adjustedTarget = max(0.0f, controlState.setpointLux - neighborContribution * 0.8);
    critical_section_exit(&commStateLock);

    // Dynamic PID adjustment based on cooperation
    pid.setTarget(adjustedTarget);
  }
}

//=============================================================================
// UTILITY FUNCTIONS
//=============================================================================

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
 * Calibrate illuminance model by measuring LED contribution
 * Measures illuminance with LED off and on to calculate system gain
 *
 * @return Calibrated gain value (y2-y1)
 */
float calibrateIlluminanceModel()
{
  const int SAMPLES = 5;               // Number of measurements to average
  const int STABILIZE_TIME = 500;      // Wait time between steady states in ms
  const int LED_RESPONSE_TIME = 10000; // Wait time for LDR to respond to LED changes in ms

  Serial.println("Calibrating illuminance model...");

  // Turn LED off and measure y1
  setLEDDutyCycle(0.0);
  delay(STABILIZE_TIME);

  // Take multiple measurements and average
  float y1 = 0.0;
  for (int i = 0; i < SAMPLES; i++)
  {
    y1 += readLux();
    delay(STABILIZE_TIME);
  }
  y1 /= SAMPLES;

  // Store in global variable for use in getExternalIlluminance()
  sensorState.baselineIlluminance = y1;

  Serial.print("Background illuminance (LED off): ");
  Serial.print(y1);
  Serial.println(" lux");

  // Turn LED to maximum and wait for LDR response
  setLEDDutyCycle(1.0);
  Serial.println("Waiting for LED and LDR to stabilize...");

  // Allow time for LED to reach full brightness and LDR to respond
  delay(LED_RESPONSE_TIME);

  // Take multiple measurements and average
  float y2 = 0.0;
  for (int i = 0; i < SAMPLES; i++)
  {
    y2 += readLux();
    delay(STABILIZE_TIME);
  }
  y2 /= SAMPLES;

  Serial.print("Total illuminance (LED on): ");
  Serial.print(y2);
  Serial.println(" lux");

  // Calculate gain: G = y2 - y1
  float gain = y2 - y1;

  Serial.print("Calibrated LED gain (G): ");
  Serial.println(gain);

  return gain;
}

/**
 * Perform comprehensive system calibration:
 * 1. Calibrate LDR sensor accuracy
 * 2. Measure LED contribution for external illuminance calculation
 *
 * @param referenceValue The reference illuminance value (typically very low like 1.0)
 * @return Calibrated LED gain value (G)
 */
float calibrateSystem(float referenceValue)
{
  const int SAMPLES = 5;               // Number of measurements to average
  const int STABILIZE_TIME = 500;      // Wait time between measurements in ms
  const int LED_RESPONSE_TIME = 10000; // Wait time for LDR to respond to LED changes

  Serial.println("Calibrating illuminance model...");

  // Turn LED off and measure y1
  setLEDDutyCycle(0.0);
  delay(STABILIZE_TIME);

  // Take multiple measurements and average
  float y1 = 0.0;
  for (int i = 0; i < SAMPLES; i++)
  {
    y1 += readLux(); // Using calibrated readings now
    delay(STABILIZE_TIME);
  }
  y1 /= SAMPLES;

  // Store baseline illuminance for external light calculation
  critical_section_enter_blocking(&commStateLock);
  sensorState.baselineIlluminance = y1;
  critical_section_exit(&commStateLock);

  Serial.print("Background illuminance (LED off): ");
  Serial.print(y1);
  Serial.println(" lux");

  // Turn LED to maximum and wait for LDR response
  setLEDDutyCycle(1.0);
  Serial.println("Waiting for LED and LDR to stabilize...");

  // Allow time for LED to reach full brightness and LDR to respond
  delay(LED_RESPONSE_TIME);

  // Take multiple measurements and average
  float y2 = 0.0;
  for (int i = 0; i < SAMPLES; i++)
  {
    y2 += readLux();
    delay(STABILIZE_TIME);
  }
  y2 /= SAMPLES;

  Serial.print("Total illuminance (LED on): ");
  Serial.print(y2);
  Serial.println(" lux");

  // Calculate gain: G = y2 - y1
  float gain = y2 - y1;

  Serial.print("Calibrated LED gain (G): ");
  Serial.println(gain);

  // Reset LED to off state after calibration
  setLEDDutyCycle(0.0);

  Serial.println("Comprehensive calibration complete!");
  return gain;
}


// NEW FUNCTIONALITY STILL TO BE VALIDATED
//=============================================================================
// NETWORK DISCOVERY HELPER FUNCTIONS
//=============================================================================

/**
 * Broadcast a discovery message to the network
 * Announces this node's presence to other nodes in the network
 * Uses a minimal CAN message to maximize network bandwidth efficiency
 * 
 * @return true if message was sent successfully
 */
bool broadcastDiscoveryMessage() {
  // Add a static counter with proper initialization
  static uint16_t broadcastCount = 0;
  
  // For discovery, we only need to send our node ID
  uint8_t myNodeId;
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  critical_section_exit(&commStateLock);

  // Increment counter BEFORE checking condition
  broadcastCount++;
  
  // Only print message every 100 broadcasts, avoiding the first broadcast
  if (broadcastCount % 100 == 0 && broadcastCount > 0) {
    Serial.print("Network: Broadcasting discovery message from node ");
    Serial.println(myNodeId);
  }
  
  // Use the value field to carry our node ID
  return sendControlCommand(CAN_ADDR_BROADCAST, CAN_DISC_HELLO, (float)myNodeId);
}
/**
 * Broadcast a ready message to the network
 * Indicates this node is ready for the calibration sequence
 * Sent after all expected nodes have been discovered
 * 
 * @return true if message was sent successfully
 */
bool broadcastReadyMessage() {
  // Get our node ID
  uint8_t myNodeId;
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  critical_section_exit(&commStateLock);
  
  Serial.print("Network: Broadcasting ready message from node ");
  Serial.println(myNodeId);
  
  // Pack node ID as the value so receivers know which node is ready
  // CAN_DISC_READY (0x31) is the message type for ready state indication
  return sendControlCommand(CAN_ADDR_BROADCAST, CAN_DISC_READY, (float)myNodeId);
}

/**
 * Broadcast a calibration message to the network
 * Controls the calibration sequence across nodes
 * Indicates which node should perform calibration next
 * 
 * @param sequenceIndex Index of the node that should perform calibration next
 * @return true if message was sent successfully
 */
bool broadcastCalibrationMessage(uint8_t sequenceIndex) {
  Serial.print("Network: Broadcasting calibration message, active node: ");
  Serial.println(sequenceIndex);
  
  // Send the sequence index so nodes know whose turn it is
  // CAN_DISC_CALIBRATION (0x32) is the message type for calibration coordination
  return sendControlCommand(CAN_ADDR_BROADCAST, CAN_DISC_CALIBRATION, (float)sequenceIndex);
}

/**
 * Initialize the network discovery and calibration system
 * Resets all network state variables and prepares for node discovery
 * Called during system startup to prepare for network coordination
 */
void initNetworkSystem() {
  // Set initial network state to boot mode
  NetworkState currentNetworkState = NET_STATE_BOOT;
  
  Serial.println("Network System: Initializing discovery and calibration subsystem...");
  
  // Reset discovered nodes tracking
  memset(discoveredNodes, 0, sizeof(discoveredNodes));
  discoveredCount = 0;
  Serial.println("Network System: Reset discovered nodes array");
  
  // Reset ready nodes tracking
  memset(readyNodes, 0, sizeof(readyNodes));
  readyCount = 0;
  Serial.println("Network System: Reset ready nodes array");
  
  // Reset calibration sequence counter and step
  calibrationNodeSequence = 0;
  calibrationStep = 0;
  Serial.println("Network System: Reset calibration sequence and step counters");
  
  // Record the current time as the starting point for state machine timing
  lastNetworkActionTime = millis();
  Serial.println("Network System: Recorded start time for state machine");
  
  // Mark calibration as incomplete to trigger the calibration process later
  isCalibrationComplete = false;
  Serial.println("Network System: Calibration marked as incomplete");
  
  // Ensure LED is off during initialization
  setLEDDutyCycle(0.0);
  Serial.println("Network System: LED turned off for initialization");
  
  // Mark our own node as discovered to include in count
  uint8_t myNodeId;
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  readyNodes[myNodeId] = false;  // Not ready yet
  discoveredNodes[myNodeId] = true;  // We know about ourselves
  critical_section_exit(&commStateLock);
  discoveredCount = 1;  // Start with just ourselves
  
  Serial.print("Network System: Initialized in state ");
  Serial.println("BOOT");
  Serial.println("Network System: Ready to begin discovery process");
}

/**
 * Process the network state machine
 * Handles network discovery, node coordination, and calibration
 */
void processNetworkStateMachine() {
  // Define the expected number of nodes in the network
  const uint8_t EXPECTED_NODES = 3;
  
  // Use global state for consistent tracking across function calls
  static NetworkState currentNetworkState = NET_STATE_BOOT;
  static unsigned long discoveryStartTime = 0;   // When we started discovery
  static unsigned long readyStartTime = 0;       // When node became ready
  static unsigned long lastBroadcastTime = 0;    // Last time we sent discovery message
  static bool hasSetNodeReady = false;           // Flag to track if we marked ourselves ready
  
  // Get current time for timing operations
  unsigned long currentTime = millis();
  
  // Get our node ID for logging and state management
  uint8_t myNodeId;
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  critical_section_exit(&commStateLock);
  
  // Process based on current network state
  switch (currentNetworkState) {
      //-------------------------------------------------------------------------
      // BOOT STATE
      //-------------------------------------------------------------------------
      case NET_STATE_BOOT:
          Serial.println("Network: Transitioning from BOOT to DISCOVERY state");
          
          // Mark our own node as discovered
          discoveredNodes[myNodeId] = true;
          discoveredCount = 1;  // Start with ourselves
          
          // Initialize discovery state variables
          discoveryStartTime = currentTime;
          lastBroadcastTime = 0;
          hasSetNodeReady = false;
          
          // Move to discovery state
          currentNetworkState = NET_STATE_DISCOVERY;
          lastNetworkActionTime = currentTime;
          break;
          
      //-------------------------------------------------------------------------
      // DISCOVERY STATE
      //-------------------------------------------------------------------------
      case NET_STATE_DISCOVERY: {
          // Periodically broadcast discovery (with rate limiting)
          // Add jitter (±50ms) to reduce collision probability
          if (currentTime - lastBroadcastTime > (500 + random(-50, 50))) {
              lastBroadcastTime = currentTime;
              
              // Static variables to track previous discovery state for change detection
              static uint8_t prevDiscoveredCount = 0;
              
              // Only log when the count changes to reduce serial spam
              if (prevDiscoveredCount != discoveredCount) {
                  Serial.print("Network: Broadcasting discovery, found ");
                  Serial.print(discoveredCount);
                  Serial.print(" of ");
                  Serial.print(EXPECTED_NODES);
                  Serial.println(" expected nodes");
                  
                  // Print known nodes
                  Serial.print("Network: Known nodes: ");
                  for (int i = 0; i < 64; i++) {
                      if (discoveredNodes[i]) {
                          Serial.print(i);
                          Serial.print(" ");
                      }
                  }
                  Serial.println();
                  
                  prevDiscoveredCount = discoveredCount;
                  
                  // Reset our overall state machine timer when progress is made
                  lastNetworkActionTime = currentTime;
              }
              
              // Send discovery broadcast
              broadcastDiscoveryMessage();
          }
          
          // Get our readiness status - check if we've already marked ourselves ready
          critical_section_enter_blocking(&commStateLock);
          bool nodeReadyStatus = readyNodes[myNodeId];
          critical_section_exit(&commStateLock);
          
          // Time-based checks for transitioning to ready state
          bool isDiscoveryTimedOut = (currentTime - discoveryStartTime) > 30000; // 30 seconds timeout
          bool foundAllNodes = (discoveredCount >= EXPECTED_NODES);
          
          // Conditions for marking ourselves ready:
          // 1. We found all expected nodes, OR
          // 2. Discovery has timed out (30 seconds)
          // 3. AND we haven't already marked ourselves ready
          if ((foundAllNodes || isDiscoveryTimedOut) && !hasSetNodeReady) {
              // Mark ourselves as ready
              readyNodes[myNodeId] = true;
              readyCount++;
              hasSetNodeReady = true;
              readyStartTime = currentTime;
              
              // Broadcast ready message to network
              broadcastReadyMessage();
              
              Serial.print("Network: Node ");
              Serial.print(myNodeId);
              Serial.println(" marked as ready for calibration");
              
              // If discovery timed out, log it clearly
              if (isDiscoveryTimedOut && !foundAllNodes) {
                  Serial.print("Network: Discovery timed out after 30 seconds. Found ");
                  Serial.print(discoveredCount);
                  Serial.print(" of ");
                  Serial.print(EXPECTED_NODES);
                  Serial.println(" expected nodes.");
              }
          }
          
          // Transition to READY state if:
          // 1. All nodes are ready, OR
          // 2. We've been in ready state for at least 10 seconds (transition timeout)
          if (readyCount >= EXPECTED_NODES || 
              (hasSetNodeReady && (currentTime - readyStartTime > 10000))) {
              Serial.println("Network: All nodes ready or timeout reached, transitioning to READY state");
              currentNetworkState = NET_STATE_READY;
              lastNetworkActionTime = currentTime;
          }
          break;
      }
          
      //-------------------------------------------------------------------------
      // READY STATE
      //-------------------------------------------------------------------------
      case NET_STATE_READY:
          // Wait a short time before starting calibration (5 seconds after entering READY)
          if (currentTime - lastNetworkActionTime > 5000) {
              Serial.println("Network: Ready timer expired, starting calibration sequence");
              
              // Call our function to start calibration
              startCalibration();
              
              // Transition to calibration state
              currentNetworkState = NET_STATE_CALIBRATION;
              Serial.println("Network: Transitioned to CALIBRATION state");
          }
          break;
          
      //-------------------------------------------------------------------------
      // CALIBRATION STATE
      //-------------------------------------------------------------------------
      case NET_STATE_CALIBRATION:
          // Call separate function to handle the calibration process
          processCalibration();
          
          // Check if calibration is complete
          if (isCalibrationComplete) {
              Serial.println("Network: Calibration complete, transitioning back to READY state");
              // Transition back to READY state
              currentNetworkState = NET_STATE_READY;
              lastNetworkActionTime = currentTime;
          }
          break;
          
      default:
          // Invalid state - reset to boot
          Serial.println("Network: Invalid state detected, resetting to BOOT");
          currentNetworkState = NET_STATE_BOOT;
          break;
  }
}

/**
 * Process the sequential LED calibration
 * Each node takes turns turning its LED on and all nodes measure the effect
 * 
 * Calibration Steps for Each Node:
 * 0: Turn all LEDs off
 * 1: Measure baseline illuminance
 * 2: Current node turns its LED on
 * 3: All nodes measure the effect (self-gain or cross-gain)
 * 4: Move to next node in sequence
 */
void processCalibration() {
  // Calibration timing constants
  const unsigned long STABILIZE_TIME = 2000;    // Time for readings to stabilize (ms)
  const unsigned long MEASURE_TIME = 3000;      // Time to take measurements (ms)
  const unsigned long LED_ON_TIME = 5000;       // Time to keep LED on (ms)
  const unsigned long TRANSITION_TIME = 1000;   // Time between nodes (ms)
  
  // Get current time
  unsigned long currentTime = millis();
  
  // Static variable to track illuminance measurements across function calls
  static float baselineLux = 0.0;      // Illuminance with all LEDs off
  static unsigned long stepStartTime = 0; // When the current step started
  static unsigned long currentStepDuration = 0; // Duration of current step
  static bool stepInitialized = false; // Flag to track step initialization
  
  // Get our node ID for comparison
  uint8_t myNodeId;
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  critical_section_exit(&commStateLock);
  
  // Map sequence index to actual nodeId
  static uint8_t nodeIdSequence[64]; // Array to store the actual node IDs in sequence
  static bool sequenceInitialized = false;
  
  if (!sequenceInitialized) {
    // Build the sequence of actual node IDs from discoveredNodes
    int seqIndex = 0;
    for (int i = 0; i < 64; i++) {
      if (discoveredNodes[i]) {
        nodeIdSequence[seqIndex++] = i;
      }
    }
    sequenceInitialized = true;
  }
  
  // Get the actual node ID for current sequence position
  uint8_t activeNodeId = nodeIdSequence[calibrationNodeSequence];
  
  // Initialize step if needed
  if (!stepInitialized) {
    stepStartTime = currentTime;
    
    // Set appropriate duration for each step
    switch (calibrationStep) {
      case 0: // Turn LEDs off
        currentStepDuration = TRANSITION_TIME;
        Serial.println("Calibration: Turning all LEDs off to establish baseline");
        break;
      case 1: // Measure baseline
        currentStepDuration = MEASURE_TIME;
        Serial.println("Calibration: Moving to step 1 - Measure baseline");
        break;
      case 2: // Turn active node LED on
        currentStepDuration = LED_ON_TIME;
        Serial.println("Calibration: Moving to step 2 - Active node turns LED on");
        if (myNodeId == activeNodeId) {
          Serial.println("Calibration: This node is active, turning LED on");
          setLEDDutyCycle(1.0);
        } else {
          Serial.print("Calibration: Waiting for Node ");
          Serial.print(activeNodeId);
          Serial.println(" to turn on its LED");
        }
        break;
      case 3: // Measure effect
        currentStepDuration = MEASURE_TIME;
        Serial.println("Calibration: Moving to step 3 - Measure effect");
        break;
      case 4: // Move to next node
        currentStepDuration = TRANSITION_TIME;
        Serial.println("Calibration: Moving to step 4 - Advance to next node");
        break;
    }
    
    stepInitialized = true;
  }
  
  // Check if it's time to move to the next step
  if (currentTime - stepStartTime > currentStepDuration) {
    // Process the current step
    switch (calibrationStep) {
      case 0: // Turn LEDs off
        // All nodes turn LED off
        setLEDDutyCycle(0.0);
        calibrationStep = 1; // Advance to next step
        break;
        
      case 1: { // Measure baseline
        // All nodes measure baseline illuminance
        const int SAMPLES = 5;  // Number of samples to average
        float totalLux = 0.0;
        
        for (int i = 0; i < SAMPLES; i++) {
          totalLux += readLux();
          delay(50); // Short delay between samples
        }
        baselineLux = totalLux / SAMPLES;
        
        Serial.print("Calibration: Baseline lux (ambient light): ");
        Serial.println(baselineLux, 2);
        
        // Send baseline to all nodes using sensor type 0 (illuminance)
        sendSensorReading(CAN_ADDR_BROADCAST, 0, baselineLux);
        
        calibrationStep = 2; // Advance to next step
        break;
      }
        
      case 2: // Turn active node LED on
        // Nothing to do at end of step - LED should already be on or off
        calibrationStep = 3; // Advance to next step
        break;
        
      case 3: { // Measure effect
        // All nodes measure illuminance with active LED on
        const int SAMPLES = 5;
        float totalLux = 0.0;
        
        for (int i = 0; i < SAMPLES; i++) {
          totalLux += readLux();
          delay(50);
        }
        float measuredLux = totalLux / SAMPLES;
        
        Serial.print("Calibration: Measured illuminance with LED: ");
        Serial.println(measuredLux, 2);
        
        // Calculate gain (current - baseline)
        float gainValue = measuredLux - baselineLux;
        
        // If this is our own node, it's self-gain
        if (myNodeId == activeNodeId) {
          selfGain = gainValue;
          Serial.print("Calibration: Self-gain for node ");
          Serial.print(myNodeId);
          Serial.print(": ");
          Serial.println(selfGain, 2);
          
          // Store in device config (for future use)
          critical_section_enter_blocking(&commStateLock);
          deviceConfig.ledGain = selfGain;
          critical_section_exit(&commStateLock);
        } else {
          // Otherwise it's cross-gain
          crossGains[activeNodeId] = gainValue;
          Serial.print("Calibration: Cross-gain from node ");
          Serial.print(activeNodeId);
          Serial.print(" to node ");
          Serial.print(myNodeId);
          Serial.print(": ");
          Serial.println(gainValue, 2);
          
          // Send this cross-gain to the active node using sensor type 10
          // (using type 10 specifically for cross-gain reporting)
          sendSensorReading(activeNodeId, 10, gainValue);
        }
        
        calibrationStep = 4; // Advance to next step
        break;
      }
        
      case 4: // Move to next node
        // Turn off LED regardless of which node we are
        setLEDDutyCycle(0.0);
        
        // Broadcast calibration message for next node
        if (myNodeId == activeNodeId) {
          // Only the active node advances the sequence
          // Increment to next node
          calibrationNodeSequence++;
          
          if (calibrationNodeSequence >= discoveredCount) {
            // We've processed all nodes, calibration is complete
            isCalibrationComplete = true;
            Serial.println("Calibration: Completed for all nodes!");
            
            // Ensure LED is off at the end
            setLEDDutyCycle(0.0);
          } else {
            // Move to the next node in sequence
            uint8_t nextNodeId = nodeIdSequence[calibrationNodeSequence];
            Serial.print("Calibration: Moving to node ");
            Serial.println(calibrationNodeSequence);
            
            // Broadcast to coordinate all nodes
            broadcastCalibrationMessage(calibrationNodeSequence);
            
            Serial.print("Calibration: Preparing for calibration of node ");
            Serial.println(nextNodeId);
          }
        }
        
        // Reset to step 0 for next node (all nodes do this)
        calibrationStep = 0;
        break;
    }
    
    // Reset step initialization flag for the new step
    stepInitialized = false;
  }
}

/**
 * Initiate the network-wide calibration sequence
 * Transitions the network state to calibration mode and starts the process
 * Coordinates all nodes to take turns measuring light contribution
 */
void startCalibration() {
  // Reset calibration variables
  calibrationNodeSequence = 0; // Start with the first node (node 0)
  calibrationStep = 0; // Reset step to beginning (turn all LEDs off)
  isCalibrationComplete = false;
  
  // Store current time for timing operations
  lastNetworkActionTime = millis();
  
  Serial.println("Network: Starting calibration sequence");
  
  // Broadcast the initial calibration message to synchronize all nodes
  // This tells node 0 to start its calibration turn
  broadcastCalibrationMessage(0);
  
  // Reset all LED states to ensure a consistent starting point
  setLEDDutyCycle(0.0);
  
  // Set global variable to make currentNetworkState accessible
  extern NetworkState currentNetworkState;
  
  Serial.println("Network: Calibration initiated, waiting for steps to complete");
}

// END OF NETWORK DISCOVERY HELPER FUNCTIONS IMPLEMENTATION TO BE VALIDATED

/******************************************************************************************************************************************************************************************************************************************************************************** */
// LED DRIVER SECTION
/******************************************************************************************************************************************************************************************************************************************************************************** */

//-----------------------------------------------------------------------------
// CONFIGURATION AND VARIABLES
//-----------------------------------------------------------------------------

// Static module variables
static int ledPin = -1;   // GPIO pin connected to the LED
static int pwmMax = 4095; // Maximum PWM value (12-bit resolution)
static int pwmMin = 0;    // Minimum PWM value (off)

// PWM configuration constants
const unsigned int PWM_FREQUENCY = 30000; // 30 kHz frequency

//-----------------------------------------------------------------------------
// INITIALIZATION FUNCTIONS
//-----------------------------------------------------------------------------

/**
 * Initialize the LED driver with the specified GPIO pin
 * Configures PWM parameters and sets initial state to off
 *
 * @param pin GPIO pin number connected to the LED
 */
void initLEDDriver(int pin)
{
  // Store pin and configure it as output
  ledPin = pin;
  pinMode(ledPin, OUTPUT);

  // Configure PWM with optimal settings
  // - Sets resolution to 12-bit (0-4095)
  // - Sets frequency to 30kHz for flicker-free operation
  analogWriteRange(pwmMax);
  analogWriteFreq(PWM_FREQUENCY);

  // Start with LED off
  analogWrite(ledPin, pwmMin);
  critical_section_enter_blocking(&commStateLock);
  controlState.dutyCycle = 0.0; // Use the global duty cycle variable
  critical_section_exit(&commStateLock);
}

//-----------------------------------------------------------------------------
// LED CONTROL FUNCTIONS
//-----------------------------------------------------------------------------

/**
 * Set LED brightness using PWM duty cycle (0.0 to 1.0)
 * This is the primary control function that other methods call
 *
 * @param newDutyCycle Duty cycle value between 0.0 (off) and 1.0 (fully on)
 */
void setLEDDutyCycle(float newDutyCycle)
{
  // Validate and constrain input
  if (isnan(newDutyCycle) || isinf(newDutyCycle))
  {
    return; // Protect against invalid inputs
  }

  // Constrain to valid range
  newDutyCycle = constrain(newDutyCycle, 0.0f, 1.0f);

  // Apply duty cycle by converting to appropriate PWM value
  int pwmValue = (int)(newDutyCycle * pwmMax);
  analogWrite(ledPin, pwmValue);

  // Update the global duty cycle
  critical_section_enter_blocking(&commStateLock);
  controlState.dutyCycle = newDutyCycle;
  critical_section_exit(&commStateLock);
}

/**
 * Set LED brightness using percentage (0% to 100%)
 * Converts percentage to duty cycle and calls setLEDDutyCycle
 *
 * @param percentage Brightness percentage between 0.0 (off) and 100.0 (fully on)
 */
void setLEDPercentage(float percentage)
{
  // Constrain to valid percentage range
  percentage = constrain(percentage, 0.0f, 100.0f);

  // Convert percentage to duty cycle
  float newDutyCycle = percentage / 100.0f;

  // Set the LED using the calculated duty cycle
  setLEDDutyCycle(newDutyCycle);
}

/**
 * Set LED brightness using direct PWM value (0 to pwmMax)
 * Bypasses duty cycle calculation for direct hardware control
 *
 * @param pwmValue PWM value between 0 (off) and pwmMax (fully on)
 */
void setLEDPWMValue(int pwmValue)
{
  // Constrain to valid PWM range
  pwmValue = constrain(pwmValue, pwmMin, pwmMax);

  // Apply PWM value directly
  analogWrite(ledPin, pwmValue);

  // Update global duty cycle to maintain state consistency
  critical_section_enter_blocking(&commStateLock);
  controlState.dutyCycle = (float)pwmValue / pwmMax;
  critical_section_exit(&commStateLock);
}

/**
 * Set LED brightness based on desired power consumption
 * Maps power in watts to appropriate duty cycle
 *
 * @param powerWatts Desired power in watts from 0.0 to MAX_POWER_WATTS
 */
void setLEDPower(float powerWatts)
{
  // Constrain to valid power range
  powerWatts = constrain(powerWatts, 0.0f, MAX_POWER_WATTS);

  // Convert power to duty cycle (assumes linear relationship)
  float newDutyCycle = powerWatts / MAX_POWER_WATTS;

  // Set the LED using the calculated duty cycle
  setLEDDutyCycle(newDutyCycle);
}

//-----------------------------------------------------------------------------
// LED STATUS QUERY FUNCTIONS
//-----------------------------------------------------------------------------

/**
 * Get current LED duty cycle setting
 *
 * @return Current duty cycle value (0.0 to 1.0)
 */
float getLEDDutyCycle()
{
  return controlState.dutyCycle;
}

/**
 * Get current LED brightness as percentage
 *
 * @return Current brightness percentage (0.0 to 100.0)
 */
float getLEDPercentage()
{
  return controlState.dutyCycle * 100.0f;
}

/**
 * Get current LED PWM value
 *
 * @return Current PWM value (0 to pwmMax)
 */
int getLEDPWMValue()
{
  return (int)(controlState.dutyCycle * pwmMax);
}

/**
 * Get estimated current LED power consumption
 *
 * @return Estimated power consumption in watts
 */
float getLEDPower()
{
  return controlState.dutyCycle * MAX_POWER_WATTS;
}

/**
 * Get historical data buffer as CSV string
 *
 * @param var Variable type (y=illuminance, u=duty cycle)
 * @param index Node index
 * @return CSV string of historical values
 */
void getLastMinuteBuffer(const char* var, int index, char* buffer, size_t bufferSize)
{
  String result = "";
  int count = getLogCount();
  if (count == 0)
  {
    strncpy(buffer, "No data available", bufferSize - 1);
    buffer[bufferSize - 1] = '\0';
    return;
  }

  LogEntry *logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;

  // Maximum number of samples to return (to avoid overflowing serial buffer)
  const int MAX_SAMPLES = 60;
  int sampleCount = min(count, MAX_SAMPLES);

  // Calculate step to get evenly distributed samples
  int step = count > MAX_SAMPLES ? count / MAX_SAMPLES : 1;

  for (int i = 0; i < count; i += step)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;

    if (strcmp(var, "y") == 0)
    {
      // For illuminance values
      result += String(logBuffer[realIndex].lux, 1);
    }
    else if (strcmp(var, "u") == 0)
    {
      // For duty cycle values
      result += String(logBuffer[realIndex].duty, 3);
    }

    if (i + step < count)
    {
      result += ",";
    }
  }
  strncpy(buffer, result.c_str(), bufferSize - 1);
  buffer[bufferSize - 1] = '\0'; // Ensure null termination
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
  delay(100); 

  // Configure hardware
  analogReadResolution(12);
  analogWriteFreq(30000);
  analogWriteRange(PWM_MAX);

  // Initialize subsystems
  initLEDDriver(LED_PIN);
  initStorage();
  initCANComm();

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

  // Configure device based on unique ID
  configureDeviceFromID();

  // Calibrate the system
  deviceConfig.ledGain = calibrateSystem(1.0);

  // Launch core 0 for CAN communication
  multicore_launch_core1(core0_main);

  initPendingQueries();

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

  // Process network discovery and calibration state machine
  processNetworkStateMachine();

  // (B) Handle streaming
  handleStreaming();
  handleRemoteStreamRequests();

  // (C) Read sensor data safely
  float lux = readLux();

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

void applyDeviceConfig(const DeviceConfig &config)
{
  deviceConfig = config; // Store the configuration

  // Apply settings
  nodeID = config.nodeId;

  // Configure PID controller
  pid.setGains(config.pidKp, config.pidKi);
  pid.setWeighting(config.pidBeta);

  Serial.print("Configured as device #");
  Serial.print(config.nodeId);
  Serial.print(", gain=");
  Serial.print(config.ledGain);
  Serial.print(", Kp=");
  Serial.print(config.pidKp);
  Serial.print(", Ki=");
  
  Serial.println(config.pidKi);
}

void configureDeviceFromID()
{
  // Read hardware ID
  pico_unique_board_id_t board_id;
  pico_get_unique_board_id(&board_id);
  
  // Generate unique node ID from the last byte (1-63)
  uint8_t nodeId = board_id.id[7] & 0x3F; // Use last 6 bits for node ID (0-63)
  
  // Avoid broadcast address (0)
  if (nodeId == 0)
    nodeId = 1;
    
  // Print full board ID for debugging
  Serial.print("Board ID: ");
  for (int i = 0; i < 8; i++) {
    if (board_id.id[i] < 16) Serial.print("0");
    Serial.print(board_id.id[i], HEX);
    if (i < 7) Serial.print(":");
  }
  Serial.print(" -> NodeID: ");
  Serial.println(nodeId);

  // Determine device type (0, 1, or 2) based on a hash of the full board ID
  uint32_t idHash = 0;
  for (int i = 0; i < 8; i++) {
    idHash = ((idHash << 5) + idHash) + board_id.id[i]; // Simple hash function
  }
  uint8_t deviceType = idHash % 3; // Still 0, 1, or 2 for PID configuration
  
  Serial.print("Device Type: ");
  Serial.println(deviceType);

  // Start with default configuration
  DeviceConfig config;
  
  // IMPORTANT: Always use the unique node ID
  config.nodeId = nodeId;
  
  // Apply device-type specific PID parameters
  switch (deviceType)
  {
  case 0: // First device type
    config.pidKp = 20.0;
    config.pidKi = 400.0;
    config.pidBeta = 0.8;
    break;

  case 1: // Second device type
    config.pidKp = 25.0;
    config.pidKi = 350.0;
    config.pidBeta = 0.75;
    break;

  case 2: // Third device type
    config.pidKp = 22.0;
    config.pidKi = 380.0;
    config.pidBeta = 0.82;
    break;
  }

  // Apply the configuration
  applyDeviceConfig(config);
}

// Core 0: CAN Communication
void core0_main()
{
  while (true)
  {
    // Process CAN messages
    canCommLoop();
    CoreMessage msg;
    if (queue_try_remove(&core1to0queue, &msg))
    {
      switch (msg.msgType)
      {
      case MSG_SEND_SENSOR:
        sendSensorReading(msg.nodeId, msg.dataType, msg.value);
        break;
      case MSG_SEND_CONTROL:
        sendControlCommand(msg.nodeId, msg.dataType, msg.value);
        break;
      case MSG_UPDATE_STATE:
        // Handle state update
        break;
      }
    }
    // Brief delay to prevent core hogging
    sleep_ms(1);
  }
}