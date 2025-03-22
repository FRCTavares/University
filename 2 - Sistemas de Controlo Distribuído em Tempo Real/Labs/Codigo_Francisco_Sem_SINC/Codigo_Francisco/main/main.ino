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

//=============================================================================
// INCLUDES AND DEPENDENCIES
//=============================================================================
#include <Arduino.h>
#include <math.h>
#include "PIController.h"
#include "Storage.h"
#include "Metrics.h"
#include "CommandInterface.h"
#include "CANComm.h"
#include "pico/multicore.h"
#include "LEDDriver.h"
#include "Globals.h"

//=============================================================================
// HARDWARE CONFIGURATION
//=============================================================================

//-----------------------------------------------------------------------------
// Sensor Configuration
//-----------------------------------------------------------------------------
#define VCC 3.3                // Supply voltage for analog reference
#define MY_ADC_RESOLUTION 4095.0  // 12-bit ADC resolution
#define FIXED_RESISTOR 10000.0  // Fixed resistor in voltage divider (ohms)

// LDR Calibration parameters (for lux conversion)
const float R10 = 225000.0;   // LDR resistance at ~10 lux (ohms)
const float LDR_M = -1.0;     // Slope of log-log resistance vs. illuminance
float LDR_B = log10(R10) - LDR_M;  // Y-intercept for log-log conversion

//-----------------------------------------------------------------------------
// Pin Assignments
//-----------------------------------------------------------------------------
const int LED_PIN = 15;       // PWM output for LED driver
const int LDR_PIN = A0;       // Analog input for light sensor

//-----------------------------------------------------------------------------
// PWM Configuration
//-----------------------------------------------------------------------------
const int PWM_MAX = 4095;     // Maximum PWM value (12-bit)
const int PWM_MIN = 0;        // Minimum PWM value (off)

//-----------------------------------------------------------------------------
// Measurement Filter Configuration
//-----------------------------------------------------------------------------
const int NUM_SAMPLES = 10;           // Samples for averaging
const float OUTLIER_THRESHOLD = 2.0;  // Standard deviations for outlier detection
const float ALPHA = 0.3;              // EMA filter coefficient (0=slow, 1=fast)

//-----------------------------------------------------------------------------
// PID Controller Parameters
//-----------------------------------------------------------------------------
const float KP = 230.0;        // Proportional gain
const float KI = 230.0;       // Integral gain
const float DT = 0.01;        // Sampling period (seconds)
const float BETA = 0.5;       // Setpoint weighting factor (0.0-1.0)

//-----------------------------------------------------------------------------
// Power Consumption Model
//-----------------------------------------------------------------------------
const float MAX_POWER_WATTS = 1.0;  // Maximum power consumption at 100% duty

//=============================================================================
// GLOBAL SYSTEM STATE
//=============================================================================

//-----------------------------------------------------------------------------
// Control System State
//-----------------------------------------------------------------------------
float setpointLux = 10.0;     // Target illuminance (lux)
float dutyCycle = 0.0;        // Current LED duty cycle [0.0-1.0]
float refIlluminance = 15.0;  // Reference illuminance for metrics
bool occupancy = false;       // Room occupancy state
bool antiWindup = false;      // Anti-windup feature for PID
bool feedbackControl = true;  // Feedback control mode (vs. manual)

// Illuminance setpoints for different states
const float SETPOINT_OFF = 0.0;        // Off state target (lux)
const float SETPOINT_UNOCCUPIED = 5.0; // Unoccupied state target (lux)
const float SETPOINT_OCCUPIED = 15.0;  // Occupied state target (lux)

// Current luminaire operating state
LuminaireState luminaireState = STATE_UNOCCUPIED;

//-----------------------------------------------------------------------------
// Sensor State
//-----------------------------------------------------------------------------
float calibrationOffset = 0.0;  // Lux sensor calibration offset
float lastFilteredLux = -1.0;   // Last filtered lux value for EMA

//-----------------------------------------------------------------------------
// External Light Tracking
//-----------------------------------------------------------------------------
float lastExternalLux = 0.0;    // Previous external light measurement
float externalLuxAverage = 0.0; // Smoothed external illuminance
const float EXT_LUX_ALPHA = 0.05; // Slow-moving average coefficient

//-----------------------------------------------------------------------------
// CAN Communication State
//-----------------------------------------------------------------------------
bool periodicCANEnabled = false;        // Enable periodic transmissions
bool canMonitorEnabled = false;         // Display received messages
uint8_t nodeID = 0;                     // Node ID (set during initialization)
unsigned long lastCANSend = 0;          // Last transmission timestamp
unsigned long lastHeartbeat = 0;        // Last heartbeat timestamp
unsigned long heartbeatInterval = 5000; // Heartbeat interval (ms)

//-----------------------------------------------------------------------------
// External Light Tracking
//-----------------------------------------------------------------------------
float ledGain = 25.0;           // Gain value G for LED contribution (calibrated at startup)

float baselineIlluminance = 0.0;  // Illuminance measured with LED off

//-----------------------------------------------------------------------------
// Streaming & Data Logging
//-----------------------------------------------------------------------------
bool streamingEnabled = false;      // Enable serial data streaming
String streamingVar = "";           // Variable to stream
int streamingIndex = 0;             // Node index for streaming
unsigned long lastStreamTime = 0;   // Last stream update timestamp

//-----------------------------------------------------------------------------
// Debug Configuration
//-----------------------------------------------------------------------------
bool DEBUG_MODE = false;     // Master debug switch
bool DEBUG_LED = false;      // LED driver debug messages
bool DEBUG_SENSOR = false;   // Sensor readings debug
bool DEBUG_PI = false;      // PI control debug messages
bool DEBUG_PLOTTING = false; // Serial plotter output

//-----------------------------------------------------------------------------
// Neighbor Management
//-----------------------------------------------------------------------------
struct NeighborInfo {
  uint8_t nodeId;             // CAN node ID
  float lastLux;              // Last reported illuminance
  float lastDuty;             // Last reported duty cycle
  LuminaireState state;       // Current operating state
  unsigned long lastUpdate;   // Last update timestamp
  bool isActive;              // Is node currently active
};

const int MAX_NEIGHBORS = 5;  // Maximum tracked neighbors
NeighborInfo neighbors[MAX_NEIGHBORS];  // Neighbor state array

//-----------------------------------------------------------------------------
// Controller Object
//-----------------------------------------------------------------------------
PIController pid(KP, KI, 1.0, DT);  // PI controller instance with Beta=1.0

//=============================================================================
// STATE MANAGEMENT SUBSYSTEM
//=============================================================================

/**
 * Change luminaire operating state and update settings accordingly
 * 
 * @param newState The target state (OFF, UNOCCUPIED, OCCUPIED)
 */
void changeState(LuminaireState newState) {
  // Don't do anything if state is unchanged
  if (newState == luminaireState) {
    return;
  }

  luminaireState = newState;

  // Update setpoint based on new state
  switch (luminaireState) {
    case STATE_OFF:
      setpointLux = SETPOINT_OFF;
      feedbackControl = false; // Turn off control when lights are off
      break;

    case STATE_UNOCCUPIED:
      setpointLux = SETPOINT_UNOCCUPIED;
      feedbackControl = true;
      break;

    case STATE_OCCUPIED:
      setpointLux = SETPOINT_OCCUPIED;
      feedbackControl = true;
      break;
  }

  // Reset PID controller to avoid integral windup during transitions
  pid.reset();

  // Update reference illuminance for metrics calculation
  refIlluminance = setpointLux;

  // Broadcast state change to network
  sendControlCommand(CAN_ADDR_BROADCAST, CAN_CTRL_STATE_CHANGE, (float)luminaireState);
}

//=============================================================================
// SENSOR SUBSYSTEM
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
float readLux() {
  float samples[NUM_SAMPLES];
  float sum = 0.0;
  float count = 0.0;

  // 1. Take multiple samples to reduce noise
  for (int i = 0; i < NUM_SAMPLES; i++) {
    int adcValue = analogRead(LDR_PIN);
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;

    // Skip invalid readings
    if (voltage <= 0.0) {
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

  // 2. Calculate mean and standard deviation
  float mean = sum / count;
  float variance = 0.0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    if (samples[i] > 0) { // Only consider valid samples
      variance += sq(samples[i] - mean);
    }
  }
  variance /= count;
  float stdDev = sqrt(variance);

  // 3. Filter outliers and recalculate mean
  float filteredSum = 0.0;
  float filteredCount = 0.0;

  for (int i = 0; i < NUM_SAMPLES; i++) {
    if (samples[i] > 0 && abs(samples[i] - mean) <= OUTLIER_THRESHOLD * stdDev) {
      filteredSum += samples[i];
      filteredCount++;
    }
  }

  float filteredMean = (filteredCount > 0) ? filteredSum / filteredCount : mean;

  // 4. Apply exponential moving average (EMA) filter for temporal smoothing
  if (lastFilteredLux < 0) {
    lastFilteredLux = filteredMean; // First valid reading
  } else {
    lastFilteredLux = ALPHA * filteredMean + (1.0 - ALPHA) * lastFilteredLux;
  }

  // 5. Apply calibration offset and safety bounds check
  float calibratedLux = lastFilteredLux + calibrationOffset;
  if (calibratedLux < 0.0)
    calibratedLux = 0.0;

  return calibratedLux;
}

/**
 * Calibrate LUX sensor using a reference measurement
 * 
 * @param knownLux Reference illuminance from trusted external meter
 */
void calibrateLuxSensor(float knownLux) {
  float measuredLux = 0.0;
  const int CAL_SAMPLES = 10;

  for (int i = 0; i < CAL_SAMPLES; i++) {
    // Use a special raw reading to avoid existing calibration
    int adcValue = analogRead(LDR_PIN);
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
    if (voltage <= 0.0)
      continue;

    float resistance = FIXED_RESISTOR * (VCC / voltage - 1.0);
    float logR = log10(resistance);
    float logLux = (logR - LDR_B) / LDR_M;
    float rawLux = pow(10, logLux);

    measuredLux += rawLux;
    delay(50); // Short delay between readings
  }
  measuredLux /= CAL_SAMPLES;

  // Calculate the offset needed
  calibrationOffset = knownLux - measuredLux;

  Serial.print("Sensor calibrated: offset = ");
  Serial.println(calibrationOffset);
}

/**
 * Get raw voltage at LDR sensing pin
 * 
 * @return Voltage at LDR pin (0-VCC)
 */
float getVoltageAtLDR() {
  int adcValue = analogRead(LDR_PIN);
  return (adcValue / MY_ADC_RESOLUTION) * VCC;
}

/**
 * Calculate external illuminance by subtracting LED contribution
 * Uses calibrated gain value G to determine LED contribution
 * 
 * @return Estimated external illuminance in lux
 */
float getExternalIlluminance() {
  float measuredLux = readLux();
  
  // Remove baseline offset
  float baselineOffset = baselineIlluminance;  // Use the global variable
  
  // Linear model using calibrated gain: d = y - G*u - baseline
  float ledContribution = dutyCycle * ledGain;
  
  // Calculate current external lux estimate
  float currentExternalLux = max(0.0f, measuredLux - ledContribution - baselineOffset);
  
  // Rest of function stays the same
  if (lastExternalLux == 0.0) {
    externalLuxAverage = currentExternalLux;
  } else {
    externalLuxAverage = EXT_LUX_ALPHA * currentExternalLux + 
                        (1.0 - EXT_LUX_ALPHA) * externalLuxAverage;
  }
  
  lastExternalLux = currentExternalLux;
  return externalLuxAverage;
}

/**
 * Adapt control system to external light changes
 * Uses a feedforward approach to assist the PID controller
 */
void adaptToExternalLight() {
  static unsigned long lastAdaptTime = 0;
  static float previousExternal = -1.0;

  // Only check every 5 seconds to avoid rapid adjustments
  if (millis() - lastAdaptTime < 5000) {
    return;
  }
  lastAdaptTime = millis();

  // Get current external illuminance
  float externalLux = getExternalIlluminance();

  // Skip first run or when in manual mode
  if (previousExternal < 0 || !feedbackControl) {
    previousExternal = externalLux;
    return;
  }

  // If external light has changed significantly (>1 lux)
  if (abs(externalLux - previousExternal) > 1.0) {
    // Calculate how much of our setpoint is satisfied by external light
    float externalContribution = min(externalLux, setpointLux);
    float requiredFromLED = max(0.0f, setpointLux - externalContribution);

    // Pre-adjust duty cycle based on external light (feedforward control)
    float estimatedDuty = requiredFromLED / 30.0; // Assuming 30 lux at full power
    estimatedDuty = constrain(estimatedDuty, 0.0, 1.0);

    // Apply a small adjustment to help PID converge faster
    float currentDuty = getLEDDutyCycle();
    float newDuty = currentDuty * 0.7 + estimatedDuty * 0.3; // Gradual adjustment

    setLEDDutyCycle(newDuty);

    if (DEBUG_MODE && DEBUG_SENSOR) {
      Serial.print("External light adaptation: ");
      Serial.print(externalLux);
      Serial.print(" lux, required from LED: ");
      Serial.print(requiredFromLED);
      Serial.print(" lux, adjusted duty: ");
      Serial.println(newDuty, 3);
    }

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
void updateNeighborInfo(uint8_t nodeId, uint8_t sensorType, float value) {
  int emptySlot = -1;

  // Find existing neighbor or empty slot
  for (int i = 0; i < MAX_NEIGHBORS; i++) {
    if (neighbors[i].isActive && neighbors[i].nodeId == nodeId) {
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

    if (!neighbors[i].isActive && emptySlot < 0) {
      emptySlot = i;
    }
  }

  // Add as new neighbor if slot available
  if (emptySlot >= 0) {
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
float getNeighborsContribution() {
  float totalContribution = 0.0;
  unsigned long currentTime = millis();
  const unsigned long NEIGHBOR_TIMEOUT = 10000; // 10 seconds timeout

  for (int i = 0; i < MAX_NEIGHBORS; i++) {
    if (neighbors[i].isActive) {
      // Mark inactive if too old
      if (currentTime - neighbors[i].lastUpdate > NEIGHBOR_TIMEOUT) {
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
void coordinateWithNeighbors() {
  // Calculate total neighbor light contribution
  float neighborContribution = getNeighborsContribution();

  if (neighborContribution > 0.5) { // Only adjust if contribution is significant
    // Adjust our target to account for light from neighbors
    float adjustedTarget = max(0.0f, setpointLux - neighborContribution * 0.8);

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
float getPowerConsumption() {
  return dutyCycle * MAX_POWER_WATTS;
}

/**
 * Get elapsed time since boot
 * 
 * @return Time in seconds
 */
unsigned long getElapsedTime() {
  return millis() / 1000;
}

/**
 * Run a quick LED test sequence
 * Ramps brightness up and down to verify hardware
 */
void testLED() {
  // Quick test of LED by ramping up and down
  Serial.println("Testing LED...");

  for (int i = 0; i <= 100; i += 10) {
    setLEDPercentage(i);
    delay(50);
  }

  for (int i = 100; i >= 0; i -= 10) {
    setLEDPercentage(i);
    delay(50);
  }

  // Set LED to off after test
  setLEDDutyCycle(0.0);
  Serial.println("LED test complete.");
}

/**
 * Calibrate illuminance model by measuring LED contribution
 * Measures illuminance with LED off and on to calculate system gain
 * 
 * @return Calibrated gain value (y2-y1)
 */
float calibrateIlluminanceModel() {
  const int SAMPLES = 5;         // Number of measurements to average
  const int STABILIZE_TIME = 500; // Wait time between steady states in ms
  const int LED_RESPONSE_TIME = 10000; // Wait time for LDR to respond to LED changes in ms
  
  Serial.println("Calibrating illuminance model...");
  
  // Turn LED off and measure y1
  setLEDDutyCycle(0.0);
  delay(STABILIZE_TIME);
  
  // Take multiple measurements and average
  float y1 = 0.0;
  for (int i = 0; i < SAMPLES; i++) {
    y1 += readLux();
    delay(STABILIZE_TIME);
  }
  y1 /= SAMPLES;
  
  // Store in global variable for use in getExternalIlluminance()
  baselineIlluminance = y1;
  
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
  for (int i = 0; i < SAMPLES; i++) {
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
float calibrateSystem(float referenceValue) {
  const int SAMPLES = 5;               // Number of measurements to average
  const int STABILIZE_TIME = 500;      // Wait time between measurements in ms
  const int LED_RESPONSE_TIME = 10000; // Wait time for LDR to respond to LED changes
  
  Serial.println("Starting comprehensive calibration...");
  
  //---------------------------------------------------------------------
  // 1. First calibrate the LDR sensor for accurate absolute readings
  //---------------------------------------------------------------------
  float measuredLux = 0.0;
  const int CAL_SAMPLES = 10;

  Serial.println("Calibrating LDR sensor...");
  
  for (int i = 0; i < CAL_SAMPLES; i++) {
    // Use special raw reading to avoid existing calibration
    int adcValue = analogRead(LDR_PIN);
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
    if (voltage <= 0.0)
      continue;

    float resistance = FIXED_RESISTOR * (VCC / voltage - 1.0);
    float logR = log10(resistance);
    float logLux = (logR - LDR_B) / LDR_M;
    float rawLux = pow(10, logLux);

    measuredLux += rawLux;
    delay(50); // Short delay between readings
  }
  measuredLux /= CAL_SAMPLES;

  // Calculate the offset needed
  calibrationOffset = referenceValue - measuredLux;

  Serial.print("Sensor calibrated: offset = ");
  Serial.println(calibrationOffset);
  
  //---------------------------------------------------------------------
  // 2. Now calibrate the illuminance model with LED contribution
  //---------------------------------------------------------------------
  Serial.println("Calibrating illuminance model...");
  
  // Turn LED off and measure y1
  setLEDDutyCycle(0.0);
  delay(STABILIZE_TIME);
  
  // Take multiple measurements and average
  float y1 = 0.0;
  for (int i = 0; i < SAMPLES; i++) {
    y1 += readLux(); // Using calibrated readings now
    delay(STABILIZE_TIME);
  }
  y1 /= SAMPLES;
  
  // Store baseline illuminance for external light calculation
  baselineIlluminance = y1;
  
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
  for (int i = 0; i < SAMPLES; i++) {
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

//=============================================================================
// DATA STREAMING SUBSYSTEM
//=============================================================================

/**
 * Start streaming a variable to serial port
 * 
 * @param var Variable to stream (y=illuminance, u=duty, etc.)
 * @param index Node index to stream
 */
void startStream(const String &var, int index) {
  streamingEnabled = true;
  streamingVar = var;
  streamingIndex = index;
  Serial.println("ack");
}

/**
 * Stop streaming a variable
 * 
 * @param var Variable to stop streaming
 * @param index Node index
 */
void stopStream(const String &var, int index) {
  streamingEnabled = false;
  streamingVar = ""; // Clear the variable
  Serial.print("Stopped streaming ");
  Serial.print(var);
  Serial.print(" for node ");
  Serial.println(index);
}

/**
 * Process streaming in main loop
 * Sends requested variable at regular intervals
 */
void handleStreaming() {
  if (!streamingEnabled || (millis() - lastStreamTime < 500)) {
    return; // Not streaming or not time to stream yet
  }

  unsigned long currentTime = millis();
  lastStreamTime = currentTime;
  String var = streamingVar;
  int index = streamingIndex;

  if (var.equalsIgnoreCase("y")) {
    float lux = readLux();
    Serial.print("s "); // Add "s" prefix
    Serial.print(var);
    Serial.print(" ");
    Serial.print(index);
    Serial.print(" ");
    Serial.print(lux, 2);
    Serial.print(" ");
    Serial.println(currentTime); // Add timestamp
  }
  else if (var.equalsIgnoreCase("u")) {
    Serial.print("s "); // Add "s" prefix
    Serial.print(var);
    Serial.print(" ");
    Serial.print(index);
    Serial.print(" ");
    Serial.print(dutyCycle, 4);
    Serial.print(" ");
    Serial.println(currentTime); // Add timestamp
  }
  else if (var.equalsIgnoreCase("p")) {
    float power = getPowerConsumption();
    Serial.print("s "); // Add "s" prefix
    Serial.print(var);
    Serial.print(" ");
    Serial.print(index);
    Serial.print(" ");
    Serial.print(power, 2);
    Serial.print(" ");
    Serial.println(currentTime); // Add timestamp
  }
}

/**
 * Get historical data buffer as CSV string
 * 
 * @param var Variable type (y=illuminance, u=duty cycle)
 * @param index Node index
 * @return CSV string of historical values
 */
String getLastMinuteBuffer(const String &var, int index) {
  String result = "";
  int count = getLogCount();
  if (count == 0)
    return result;

  LogEntry *logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;

  // Maximum number of samples to return (to avoid overflowing serial buffer)
  const int MAX_SAMPLES = 60;
  int sampleCount = min(count, MAX_SAMPLES);

  // Calculate step to get evenly distributed samples
  int step = count > MAX_SAMPLES ? count / MAX_SAMPLES : 1;

  for (int i = 0; i < count; i += step) {
    int realIndex = (startIndex + i) % LOG_SIZE;

    if (var.equalsIgnoreCase("y")) {
      // For illuminance values
      result += String(logBuffer[realIndex].lux, 1);
    }
    else if (var.equalsIgnoreCase("u")) {
      // For duty cycle values
      result += String(logBuffer[realIndex].duty, 3);
    }

    if (i + step < count) {
      result += ",";
    }
  }

  return result;
}

//=============================================================================
// MAIN PROGRAM
//=============================================================================

/**
 * Arduino setup function
 * Initializes hardware and subsystems
 */
void setup() {
  Serial.begin(115200);

  // Debug board ID
  pico_unique_board_id_t board_id;
  pico_get_unique_board_id(&board_id);

  // Configure ADC and PWM
  analogReadResolution(12);
  analogWriteFreq(30000);
  analogWriteRange(PWM_MAX);

  // Initialize LED driver with the LED pin
  initLEDDriver(LED_PIN);

  // Run LED test to verify hardware
  testLED();

  // Initialize circular buffer storage for logging
  initStorage();

  Serial.println("Distributed Control System with CAN-BUS and Command Interface");

  // Initialize CAN communication
  initCANComm();

  // Calibrate the LDR sensor
  ledGain = calibrateSystem(1.0); // Use a lower reference value like 1.0 lux

  // Synchronize initial setpoint and reference
  setpointLux = SETPOINT_UNOCCUPIED;
  refIlluminance = setpointLux;

  // Print header for Serial Plotter (if using)
  if (DEBUG_PLOTTING) {
    Serial.println("MeasuredLux\tSetpoint");
  }
}

/**
 * Arduino main loop
 * Processes sensor readings, controls, and communication
 */
void loop() {
  // (A) Process incoming serial commands
  processSerialCommands();

  // (B) Handle any active streaming
  handleStreaming();

  // (C) Read sensor data
  float lux = readLux();

  // (D) Adapt to external light conditions
  adaptToExternalLight();

  // (E) Coordinate with neighbors for energy optimization
  if (luminaireState != STATE_OFF) {
    coordinateWithNeighbors();
  }

  // (F) Control action computation and application
  if (luminaireState == STATE_OFF) {
    // Turn off the light when in OFF state
    setLEDDutyCycle(0.0);
  }
  else if (feedbackControl) {
    // Use PID control in feedback mode
    float u = pid.compute(setpointLux, lux);
    setLEDPWMValue((int)u);
  }
  else {
    // Direct duty cycle control in manual mode
    setLEDDutyCycle(dutyCycle);
  }

  // (G) Log the current sample in the circular buffer
  logData(millis(), lux, dutyCycle);

  // (H) Process CAN messages (nonblocking)
  canCommLoop();

  // (I) Periodic CAN tasks
  unsigned long now = millis();

  // Send sensor data if periodic mode is enabled
  if (periodicCANEnabled && (now - lastCANSend >= 1000)) {
    lastCANSend = now;

    // Send illuminance reading (broadcast)
    sendSensorReading(CAN_ADDR_BROADCAST, 0, lux);

    // Send duty cycle (broadcast)
    sendSensorReading(CAN_ADDR_BROADCAST, 1, dutyCycle);

    // Send state information (broadcast)
    sendSensorReading(CAN_ADDR_BROADCAST, 2, (float)luminaireState);

    // Send external light estimate (broadcast)
    sendSensorReading(CAN_ADDR_BROADCAST, 3, getExternalIlluminance());
  }

  // Debug plotting if enabled
  if (DEBUG_MODE && DEBUG_PLOTTING) {
    Serial.print(lux, 2);
    Serial.print("\t");
    Serial.print(setpointLux, 2);
    Serial.print("\t");
    Serial.print(30.0, 2); // Upper limit
    Serial.print("\t");
    Serial.println(0.0, 2); // Lower limit
  }

  // Wait for next control cycle
  delay((int)(pid.getSamplingTime() * 1000));
}