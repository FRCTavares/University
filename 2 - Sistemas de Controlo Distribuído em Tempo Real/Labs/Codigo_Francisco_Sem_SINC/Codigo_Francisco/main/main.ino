#include <Arduino.h>
#include <math.h>
#include "PIDController.h"
#include "Storage.h"
#include "Metrics.h"
#include "CommandInterface.h"
#include "CANComm.h"
#include "pico/multicore.h"
#include "LEDDriver.h"
#include "Globals.h"

// ===================== CONFIGURATION =====================

// --- Sensor Calibration ---
#define VCC 3.3
#define MY_ADC_RESOLUTION 4095.0
#define FIXED_RESISTOR 10000.0

// Example LDR reference at ~10 lux
const float R10 = 225000.0;
const float LDR_M = -1.0;
float LDR_B = log10(R10) - LDR_M;

// --- PIN Assignments ---
const int LED_PIN = 15;
const int LDR_PIN = A0;

// --- PWM Configuration ---
const int PWM_MAX = 4095;
const int PWM_MIN = 0;

// --- Measurement Filter Configuration ---
const int NUM_SAMPLES = 10;          // Number of samples for averaging
const float OUTLIER_THRESHOLD = 2.0; // Standard deviations for outlier rejection
const float ALPHA = 0.3;             // EMA filter coefficient (0-1)

// --- PID Controller Parameters ---
const float KP = 28.0;  // Proportional gain
const float KI = 230.0; // Integral gain
const float KD = 0.0;   // Derivative gain
const float N = 10.0;   // Filter coefficient
const float DT = 0.01;  // Sampling time (seconds)

// --- Power Consumption Model ---
const float MAX_POWER_WATTS = 1.0; // Maximum power at 100% duty

// ===================== GLOBAL VARIABLES =====================

// --- Control System State ---
float setpointLux = 10.0;    // Desired lux (setpoint)
float dutyCycle = 0.0;       // Current duty cycle [0..1]
float refIlluminance = 15.0; // Reference illuminance
bool occupancy = false;      // Occupancy flag
bool antiWindup = false;     // Anti-windup flag for PID controller
bool feedbackControl = true; // Enable/disable feedback control

// --- Sensor Calibration State ---
float calibrationOffset = 0.0; // Calibration offset
float lastFilteredLux = -1.0;  // Last filtered value for EMA

// --- Communication & Timing ---
unsigned long lastTransmit = 0; // Last CAN transmission time

// --- CAN Communication ---
bool periodicCANEnabled = false;        // Enable periodic message sending
bool canMonitorEnabled = false;         // Display received messages
uint8_t nodeID = 0;                     // Will be assigned in initCANComm()
unsigned long lastHeartbeat = 0;        // Time of last heartbeat
unsigned long heartbeatInterval = 5000; // Send heartbeat every 5 seconds

// --- Controller Object ---
PIDController pid(KP, KI, KD, N, DT);

// --- Luminaire State Management ---
LuminaireState luminaireState = STATE_UNOCCUPIED;

// --- Streaming Variables ---
bool streamingEnabled = false;    // Flag to enable streaming
String streamingVar = "";         // Variable to stream
int streamingIndex = 0;           // Index for multi-node systems
unsigned long lastStreamTime = 0; // Timestamp of last stream update

// --- Debug Flags ---
bool DEBUG_MODE = false;     // Master debug switch
bool DEBUG_LED = false;      // LED driver debug messages
bool DEBUG_SENSOR = false;   // Sensor readings debug
bool DEBUG_PID = false;      // PID control debug
bool DEBUG_PLOTTING = false; // Serial plotter output

// Illuminance setpoints for different states
const float SETPOINT_OFF = 0.0;        // 0 lux when off
const float SETPOINT_UNOCCUPIED = 5.0; // 5 lux when unoccupied
const float SETPOINT_OCCUPIED = 15.0;  // 15 lux when occupied

// Function to change state and update setpoints accordingly
void changeState(LuminaireState newState)
{
  // Don't do anything if state is unchanged
  if (newState == luminaireState)
  {
    return;
  }

  luminaireState = newState;

  // Update setpoint based on new state
  switch (luminaireState)
  {
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

// --- Neighbor Management ---
struct NeighborInfo
{
  uint8_t nodeId;
  float lastLux;
  float lastDuty;
  LuminaireState state;
  unsigned long lastUpdate;
  bool isActive;
};

const int MAX_NEIGHBORS = 5;
NeighborInfo neighbors[MAX_NEIGHBORS];

// Update neighbor information when receiving CAN messages
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

// Calculate light contribution from neighbors
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

// Coordination algorithm to optimize energy usage
void coordinateWithNeighbors()
{
  // Calculate total neighbor light contribution
  float neighborContribution = getNeighborsContribution();

  if (neighborContribution > 0.5)
  { // Only adjust if contribution is significant
    // Adjust our target to account for light from neighbors
    float adjustedTarget = max(0.0f, setpointLux - neighborContribution * 0.8);

    // Dynamic PID adjustment based on cooperation
    pid.setTarget(adjustedTarget);
  }
}

// ===================== SENSOR FUNCTIONS =====================

/**
 * Advanced LUX measurement with multi-stage filtering:
 * 1. Multiple samples to reduce noise
 * 2. Statistical outlier rejection
 * 3. EMA filtering for temporal smoothing
 * 4. Calibration offset application
 */
float readLux()
{
  float samples[NUM_SAMPLES];
  float sum = 0.0;
  float count = 0.0;

  // 1. Take multiple samples to reduce noise
  for (int i = 0; i < NUM_SAMPLES; i++)
  {
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
  variance /= count;
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
  if (lastFilteredLux < 0)
  {
    lastFilteredLux = filteredMean; // First valid reading
  }
  else
  {
    lastFilteredLux = ALPHA * filteredMean + (1.0 - ALPHA) * lastFilteredLux;
  }

  // 5. Apply calibration offset and safety bounds check
  float calibratedLux = lastFilteredLux + calibrationOffset;
  if (calibratedLux < 0.0)
    calibratedLux = 0.0;

  return calibratedLux;
}

/**
 * Calibrates the LUX sensor using a known reference value
 * This should be called when a trustworthy external light meter is available
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
    float rawLux = pow(10, logLux);

    measuredLux += rawLux;
    delay(5000);
  }
  measuredLux /= CAL_SAMPLES;

  // Calculate the offset needed
  calibrationOffset = knownLux - measuredLux;

  Serial.print("Sensor calibrated: offset = ");
  Serial.println(calibrationOffset);
}

// Return voltage at LDR pin
float getVoltageAtLDR()
{
  int adcValue = analogRead(LDR_PIN);
  return (adcValue / MY_ADC_RESOLUTION) * VCC;
}

// ===================== UTILITY FUNCTIONS =====================
// --- Test LED functionality ---
// Add this code to your main.ino temporarily for testing

// --- External Light Adaptation ---
float lastExternalLux = 0.0;
float externalLuxAverage = 0.0;
const float EXT_LUX_ALPHA = 0.05; // Slow-moving average for stability

// Enhanced external illuminance calculation
float getExternalIlluminance()
{
  float measuredLux = readLux();

  // More precise non-linear model of LED contribution
  float ledContribution;
  if (dutyCycle < 0.1)
  {
    ledContribution = dutyCycle * 15.0; // Linear at low duty cycles
  }
  else
  {
    ledContribution = dutyCycle * dutyCycle * 35.0; // Non-linear at higher duty cycles
  }

  // Calculate current external lux estimate
  float currentExternalLux = max(0.0f, measuredLux - ledContribution);

  // Apply slow-moving average to external illuminance
  if (lastExternalLux == 0.0)
  {
    externalLuxAverage = currentExternalLux;
  }
  else
  {
    externalLuxAverage = EXT_LUX_ALPHA * currentExternalLux +
                         (1.0 - EXT_LUX_ALPHA) * externalLuxAverage;
  }

  lastExternalLux = currentExternalLux;
  return externalLuxAverage;
}

// Adapt control to external light changes
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

  // Skip first run or when in manual mode
  if (previousExternal < 0 || !feedbackControl)
  {
    previousExternal = externalLux;
    return;
  }

  // If external light has changed significantly (>1 lux)
  if (abs(externalLux - previousExternal) > 1.0)
  {
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

    if (DEBUG_MODE && DEBUG_SENSOR)
    {
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

// --- Return power consumption estimate ---
float getPowerConsumption()
{
  return dutyCycle * MAX_POWER_WATTS;
}

// --- Return elapsed time since boot in seconds ---
unsigned long getElapsedTime()
{
  return millis() / 1000;
}

// --- Test LED functionality ---
void testLED()
{
  // Quick test of LED by ramping up and down
  Serial.println("Testing LED...");

  for (int i = 0; i <= 100; i += 10)
  {
    setLEDPercentage(i);
    delay(50);
  }

  for (int i = 100; i >= 0; i -= 10)
  {
    setLEDPercentage(i);
    delay(50);
  }

  // Set LED to off after test
  setLEDDutyCycle(0.0);
  Serial.println("LED test complete.");
}

// ===================== STREAMING FUNCTIONS =====================

// --- Start streaming a variable ---
void startStream(const String &var, int index)
{
  streamingEnabled = true;
  streamingVar = var;
  streamingIndex = index;
  Serial.println("ack");
}

// --- Stop streaming ---
void stopStream(const String &var, int index)
{
  streamingEnabled = false;
  streamingVar = ""; // Clear the variable
  Serial.print("Stopped streaming ");
  Serial.print(var);
  Serial.print(" for node ");
  Serial.println(index);
}

// --- Handle streaming in the main loop ---
void handleStreaming()
{
  if (!streamingEnabled || (millis() - lastStreamTime < 500))
  {
    return; // Not streaming or not time to stream yet
  }

  unsigned long currentTime = millis();
  lastStreamTime = currentTime;
  String var = streamingVar;
  int index = streamingIndex;

  if (var.equalsIgnoreCase("y"))
  {
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
  else if (var.equalsIgnoreCase("u"))
  {
    Serial.print("s "); // Add "s" prefix
    Serial.print(var);
    Serial.print(" ");
    Serial.print(index);
    Serial.print(" ");
    Serial.print(dutyCycle, 4);
    Serial.print(" ");
    Serial.println(currentTime); // Add timestamp
  }
  else if (var.equalsIgnoreCase("p"))
  {
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

// Return last minute buffer as a comma-separated string
String getLastMinuteBuffer(const String &var, int index)
{
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

  for (int i = 0; i < count; i += step)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;

    if (var.equalsIgnoreCase("y"))
    {
      // For illuminance values
      result += String(logBuffer[realIndex].lux, 1);
    }
    else if (var.equalsIgnoreCase("u"))
    {
      // For duty cycle values
      result += String(logBuffer[realIndex].duty, 3);
    }

    if (i + step < count)
    {
      result += ",";
    }
  }

  return result;
}

// ===================== MAIN PROGRAM =====================

// In your loop function, add a periodic CAN message:
unsigned long lastCANSend = 0;

void setup()
{
  Serial.begin(115200);

  // Debug board ID
  pico_unique_board_id_t board_id;
  pico_get_unique_board_id(&board_id);

  // Configure ADC and PWM
  analogReadResolution(12);
  analogWriteFreq(30000);
  analogWriteRange(PWM_MAX);

  // Calibrate the LDR sensor
  calibrateLuxSensor(10.0);

  // Initialize LED driver with the LED pin
  initLEDDriver(LED_PIN);

  // Run LED test
  testLED();

  // Initialize circular buffer storage for logging.
  initStorage();

  Serial.println("Distributed Control System with CAN-BUS and Command Interface");

  // Initialize CAN communication.
  initCANComm();

  // Synchronize initial setpoint and reference.
  setpointLux = 15.0;
  refIlluminance = setpointLux;

  // Print header for Serial Plotter (two numeric columns: Measured Lux and Setpoint)
  Serial.println("MeasuredLux\tSetpoint");
}

void loop()
{
  // (A) Process incoming serial commands
  processSerialCommands();

  // (B) Handle any active streaming
  handleStreaming();

  // (C) Read sensor data
  float lux = readLux();

  // (D) Adapt to external light conditions
  adaptToExternalLight();

  // (E) Coordinate with neighbors for energy optimization
  if (luminaireState != STATE_OFF)
  {
    coordinateWithNeighbors();
  }

  // (F) Control action computation and application
  if (luminaireState == STATE_OFF)
  {
    // Turn off the light when in OFF state
    setLEDDutyCycle(0.0);
  }
  else if (feedbackControl)
  {
    // Use PID control in feedback mode
    float u = pid.compute(setpointLux, lux);
    setLEDPWMValue((int)u);
  }
  else
  {
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
  if (periodicCANEnabled && (now - lastCANSend >= 1000))
  {
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
  if (DEBUG_MODE && DEBUG_PLOTTING)
  {
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