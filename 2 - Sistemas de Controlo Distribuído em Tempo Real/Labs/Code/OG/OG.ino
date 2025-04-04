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
const float Kp = 20;                // Initial proportional gain (will be replaced by ledGain)
const float Ki = 400;               // Integral gain
const float DT = 0.01;              // Sampling period (seconds)
const float BETA = 0.8;             // Setpoint weighting factor (0.0-1.0)
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
bool canMonitorEnabled = false; // Display received messages
uint8_t nodeID = 0;             // Node ID (set during initialization)
unsigned long lastCANSend = 0;  // Last transmission timestamp

//-----------------------------------------------------------------------------
// Network Discovery and Calibration
//-----------------------------------------------------------------------------

float myFitness = 0.0;                 // This node's fitness score for coordination
float maxFitness = 0.0;                // Highest fitness score among all nodes
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
  float calibratedLux = lastFiltered + deviceConfig.calibrationOffset;
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
  deviceConfig.calibrationOffset = y1; // Calculate offset
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
void getLastMinuteBuffer(const char *var, int index, char *buffer, size_t bufferSize)
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
// WAKE-UP STATE MACHINE
//=============================================================================
/**
 * Process wake-up state machine
 * Manages the multi-phase wake-up protocol for coordinated system startup
 */
void processWakeUpStateMachine()
{
  // Get current state with thread safety
  critical_section_enter_blocking(&commStateLock);
  WakeUpState state = controlState.wakeUpState;
  // bool isMaster = controlState.isWakeUpMaster;
  unsigned long stateTime = controlState.wakeUpStateTime;
  critical_section_exit(&commStateLock);

  // If not in wake-up mode, nothing to do
  if (state == WAKEUP_IDLE || state == WAKEUP_COMPLETE)
  {
    return;
  }

  unsigned long currentTime = millis();
  unsigned long elapsedTime = currentTime - stateTime;

  // All nodes process the same state machine but make decisions differently
  switch (state)
  {
  case WAKEUP_RESET:
    // Any node can propose moving to ACK wait phase after timeout
    if (elapsedTime > 1000)
    {
      // Nodes decide to send messages based on fitness-weighted probability
      float transitionProbability = 0.2 + (myFitness / maxFitness) * 0.6;

      // Randomize slightly so not all nodes act simultaneously
      if (random(100) / 100.0f < transitionProbability)
      {
        critical_section_enter_blocking(&commStateLock);
        controlState.wakeUpState = WAKEUP_ACK_WAIT;
        controlState.wakeUpStateTime = currentTime;
        critical_section_exit(&commStateLock);

        // Only the fittest nodes announce the transition
        if (transitionProbability > 0.5)
        {
          Serial.println("Wake-up: Moving to acknowledgment phase...");
          sendControlCommand(CAN_ADDR_BROADCAST, CAN_CTRL_WAKEUP_ACK, 0.0f);
        }
      }
    }
    break;

  case WAKEUP_ACK_WAIT:
    // After waiting for ACKs, move to calibration
    if (elapsedTime > 3000)
    {
      // Probabilistic decision based on fitness
      float transitionProbability = 0.2 + (myFitness / maxFitness) * 0.6;

      // Add a staggered delay based on node ID to prevent simultaneous calibration
      int myCalibrationDelay = deviceConfig.nodeId * 500; // 500ms delay per node ID

      if (elapsedTime > (3000 + myCalibrationDelay) &&
          random(100) / 100.0f < transitionProbability)
      {

        critical_section_enter_blocking(&commStateLock);
        controlState.wakeUpState = WAKEUP_CALIBRATE;
        controlState.wakeUpStateTime = currentTime;
        critical_section_exit(&commStateLock);

        // Only some nodes announce the transition
        if (transitionProbability > 0.5)
        {
          Serial.println("Wake-up: Starting calibration phase...");
          sendControlCommand(CAN_ADDR_BROADCAST, CAN_CTRL_WAKEUP_CALIBRATE, 0.0f);
        }
      }
    }
    break;

  case WAKEUP_CALIBRATE:
    // After calibration completes, transition to normal operation
    if (elapsedTime > 15000)
    {
      // Weighted probability based on fitness
      float transitionProbability = 0.2 + (myFitness / maxFitness) * 0.6;

      if (random(100) / 100.0f < transitionProbability)
      {
        critical_section_enter_blocking(&commStateLock);
        controlState.wakeUpState = WAKEUP_TRANSITION;
        controlState.wakeUpStateTime = currentTime;
        critical_section_exit(&commStateLock);

        // Only some nodes broadcast completion
        if (transitionProbability > 0.5)
        {
          Serial.println("Wake-up: Calibration complete, transitioning to normal operation...");
          sendControlCommand(CAN_ADDR_BROADCAST, CAN_CTRL_WAKEUP_COMPLETE, 0.0f);
        }
      }
    }
    break;

  case WAKEUP_TRANSITION:
    // All nodes can independently complete the wake-up process
    if (elapsedTime > 2000)
    {
      critical_section_enter_blocking(&commStateLock);
      controlState.wakeUpState = WAKEUP_COMPLETE;
      controlState.wakeUpStateTime = currentTime;

      // All nodes enter the same state
      controlState.luminaireState = STATE_UNOCCUPIED;
      controlState.setpointLux = SETPOINT_UNOCCUPIED;
      controlState.feedbackControl = true;
      critical_section_exit(&commStateLock);

      // Reset PID controller for clean start
      pid.reset();

      Serial.println("Wake-up: Complete - System now in normal operation");
    }
    break;

  default:
    break;
  }

  // Universal timeout safety - all nodes can recover independently
  if (elapsedTime > 30000)
  { // 30 seconds timeout for any state
    critical_section_enter_blocking(&commStateLock);
    controlState.wakeUpState = WAKEUP_IDLE;
    critical_section_exit(&commStateLock);

    Serial.println("Wake-up: Timeout, wake-up process aborted");
  }
}

/**
 * Determine if this node should coordinate based on capabilities and consensus
 * Uses a distributed election algorithm rather than just node ID
 *
 * @return true if this node should be coordinator

bool shouldBeCoordinator()
{
  uint8_t myNodeId;
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  critical_section_exit(&commStateLock);

  // Count active nodes (including ourselves)
  int activeNodes = 1; // Start with 1 for ourselves
  float totalFitness = calculateNodeFitness(myNodeId);
  myFitness = totalFitness; // Store in global for other functions
  maxFitness = myFitness;   // Initialize max to our fitness
  uint8_t fittestNode = myNodeId;

  // Calculate total system fitness and find the fittest node
  for (int i = 0; i < MAX_NEIGHBORS; i++)
  {
    if (neighbors[i].isActive)
    {
      activeNodes++;

      // Calculate this node's fitness
      float nodeFitness = calculateNodeFitness(neighbors[i].nodeId);
      totalFitness += nodeFitness;

      // Track the fittest node (or lowest ID in case of tie)
      if (nodeFitness > maxFitness ||
          (nodeFitness == maxFitness && neighbors[i].nodeId < fittestNode))
      {
        maxFitness = nodeFitness;
        fittestNode = neighbors[i].nodeId;
      }
    }
  }

  // Weighted probability approach - nodes with higher fitness have
  // higher probability of becoming coordinator
  float coordinationThreshold = myFitness / totalFitness;

  // Use our node ID and uptime to generate a stable-ish random number
  // This ensures consistent coordinator selection within a short time period
  uint32_t seed = myNodeId * 1000 + (uint32_t)(getElapsedTime() / 10.0);
  srand(seed);
  float randomValue = (float)rand() / RAND_MAX;

  // Log decision factors if debugging
  if (canMonitorEnabled)
  {
    Serial.print("Coordination fitness: ");
    Serial.print(myFitness);
    Serial.print("/");
    Serial.print(totalFitness);
    Serial.print(" (");
    Serial.print(coordinationThreshold * 100);
    Serial.print("%), random: ");
    Serial.print(randomValue);
    Serial.print(", fittest node: ");
    Serial.println(fittestNode);
  }

  // If we're clearly the fittest node, we should coordinate
  if (fittestNode == myNodeId)
  {
    return true;
  }

  // Otherwise use probabilistic approach
  return randomValue < coordinationThreshold;
}*/

/**
 * Calculate node fitness for coordinator role based on capabilities
 * Higher value means better suited to be coordinator
 *
 * @param nodeId Node to evaluate
 * @return Fitness score

float calculateNodeFitness(uint8_t nodeId)
{
  float fitness = 1.0; // Base fitness

  // Check if this is our own node
  critical_section_enter_blocking(&commStateLock);
  bool isOurNode = (nodeId == deviceConfig.nodeId);
  critical_section_exit(&commStateLock);

  // 1. Uptime contribution - stable nodes make better coordinators
  // Max contribution: 1.0 (after running for 1+ hours)
  float uptime = 0.0;
  if (isOurNode)
  {
    uptime = min(getElapsedTime() / 3600.0f, 1.0f);
  }
  else
  {
    // Estimate neighbor uptime based on how long we've known them
    for (int i = 0; i < MAX_NEIGHBORS; i++)
    {
      if (neighbors[i].isActive && neighbors[i].nodeId == nodeId)
      {
        uptime = min(float(millis() - neighbors[i].firstSeen) / 3600000.0f, 1.0f);
        break;
      }
    }
  }
  fitness += uptime;

  // 2. Network centrality - prefer nodes that can see more of the network
  float centrality = 0.0;
  if (isOurNode)
  {
    // Count our neighbors as a fraction of MAX_NEIGHBORS
    int neighborCount = 0;
    for (int i = 0; i < MAX_NEIGHBORS; i++)
    {
      if (neighbors[i].isActive)
      {
        neighborCount++;
      }
    }
    centrality = float(neighborCount) / MAX_NEIGHBORS;
  }
  else
  {
    // For other nodes, we estimate based on observations
    centrality = 0.5; // Default assumption
  }
  fitness += centrality;

  return fitness;
}*/

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
// Network Discovery Process:
// 1. Nodes send heartbeats every 5 seconds
// 2. Receiving a heartbeat from a new node updates the neighbor list
// 3. Nodes timeout after 20 seconds without heartbeats
// 4. Network is considered "ready" after 30 seconds of stability
// 5. Any node joining or leaving resets the stability timer
void updateNeighborInfo(uint8_t nodeId, uint8_t sensorType, float value)
{
  // Find node in list or empty slot
  int nodeSlot = -1;
  int emptySlot = -1;

  for (int i = 0; i < MAX_NEIGHBORS; i++)
  {
    if (neighbors[i].isActive && neighbors[i].nodeId == nodeId)
    {
      nodeSlot = i;
      break;
    }
    else if (!neighbors[i].isActive && emptySlot < 0)
    {
      emptySlot = i;
    }
  }

  // Add new neighbor if not found and empty slot available
  if (nodeSlot < 0 && emptySlot >= 0)
  {
    neighbors[emptySlot].nodeId = nodeId;
    neighbors[emptySlot].isActive = true;
    neighbors[emptySlot].lastUpdate = millis();
    neighbors[emptySlot].firstSeen = millis(); // Record when first discovered
    neighbors[emptySlot].lastLux = 0.0;
    neighbors[emptySlot].lastDuty = 0.0;
    neighbors[emptySlot].state = STATE_OFF;
    nodeSlot = emptySlot;

    // Mark network change to reset stability timer
    critical_section_enter_blocking(&commStateLock);
    controlState.networkReady = false;
    controlState.lastNetworkChange = millis();
    critical_section_exit(&commStateLock);

    Serial.print("New neighbor detected: Node ID ");
    Serial.println(nodeId);
  }

  // Update node information
  if (nodeSlot >= 0)
  {
    neighbors[nodeSlot].lastUpdate = millis();

    if (sensorType == 0)
    {
      neighbors[nodeSlot].lastLux = value;
    }
    else if (sensorType == 1)
    {
      neighbors[nodeSlot].lastDuty = value;
    }
    else if (sensorType == 2)
    {
      neighbors[nodeSlot].state = (LuminaireState)((int)value);
    }
  }
}

void checkNeighborTimeouts()
{
  unsigned long currentTime = millis();
  const unsigned long NEIGHBOR_TIMEOUT = 20000; // 20 seconds timeout
  bool networkChanged = false;

  for (int i = 0; i < MAX_NEIGHBORS; i++)
  {
    if (neighbors[i].isActive)
    {
      // Mark inactive if no heartbeat for more than 20 seconds
      if (currentTime - neighbors[i].lastUpdate > NEIGHBOR_TIMEOUT)
      {
        Serial.print("Node ");
        Serial.print(neighbors[i].nodeId);
        Serial.println(" timed out (no heartbeat received)");

        neighbors[i].isActive = false;
        networkChanged = true;
      }
    }
  }

  // Reset network stability timer if any node was removed
  if (networkChanged)
  {
    critical_section_enter_blocking(&commStateLock);
    controlState.networkReady = false;
    controlState.lastNetworkChange = currentTime;
    critical_section_exit(&commStateLock);
  }
}

void checkNetworkStability()
{
  const unsigned long STABILITY_TIMEOUT = 30000; // 30 seconds with no changes
  unsigned long currentTime = millis();

  critical_section_enter_blocking(&commStateLock);
  bool isReady = controlState.networkReady;
  unsigned long lastChange = controlState.lastNetworkChange;
  critical_section_exit(&commStateLock);

  // If not ready yet, check if we've been stable for long enough
  if (!isReady && (currentTime - lastChange > STABILITY_TIMEOUT))
  {
    critical_section_enter_blocking(&commStateLock);
    controlState.networkReady = true;
    critical_section_exit(&commStateLock);

    Serial.println("Network discovery complete - system ready");

    // Log all active neighbors
    int activeCount = 0;
    Serial.println("Active nodes:");
    for (int i = 0; i < MAX_NEIGHBORS; i++)
    {
      if (neighbors[i].isActive)
      {
        activeCount++;
        Serial.print("  Node ");
        Serial.println(neighbors[i].nodeId);
      }
    }
    Serial.print("Total active nodes (including self): ");
    Serial.println(activeCount + 1);
  }
}

/**
 * Calculate light contribution from neighboring luminaires
 *
 * @return Estimated illuminance contribution from neighbors (lux)

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
}*/

/**
 * Coordinate illuminance with neighbors to optimize energy usage
 * Adjusts PID target to account for light from neighboring nodes

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
 */
//=============================================================================
// MAIN PROGRAM
//=============================================================================

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
  Serial.print(", Kp=");
  Serial.print(config.pidKp);
  Serial.print(", Ki=");
  Serial.print(config.pidKi);
  Serial.print(", Beta=");
  Serial.print(config.pidBeta);
  Serial.print(", LED Gain=");
  Serial.print(config.ledGain);
  Serial.print(", Calibration Offset=");
  Serial.println(config.calibrationOffset);
  Serial.println();
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
  for (int i = 0; i < 8; i++)
  {
    if (board_id.id[i] < 16)
      Serial.print("0");
    Serial.print(board_id.id[i], HEX);
    if (i < 7)
      Serial.print(":");
  }
  Serial.print(" -> NodeID: ");
  Serial.println(nodeId);

  // Save existing calibration values
  float existingGain = deviceConfig.ledGain;
  float existingOffset = deviceConfig.calibrationOffset;

  // Start with default configuration
  DeviceConfig config;

  // IMPORTANT: Always use the unique node ID
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
  analogWriteRange(PWM_MAX);

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
  controlState.wakeUpState = WAKEUP_IDLE;
  // controlState.isWakeUpMaster = false;
  controlState.wakeUpStateTime = 0;

  commState.streamingEnabled = false;

  critical_section_enter_blocking(&commStateLock);
  controlState.networkReady = false;
  controlState.lastNetworkChange = millis();
  critical_section_exit(&commStateLock);

  // Calibrate the system
  deviceConfig.ledGain = calibrateSystem(1.0);
  Serial.println();

  // Configure device based on unique ID
  configureDeviceFromID();

  initCANComm();

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

  // Process wake-up state machine
  processWakeUpStateMachine();

  // Send periodic heartbeat (every 5 seconds)
  static unsigned long lastHeartbeatTime = 0;

  if (millis() - lastHeartbeatTime > 5000)
  {
    sendHeartbeat();
    lastHeartbeatTime = millis();
  }

  static unsigned long lastNeighborCheckTime = 0;

  // Check neighbor timeouts and network stability every 2 seconds
  if (millis() - lastNeighborCheckTime > 2000)
  {
    checkNeighborTimeouts();
    checkNetworkStability();
    lastNeighborCheckTime = millis();
  }

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

// Core 0: CAN Communication
void core0_main()
{
  CoreMessage msg;

  while (true)
  {
    // Process CAN messages
    canCommLoop();

    // Check for messages from Core 1
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
        // Handle state update as needed
        break;
      }
    }

    // Brief delay to prevent core hogging
    sleep_ms(1);
  }
}