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
const int NUM_SAMPLES = 10;            // Number of samples for averaging
const float OUTLIER_THRESHOLD = 2.0;   // Standard deviations for outlier rejection
const float ALPHA = 0.3;               // EMA filter coefficient (0-1)

// --- PID Controller Parameters ---
const float KP = 28.0;   // Proportional gain
const float KI = 230.0;  // Integral gain
const float KD = 0.0;    // Derivative gain
const float N = 10.0;    // Filter coefficient
const float DT = 0.01;   // Sampling time (seconds)

// --- Power Consumption Model ---
const float MAX_POWER_WATTS = 1.0;    // Maximum power at 100% duty

// ===================== GLOBAL VARIABLES =====================

// --- Control System State ---
float setpointLux = 10.0;        // Desired lux (setpoint)
float dutyCycle = 0.0;           // Current duty cycle [0..1]
float refIlluminance = 15.0;     // Reference illuminance
bool occupancy = false;          // Occupancy flag
bool antiWindup = false;         // Anti-windup flag for PID controller
bool feedbackControl = true;     // Enable/disable feedback control

// --- Sensor Calibration State ---
float calibrationOffset = 0.0;   // Calibration offset
float lastFilteredLux = -1.0;    // Last filtered value for EMA

// --- Communication & Timing ---
unsigned long lastTransmit = 0;  // Last CAN transmission time

// --- Streaming Variables ---
bool streamingEnabled = false;   // Flag for streaming mode
String streamingVar = "";        // Variable being streamed
int streamingIndex = 0;          // Index for the streaming command
unsigned long lastStreamTime = 0; // Time of last stream update

// --- CAN Communication ---
bool periodicCANEnabled = false;    // Enable periodic message sending
bool canMonitorEnabled = false;     // Display received messages
uint8_t nodeID = 0;                 // Will be assigned in initCANComm()
unsigned long lastHeartbeat = 0;    // Time of last heartbeat
unsigned long heartbeatInterval = 5000; // Send heartbeat every 5 seconds

// --- Debugging ---
bool DEBUG_MODE = false;      // Master debug switch
bool DEBUG_LED = false;       // LED driver debug messages
bool DEBUG_SENSOR = false;    // Sensor readings debug
bool DEBUG_PID = false;       // PID control debug
bool DEBUG_PLOTTING = false;  // Serial plotter output

// --- Controller Object ---
PIDController pid(KP, KI, KD, N, DT);

// ===================== SENSOR FUNCTIONS =====================

/**
 * Advanced LUX measurement with multi-stage filtering:
 * 1. Multiple samples to reduce noise
 * 2. Statistical outlier rejection
 * 3. EMA filtering for temporal smoothing
 * 4. Calibration offset application
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
  
  if (count == 0) return 0.0; // No valid readings
  
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
  if (calibratedLux < 0.0) calibratedLux = 0.0;
  
  return calibratedLux;
}

/**
 * Calibrates the LUX sensor using a known reference value
 * This should be called when a trustworthy external light meter is available
 */
void calibrateLuxSensor(float knownLux) {
  float measuredLux = 0.0;
  const int CAL_SAMPLES = 20;
  
  for (int i = 0; i < CAL_SAMPLES; i++) {
    // Use a special raw reading to avoid existing calibration
    int adcValue = analogRead(LDR_PIN);
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
    if (voltage <= 0.0) continue;
    
    float resistance = FIXED_RESISTOR * (VCC / voltage - 1.0);
    float logR = log10(resistance);
    float logLux = (logR - LDR_B) / LDR_M;
    float rawLux = pow(10, logLux);
    
    measuredLux += rawLux;
    delay(50);
  }
  measuredLux /= CAL_SAMPLES;
  
  // Calculate the offset needed
  calibrationOffset = knownLux - measuredLux;
  
  Serial.print("Sensor calibrated: offset = ");
  Serial.println(calibrationOffset);
}

// Return voltage at LDR pin
float getVoltageAtLDR() {
  int adcValue = analogRead(LDR_PIN);
  return (adcValue / MY_ADC_RESOLUTION) * VCC;
}

// ===================== UTILITY FUNCTIONS =====================
// --- Test LED functionality ---
// Add this code to your main.ino temporarily for testing
void testLED() {
  Serial.println("Testing LED...");
  
  // Test full brightness
  Serial.println("Full brightness");
  setLEDDutyCycle(1.0);
  delay(1000);
  
  // Test half brightness
  Serial.println("Half brightness");
  setLEDDutyCycle(0.5);
  delay(1000);
  
  // Test off
  Serial.println("Off");
  setLEDDutyCycle(0.0);
  delay(1000);
  
  Serial.println("LED test complete");
}

// --- Return external illuminance (from environment) ---
float getExternalIlluminance() {
  float measuredLux = readLux();
  float ledContribution = dutyCycle * 30.0; // Estimate: max LED contribution is 30 lux
  
  float externalLux = max(0.0f, measuredLux - ledContribution);
  return externalLux;
}

// --- Return power consumption estimate ---
float getPowerConsumption() {
  return dutyCycle * MAX_POWER_WATTS;
}

// --- Return elapsed time since boot in seconds ---
unsigned long getElapsedTime() {
  return millis() / 1000;
}

// ===================== STREAMING FUNCTIONS =====================

// --- Start streaming a variable ---
void startStream(const String &var, int index) {
  streamingEnabled = true;
  streamingVar = var;
  streamingIndex = index;
  Serial.println("ack");
}

// --- Stop streaming ---
void stopStream(const String &var, int index) {
  streamingEnabled = false;
}

// --- Handle streaming in the main loop ---
void handleStreaming() {
  if (!streamingEnabled || (millis() - lastStreamTime < 500)) {
    return; // Not streaming or not time to stream yet
  }
  
  lastStreamTime = millis();
  String var = streamingVar;
  int index = streamingIndex;
  
  if (var.equalsIgnoreCase("y")) {
    float lux = readLux();
    Serial.print("y ");
    Serial.print(index);
    Serial.print(" ");
    Serial.println(lux, 2);
  } else if (var.equalsIgnoreCase("u")) {
    Serial.print("u ");
    Serial.print(index);
    Serial.print(" ");
    Serial.println(dutyCycle, 4);
  } else if (var.equalsIgnoreCase("p")) {
    float power = getPowerConsumption();
    Serial.print("p ");
    Serial.print(index);
    Serial.print(" ");
    Serial.println(power, 2);
  }
}

// Return last minute buffer as a comma-separated string
String getLastMinuteBuffer(const String &var, int index) {
  String result = "";
  int count = getLogCount();
  if (count == 0) return result;
  
  LogEntry* logBuffer = getLogBuffer();
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
    } else if (var.equalsIgnoreCase("u")) {
      // For duty cycle values
      result += String(logBuffer[realIndex].duty, 3);
    }
    
    if (i + step < count) {
      result += ",";
    }
  }
  
  return result;
}

// ===================== MAIN PROGRAM =====================

// In your loop function, add a periodic CAN message:
unsigned long lastCANSend = 0;
uint8_t messageCounter = 0;


void setup() {
  Serial.begin(115200);

  // Configure ADC and PWM
  analogReadResolution(12);
  analogWriteFreq(30000);
  analogWriteRange(PWM_MAX);

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

void loop() {
  // (A) Process incoming serial commands
  processSerialCommands();

  // (B) Handle any active streaming
  handleStreaming();
  
  // (C) Read sensor data
  float lux = readLux();
  
  // (D) Control action computation and application
  if (feedbackControl) {
    // Compute PID control with feedback
    float u = pid.compute(setpointLux, lux);
    
    // Use LED driver - this will update both the LED and dutyCycle
    setLEDPWMValue((int)u);
  } else {
    // Direct duty cycle control (open loop)
    // dutyCycle is already updated by the command interface
    setLEDDutyCycle(dutyCycle);
  }
  
  // (E) Log the current sample in the circular buffer
  logData(millis(), lux, dutyCycle);
  
  // (F) Process CAN messages (nonblocking)
  canCommLoop();
  
  // (G) Periodic CAN tasks
  unsigned long now = millis();
  
  // Send heartbeat periodically
  if (now - lastHeartbeat >= heartbeatInterval) {
    lastHeartbeat = now;
    sendHeartbeat();
  }
  
  // Send sensor data if periodic mode is enabled
  if (periodicCANEnabled && (now - lastCANSend >= 1000)) {
    lastCANSend = now;
    
    // Send illuminance reading (broadcast)
    float lux = readLux();
    sendSensorReading(CAN_ADDR_BROADCAST, 0, lux);
    
    // Send duty cycle (broadcast)
    sendSensorReading(CAN_ADDR_BROADCAST, 1, dutyCycle);
  }
  

  // (H) Optional: Serial plotting data
  if (DEBUG_MODE && DEBUG_PLOTTING) {
    // Serial plotter output
    Serial.print(lux, 2);
    Serial.print("\t");
    Serial.print(setpointLux, 2);
    Serial.print("\t");
    Serial.print(30.0, 2);  // Upper limit
    Serial.print("\t");
    Serial.println(0.0, 2); // Lower limit
  }
  
  // Optional: Print debug info about sensor readings
  if (DEBUG_MODE && DEBUG_SENSOR && (millis() % 1000 < 10)) {
    Serial.print("LUX Reading: ");
    Serial.println(lux);
  }
  
  
  // Wait for next control cycle
  delay((int)(pid.getSamplingTime() * 1000));
}