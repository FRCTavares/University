#include <Arduino.h>
#include <math.h>
#include "Controller.h"
#include "CANComm.h"
#include "Header.h"

// Global variables
float setpointLux = 0.0;
float ledGain = 0.0;
float baselineIlluminance = 0.0;
float externalLuxAverage = 0.0;
float lastExternalLux = 0.0;
float calibrationOffset = 0.0;
bool feedbackControl = true;
bool occupancy = false;
float prevDutyWithoutFilter1 = 0.0;
float prevDutyWithoutFilter2 = 0.0;
float refIlluminance = 0.0;

PIController pid;

// External Function Prototypes TO BE CHANGED
void processSerialCommands();
void handleStreaming();
void handleRemoteStreamRequests();
extern void setLEDDutyCycle(float duty);
void setLEDPWMValue(int value);
void initCANComm();
void initStorage();
void logData(unsigned long time, float lux, float duty);
void canCommLoop();

// Luminaire/Node Structure
struct node // I'd recommend capitalizing struct names
{
  float id;
  float state; //OFF: 0, UNOCCUPIED: 1, OCCUPIED: 2
  float Kp;
  float Ki;
  float DT;
  float BETA;
  bool antiWindup;
  bool feedbackControl;
  bool feedforwardControl;
  float LuxOffset; 
  LogEntry logBuffer[LOG_SIZE];
  float dutyCycle;
} node;

struct LogEntry {
  unsigned long timestamp;
  float lux;
  float duty;
  float setpoint;
  float flicker;
  float jitter;
  float extLux;
} LogEntry;


struct calibration
{
  float gain;
  float offset;
};


/**
 * Change luminaire operating state and update settings accordingly
 *
 * @param newState The target state (OFF, UNOCCUPIED, OCCUPIED)
 */
void changeState(float newState)
{
  // Don't do anything if state is unchanged
  if (newState == (float)node.state)
  {
    return;
  }

  node.state = newState;

  // Update setpoint based on new state
  switch ((float)node.state)
  {
  case 0: // OFF
    setpointLux = SETPOINT_OFF;
    feedbackControl = false; // Turn off control when lights are off
    break;

  case 1: // UNOCCUPIED
    setpointLux = SETPOINT_UNOCCUPIED;
    feedbackControl = true;
    break;

  case 2: // OCCUPIED
    setpointLux = SETPOINT_OCCUPIED;
    feedbackControl = true;
    break;
  }

  // Reset PID controller to avoid integral windup during transitions
  pid.reset();

  // Update reference illuminance for metrics calculation
  refIlluminance = setpointLux;

  // Broadcast state change to network
  sendControlCommand(CAN_ADDR_BROADCAST, CAN_CTRL_STATE_CHANGE, (float)node.state);
}

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
  // Constants should be defined outside the function (moved to global constants)
  const int SAMPLES = 10;

  float samples[SAMPLES];
  float sum = 0.0;
  float count = 0.0;

  // Take multiple samples to reduce noise
  for (int i = 0; i < SAMPLES; i++)
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
  rawLux = sum / count;
  
  // Store for external light calculations
  lastMeasuredLux = rawLux;

  // If filtering is disabled, return the raw value
  if (!filterEnabled)
  {
    return rawLux;
  }

  // Apply filtering to get processed lux value
  float mean = rawLux;
  float variance = 0.0;
  
  for (int i = 0; i < SAMPLES; i++)
  {
    if (samples[i] > 0)
    { 
      variance += sq(samples[i] - mean);
    }
  }
  variance /= count;
  float stdDev = sqrt(variance);
  
  // Filter outliers
  float filteredSum = 0.0;
  float filteredCount = 0.0;
  
  for (int i = 0; i < SAMPLES; i++)
  {
    if (samples[i] > 0 && abs(samples[i] - mean) <= OUTLIER_THRESHOLD * stdDev)
    {
      filteredSum += samples[i];
      filteredCount++;
    }
  }
  
  float filteredMean = (filteredCount > 0) ? filteredSum / filteredCount : mean;
  
  // Apply EMA filter
  if (lastFilteredLux < 0)
  {
    lastFilteredLux = filteredMean; // First valid reading
  }
  else
  {
    lastFilteredLux = ALPHA * filteredMean + (1.0 - ALPHA) * lastFilteredLux;
  }
  
  // Apply calibration offset
  float processedLux = lastFilteredLux + calibrationOffset;
  if (processedLux < 0.0)
    processedLux = 0.0;
  
  // Update lastMeasuredLux for external illuminance calculations
  lastMeasuredLux = processedLux;

  return processedLux;
}

/**
 * Apply filtering techniques to raw lux readings
 * Helper function for readLux to improve code organization
 */
float applyFiltering(float samples[], float rawLux, float count)
{
  // Calculate mean and standard deviation for outlier rejection
  float mean = rawLux;
  float variance = 0.0;
  const int NUM_SAMPLES = SAMPLES;
  
  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    if (samples[i] > 0)
    { 
      variance += sq(samples[i] - mean);
    }
  }
  variance /= count;
  float stdDev = sqrt(variance);
  
  // Filter outliers
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
  
  // Apply EMA filter
  if (lastFilteredLux < 0)
  {
    lastFilteredLux = filteredMean; // First valid reading
  }
  else
  {
    lastFilteredLux = ALPHA * filteredMean + (1.0 - ALPHA) * lastFilteredLux;
  }
  
  // Apply calibration offset
  float calibratedLux = lastFilteredLux + calibrationOffset;
  if (calibratedLux < 0.0)
    calibratedLux = 0.0;
    
  return calibratedLux;
}

/**
 * Calculate external illuminance by subtracting LED contribution
 * Uses calibrated gain value G to determine LED contribution
 *
 * @return Estimated external illuminance in lux
 */
float getExternalIlluminance()
{
  // Get the last processed measurement
  float measuredLux = lastMeasuredLux;
  
  // Calculate the external illuminance 
  float ledContribution = node.dutyCycle * node.Kp;
  float currentExternalLux = max(0.0f, measuredLux - ledContribution - baselineIlluminance);
  
  // Apply temporal filtering
  if (lastExternalLux == 0.0)
  {
    externalLuxAverage = currentExternalLux;
  }
  else
  {
    externalLuxAverage = EXT_LUX_ALPHA * currentExternalLux +
                         (1.0 - EXT_LUX_ALPHA) * externalLuxAverage;
  }
  
  // Return the filtered external illuminance value (without adapting setpoint)
  return externalLuxAverage;
}

/**
 * Perform comprehensive system calibration:
 * 1. Calibrate LDR sensor accuracy
 * 2. Measure LED contribution for external illuminance calculation
 *
 * @param referenceValue The reference illuminance value (typically very low like 1.0)
 * @return Calibrated LED gain value (G)
 */
calibration calibrateSystem(float referenceValue)
{
  const int SAMPLES = 5;               // Number of measurements to average
  const int STABILIZE_TIME = 500;      // Wait time between measurements in ms
  const int LED_RESPONSE_TIME = 10000; // Wait time for LDR to respond to LED changes

  calibration result;

  Serial.println("Starting comprehensive calibration...");

  //---------------------------------------------------------------------
  // 1. First calibrate the LDR sensor for accurate absolute readings
  //---------------------------------------------------------------------
  float measuredLux = 0.0;
  const int CAL_SAMPLES = 10;

  Serial.println("Calibrating LDR sensor...");

  for (int i = 0; i < CAL_SAMPLES; i++)
  {
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
    delay(500); // Short delay between readings
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
  for (int i = 0; i < SAMPLES; i++)
  {
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

  result.gain = gain;
  result.offset = calibrationOffset;

  return result;
}

/**
 * Adapt the setpoint based on external light conditions
 * Uses external light to adjust the setpoint in adaptive mode
 */
void adaptToExternalLight()
{
  // Get current external illuminance
  float externalLux = getExternalIlluminance();
  
  // Check if there's been a significant change in external light
  if (abs(externalLux - lastExternalLux) > EXTERNAL_LIGHT_THRESHOLD)
  {
    // Adjust setpoint based on external light if adaptive mode is enabled
    if (adaptiveSetpointEnabled && (float)node.state != STATE_OFF)
    {
      float targetSetpoint = ((float)node.state == STATE_OCCUPIED) ? 
                            SETPOINT_OCCUPIED : SETPOINT_UNOCCUPIED;
      
      // Scale setpoint based on external light
      float scaledSetpoint = max(MIN_SETPOINT, 
                                targetSetpoint - externalLux * EXTERNAL_SCALE_FACTOR);
      
      // Only update if change is significant
      if (abs(scaledSetpoint - setpointLux) > MIN_SETPOINT_CHANGE)
      {
        setpointLux = scaledSetpoint;
        pid.setTarget(setpointLux);
      }
    }
    
    // Store new external light value
    lastExternalLux = externalLux;
  }
}

/**
 * Arduino setup function
 * Initializes hardware and subsystems
 */
void setup()
{
  // Initialize serial communication
  Serial.begin(115200);
  delay(1000);
  Serial.println("\nDistributed Lighting Control System");

  // Initialize hardware
  pinMode(LED_PIN, OUTPUT);
  pinMode(LDR_PIN, INPUT);

  // Debug board ID
  pico_unique_board_id_t board_id;
  pico_get_unique_board_id(&board_id);

  // Set node id in struct as last 4 bytes of board ID
  node.id = board_id.id[6] & 0x3F; // Use last 6 bits for node ID (1-63)

  // Initialize LED driver
  initLEDDriver();
  
  // Initialize CAN communication
  initCANComm();
  
  // Initialize storage for data logging
  initStorage();
  
  // Perform initial calibration
  calibration cal= calibrateSystem(1.0);

  node.Kp = cal.gain;
  node.LuxOffset = cal.offset;

  if (node.id == 33) {
    node.Ki = 400;
    node.BETA = 0.5;
    node.DT = 0.1;
  } else if (node.id == 40) {
    node.Ki = 400;
    node.BETA = 0.5;
    node.DT = 0.1;
  } else if (node.id == 52) {
    node.Ki = 400;
    node.BETA = 0.5;
    node.DT = 0.1;
  }

  node.antiWindup = true;
  node.feedbackControl = true;
  node.feedforwardControl = false;
  pid.setGains(node.Kp, node.Ki, node.BETA);
  pid.setSamplingTime(node.DT);

  node.state = STATE_OFF;
  setpointLux = SETPOINT_OFF; 
  pid.setTarget(setpointLux);

  // Initialize external light tracking
  lastExternalLux = getExternalIlluminance();
}


/**
 * Arduino main loop
 * Processes sensor readings, controls, and communication
 */
void loop()
{
  unsigned long currentTime = millis();

  float lux = readLux();
  float externalLux = getExternalIlluminance();

  // (A) Process incoming serial commands
  processSerialCommands();

  // (B) Handle any active streaming
  handleStreaming();
  handleRemoteStreamRequests();

  // (c) Update 
  if ((float)node.state != STATE_OFF)
  {
    // Process occupancy to change state if needed
    if (occupancy && (float)node.state == STATE_UNOCCUPIED)
    {
      changeState(STATE_OCCUPIED);
    }
    else if (!occupancy && (float)node.state == STATE_OCCUPIED)
    {
      changeState(STATE_UNOCCUPIED);
    }
    
    // Adapt to external light changes
    adaptToExternalLight();
  }


  // (F) Control action computation and application
  if ((float)node.state == STATE_OFF)
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
    setLEDDutyCycle(node.dutyCycle);
  }

  // After calculating a new duty cycle, update the unfiltered tracking variables
  prevDutyWithoutFilter2 = prevDutyWithoutFilter1;
  prevDutyWithoutFilter1 = node.dutyCycle;

  // (G) Log the current sample in the circular buffer
  logData(millis(), lux, node.dutyCycle);

  // (H) Process CAN messages (nonblocking)
  canCommLoop();

  // Wait for next control cycle
  delay((int)(pid.getSamplingTime() * 1000));
}