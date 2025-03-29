#include "PIController.h"
#include "Globals.h"
#include <Arduino.h>
#include <math.h>

//-----------------------------------------------------------------------------
// CONFIGURATION AND VARIABLES
//-----------------------------------------------------------------------------

// Control constants
const float ANTI_WINDUP_GAIN = 0.1f;       // Anti-windup correction strength
const float FEEDFORWARD_THRESHOLD = 0.01f; // Minimum setpoint change to trigger feedforward

// PWM output limits for saturation
#define PWM_MAX 4095 // Maximum PWM value (12-bit resolution)
#define PWM_MIN 0    // Minimum PWM value (off)

// Animation and transition constants
const int MIN_TRANSITION_TIME_MS = 10;       // Minimum LED transition time
const int MIN_PULSE_DURATION_MS = 100;       // Minimum LED pulse effect duration
const int ANIMATION_UPDATE_INTERVAL_MS = 20; // Interval for animation updates

// Numeric comparison thresholds
const float FLOAT_COMPARISON_THRESHOLD = 0.001f; // For float equality comparisons

//=============================================================================
// CONSTRUCTOR AND INITIALIZATION
//=============================================================================

/**
 * PI Controller Constructor
 * Initializes a PI controller with anti-windup capabilities
 *
 * @param kp Proportional gain coefficient
 * @param ki Integral gain coefficient
 * @param beta Setpoint weighting factor (0.0-1.0)
 * @param samplingTime Control loop sampling time in seconds
 */
PIController::PIController(float kp, float ki, float beta, float samplingTime)
    : Kp(kp), Ki(ki), Beta(beta), h(samplingTime),
      Iterm(0), e_old(0), Pterm(0),
      internalTarget(0), useInternalTarget(false),
      Kff(0.8f), prevSetpoint(0), useFeedforward(true)
{
  // All variables are initialized in the initializer list
}

//=============================================================================
// CONTROL COMPUTATION
//=============================================================================

/**
 * Compute PI control action for the current sample
 *
 * Implements a discretized PI controller with setpoint weighting:
 * u(t) = Kp·(β·r(t) - y(t)) + Ki·∫e(t)dt
 *
 * Features:
 * - Setpoint weighting to reduce overshoot
 * - Back-calculation anti-windup for saturation handling
 * - Support for setpoint coordination through internal target
 *
 * @param setpoint Desired target value (setpoint)
 * @param measurement Current process value (feedback)
 * @return Control action value (typically PWM value)
 */
float PIController::compute(float setpoint, float measurement)
{
  // Get current ledGain to use as the basis for Kp
  float dynamicKp;
  critical_section_enter(&commStateLock);
  dynamicKp = deviceConfig.ledGain;
  critical_section_exit(&commStateLock);

  //-------------------------------------------------------------------------
  // Determine actual setpoint (internal or external)
  //-------------------------------------------------------------------------
  float actualSetpoint = useInternalTarget ? internalTarget : setpoint;

  //-------------------------------------------------------------------------
  // Calculate error and proportional term with setpoint weighting
  //-------------------------------------------------------------------------
  float e = actualSetpoint - measurement;                    // Standard error for integral term
  Pterm = dynamicKp * (Beta * actualSetpoint - measurement); // Use dynamic Kp

  //-------------------------------------------------------------------------
  // Calculate Feedforward Term for Step Response
  //-------------------------------------------------------------------------
  float ffTerm = 0.0f;
  if (useFeedforward)
  {
    // Only apply feedforward when setpoint changes
    float setpointChange = actualSetpoint - prevSetpoint;
    if (fabs(setpointChange) > FEEDFORWARD_THRESHOLD) // Changed abs() to fabs()
    {                                                 // Small threshold to detect real changes
      ffTerm = Kff * setpointChange;
    }
  }
  prevSetpoint = actualSetpoint; // Store for next iteration

  //-------------------------------------------------------------------------
  // Calculate Unsaturated Control Action (PI + Feedforward)
  //-------------------------------------------------------------------------
  float u_unsat = Pterm + Iterm + ffTerm;

  //-------------------------------------------------------------------------
  // Apply Anti-Windup using Back-Calculation
  //-------------------------------------------------------------------------
  float u_sat;

  // Apply saturation limits
  if (u_unsat > PWM_MAX)
  {
    u_sat = PWM_MAX;
  }
  else if (u_unsat < PWM_MIN)
  {
    u_sat = PWM_MIN;
  }
  else
  {
    u_sat = u_unsat;
  }

  // Back-calculation: adjust integral when saturated
  float saturation_error = u_sat - u_unsat;

  // Update integral term using backward Euler integration and anti-windup
  // Only apply anti-windup correction if the feature is enabled
  // Get the anti-windup flag from controlState with proper synchronization
  bool useAntiWindup;
  critical_section_enter(&commStateLock);
  useAntiWindup = controlState.antiWindup;
  critical_section_exit(&commStateLock);

  // Now use the synchronized local variable
  if (useAntiWindup)
  {
    // Apply standard integral update plus anti-windup correction
    Iterm += Ki * e * h + ANTI_WINDUP_GAIN * saturation_error;
  }
  else
  {
    // Standard integral update without anti-windup
    Iterm += Ki * e * h;
  }

  // Store error for next iteration
  e_old = e;

  // Return saturated control action
  return u_sat;
}

//=============================================================================
// CONTROLLER MANAGEMENT FUNCTIONS
//=============================================================================

/**
 * Reset controller state
 * Clears integral term and resets error history
 * Should be called when control is re-enabled after being off or when
 * making large changes to setpoint to prevent integral windup
 */
void PIController::reset()
{
  Iterm = 0;
  e_old = 0;
  Pterm = 0;
}

/**
 * Update controller gain parameters
 * Allows runtime modification of controller behavior
 *
 * @param kp New proportional gain coefficient
 * @param ki New integral gain coefficient
 */
void PIController::setGains(float kp, float ki)
{
  // We'll keep the original Kp but just store it
  // Actual Kp used will be the ledGain in the compute method
  Kp = kp;
  Ki = ki;
}

/**
 * Set setpoint weighting factor
 * Controls how setpoint changes affect proportional action
 *
 * @param beta New setpoint weighting factor (0.0-1.0)
 */
void PIController::setWeighting(float beta)
{
  Beta = beta;
}

/**
 * Get the controller's sampling time
 *
 * @return Sampling time in seconds
 */
float PIController::getSamplingTime() const
{
  return h;
}

/**
 * Set an internal coordination target
 * This allows coordination algorithms to temporarily adjust the target
 * without changing the user-defined setpoint
 *
 * @param newTarget New internally managed setpoint value
 */
void PIController::setTarget(float newTarget)
{
  internalTarget = newTarget;
  useInternalTarget = true;
}

/**
 * Restore using the external setpoint
 * Disables the internal target and reverts to using the setpoint
 * passed directly to the compute() method
 */
void PIController::clearInternalTarget()
{
  useInternalTarget = false;
}

/**
 * Check if controller is currently using an internal target
 *
 * @return true if using internal target, false if using external setpoint
 */
bool PIController::isUsingInternalTarget() const
{
  return useInternalTarget;
}

/**
 * Get current values of PI terms for diagnostics
 *
 * @param p Output parameter for proportional term
 * @param i Output parameter for integral term
 */
void PIController::getTerms(float &p, float &i) const
{
  // For the most recent control calculation:
  p = Pterm; // Proportional term
  i = Iterm; // Integral term
}

/**
 * Enable or disable feedforward control
 *
 * @param enable True to enable feedforward, false to disable
 * @param ffGain Feedforward gain to be used when enabled
 */
void PIController::enableFeedforward(bool enable, float ffGain)
{
  useFeedforward = enable;
  if (enable)
  {
    Kff = ffGain;
  }
}

/******************************************************************************************************************************************************************************************************************************************************************************** */
// CIRCULAR BUFFER SECTION
/******************************************************************************************************************************************************************************************************************************************************************************** */

//=============================================================================
// CIRCULAR BUFFER DATA STORAGE
//=============================================================================

// Circular buffer for storing time-series data
LogEntry logBuffer[LOG_SIZE];

// Current write position in the buffer
int logIndex = 0;

// Flag indicating if buffer has wrapped around (contains LOG_SIZE entries)
bool bufferFull = false;

// Sample counter for downsampling
unsigned int sampleCounter = 0;

static unsigned long lastSampleMicros = 0;

// Downsampling rate - only store every Nth sample
const unsigned int DOWNSAMPLE_RATE = 10;

extern float calculateFlickerValue(float d0, float d1, float d2);
static float lastDuty = -1.0; // -1.0 indicates that no previous value exists yet

// Track previous duty cycles for flicker calculation
static float prevDuty1 = 0.0;
static float prevDuty2 = 0.0;
static bool enoughSamplesForFlicker = false;
static float cumulativeFlicker = 0.0; // Track the running sum of flicker values

//=============================================================================
// INITIALIZATION FUNCTIONS
//=============================================================================

/**
 * Initialize the storage system
 * Resets buffer position and full flag
 */
void initStorage()
{
  logIndex = 0;
  bufferFull = false;
  sampleCounter = 0;
  cumulativeFlicker = 0.0; // Reset the cumulative flicker
  lastDuty = -1.0;
}

//=============================================================================
// DATA LOGGING FUNCTIONS
//=============================================================================

// This function is called at every loop iteration or at some periodic rate
void logData(unsigned long timestampMs, float lux, float duty)
{
  // Validate
  if (isnan(lux) || isnan(duty))
  {
    return;
  }

  // We only store data every DOWNSAMPLE_RATE calls
  sampleCounter++;
  if (sampleCounter < DOWNSAMPLE_RATE)
  {
    return;
  }
  sampleCounter = 0;

  // Compute instantaneous flicker error:
  // If this is the first sample, we cannot compute flicker.
  float flickerError = 0.0;
  if (lastDuty >= 0.0)
  {
    // Simply compute the absolute difference between the current and last duty cycle.
    flickerError = fabs(duty - lastDuty);
  }
  // Update lastDuty for next computation
  lastDuty = duty;

  // Compute jitter:
  unsigned long nowMicros = micros();
  float jitterUs = 0.0f;
  if (lastSampleMicros != 0)
  {
    unsigned long deltaMicros = nowMicros - lastSampleMicros;
    const float nominalPeriodUs = 10000.0f; // for a 10 ms period
    jitterUs = (float)deltaMicros - nominalPeriodUs;
  }
  lastSampleMicros = nowMicros;

  // Get external illuminance
  float externalLux = getExternalIlluminance();

  // Save the data into the log buffer:
  critical_section_enter(&commStateLock);
  logBuffer[logIndex].timestamp = timestampMs;
  logBuffer[logIndex].lux = lux;
  logBuffer[logIndex].duty = duty;
  logBuffer[logIndex].setpoint = controlState.setpointLux;
  logBuffer[logIndex].flicker = flickerError;
  logBuffer[logIndex].jitter = jitterUs;
  logBuffer[logIndex].extLux = externalLux; // Store external illuminance
  critical_section_exit(&commStateLock);

  // Advance circular buffer index
  logIndex++;
  if (logIndex >= LOG_SIZE)
  {
    logIndex = 0;
    bufferFull = true;
  }
}

// Example “mdump” or “dumpBufferToSerial” function
void dumpBufferToSerial()
{
  // Print CSV header with new jitter column
  Serial.println("timestamp_ms,rawLux,duty,jitter_us");

  int count = bufferFull ? LOG_SIZE : logIndex;
  int startIndex = bufferFull ? logIndex : 0;

  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    unsigned long t = logBuffer[realIndex].timestamp;
    float lx = logBuffer[realIndex].lux;
    float d = logBuffer[realIndex].duty;
    float j = logBuffer[realIndex].jitter;

    // Print CSV row
    Serial.print(t);
    Serial.print(",");
    Serial.print(lx, 2);
    Serial.print(",");
    Serial.print(d, 4);
    Serial.print(",");
    Serial.println(j, 4);
  }
  Serial.println("End of mdump.\n");
}

//=============================================================================
// DATA ACCESS FUNCTIONS
//=============================================================================

/**
 * Get direct access to the log buffer array
 * Use with caution - returns pointer to the actual buffer
 *
 * @return Pointer to the log buffer array
 */
LogEntry *getLogBuffer()
{
  return logBuffer;
}

/**
 * Get the number of valid entries in the buffer
 *
 * @return Number of entries (maximum LOG_SIZE)
 */
int getLogCount()
{
  return bufferFull ? LOG_SIZE : logIndex;
}

/**
 * Check if the buffer has filled completely at least once
 *
 * @return true if buffer has wrapped around, false otherwise
 */
bool isBufferFull()
{
  return bufferFull;
}

/**
 * Get current write position in buffer
 *
 * @return Current buffer index
 */
int getCurrentIndex()
{
  return logIndex;
}

//=============================================================================
// ADDITIONAL UTILITY FUNCTIONS
//=============================================================================

/**
 * Clear all data in the buffer
 * Resets buffer to empty state
 */
void clearBuffer()
{
  logIndex = 0;
  bufferFull = false;
}

/**
 * Get the oldest timestamp in the buffer
 *
 * @return Timestamp of oldest entry or 0 if buffer is empty
 */
unsigned long getOldestTimestamp()
{
  if (getLogCount() == 0)
  {
    return 0; // Buffer is empty
  }

  int startIndex = bufferFull ? logIndex : 0;
  return logBuffer[startIndex].timestamp;
}

/**
 * Get the newest timestamp in the buffer
 *
 * @return Timestamp of newest entry or 0 if buffer is empty
 */
unsigned long getNewestTimestamp()
{
  if (getLogCount() == 0)
  {
    return 0; // Buffer is empty
  }

  int newestIndex = (logIndex == 0) ? (bufferFull ? LOG_SIZE - 1 : 0) : (logIndex - 1);
  return logBuffer[newestIndex].timestamp;
}

/**
 * Calculate buffer duration in milliseconds
 *
 * @return Time span covered by buffer entries or 0 if fewer than 2 entries
 */
unsigned long getBufferDuration()
{
  if (getLogCount() < 2)
  {
    return 0; // Need at least 2 entries to calculate duration
  }

  return getNewestTimestamp() - getOldestTimestamp();
}

/**
 * Find the closest data entry to a given timestamp
 *
 * @param timestamp Target timestamp to search for
 * @return Index of closest entry or -1 if buffer is empty
 */
int findClosestEntry(unsigned long timestamp)
{
  if (getLogCount() == 0)
  {
    return -1; // Return -1 if the buffer is empty
  }

  int count = getLogCount();
  int closestIndex = 0;

  // For unsigned types, calculate difference safely
  unsigned long closestDiff;
  if (logBuffer[closestIndex].timestamp >= timestamp)
  {
    closestDiff = logBuffer[closestIndex].timestamp - timestamp;
  }
  else
  {
    closestDiff = timestamp - logBuffer[closestIndex].timestamp;
  }

  // Iterate through all entries to find the closest one
  for (int i = 1; i < count; i++)
  {
    int realIndex = (bufferFull) ? (logIndex + i) % LOG_SIZE : i;

    // Calculate absolute difference safely for unsigned types
    unsigned long diff;
    if (logBuffer[realIndex].timestamp >= timestamp)
    {
      diff = logBuffer[realIndex].timestamp - timestamp;
    }
    else
    {
      diff = timestamp - logBuffer[realIndex].timestamp;
    }

    if (diff < closestDiff)
    {
      closestDiff = diff;
      closestIndex = realIndex;
    }
  }

  return closestIndex;
}
/******************************************************************************************************************************************************************************************************************************************************************************** */
// PERFORMANCE METRICS SECTION
/******************************************************************************************************************************************************************************************************************************************************************************** */

//=============================================================================
// PERFORMANCE METRICS CONFIGURATION
//=============================================================================

// Power consumption parameters
const float Pmax = 0.08755; // Maximum LED power in Watts

// External references
extern float controlState.setpointLux; // Reference illuminance target from main.ino

//=============================================================================
// METRICS COMPUTATION AND REPORTING
//=============================================================================

/**
 * Compute and display all lighting quality metrics
 * Calculates energy usage, visibility error, and flicker from logged data
 * and outputs the results to the serial console
 */
void computeAndPrintMetrics()
{
  float E = computeEnergyFromBuffer();
  float VE = computeVisibilityErrorFromBuffer();
  float F = computeFlickerFromBuffer();

  Serial.println("----- Metrics from Circular Buffer -----");
  Serial.print("Energy (J approx): ");
  Serial.println(E, 4);
  Serial.print("Visibility Error (lux): ");
  Serial.println(VE, 2);
  Serial.print("Flicker: ");
  Serial.println(F, 4);
  Serial.println("----------------------------------------\n");
}

//-----------------------------------------------------------------------------
// Energy Consumption Metric
//-----------------------------------------------------------------------------

/**
 * Calculate energy consumption from duty cycle history
 *
 * Energy is computed by integrating power over time:
 * E = ∫ P(t) dt
 *
 * Since we have discrete samples, we use:
 * E = Σ (P × Δt)
 *
 * where P = Pmax × duty_cycle
 *
 * @return Total energy consumption in joules
 */
float computeEnergyFromBuffer()
{
  int count = getLogCount();
  if (count < 2)
    return 0.0; // Need at least 2 samples for time difference

  // Get access to the log buffer
  LogEntry *logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;

  // Variables for energy computation
  unsigned long prevTime = 0;
  float prevDuty = 0.0;
  bool first = true;
  float totalE = 0.0;

  // Iterate through all samples in the buffer
  for (int i = 0; i < count; i++)
  {
    // Calculate the actual index in the circular buffer
    int realIndex = (startIndex + i) % LOG_SIZE;

    // Get timestamp and duty cycle for current sample
    unsigned long t = logBuffer[realIndex].timestamp;
    float d = logBuffer[realIndex].duty;

    if (!first)
    {
      // Calculate time delta in seconds
      float dt = (t - prevTime) / 1000.0;

      // Energy = Power × Time
      // Power = Pmax × Duty cycle
      totalE += (Pmax * prevDuty * dt);
    }
    else
    {
      // Skip first sample (need two points for time difference)
      first = false;
    }

    // Save current values for next iteration
    prevTime = t;
    prevDuty = d;
  }

  return totalE;
}

//-----------------------------------------------------------------------------
// Visibility Error Metric
//-----------------------------------------------------------------------------

/**
 * Calculate visibility error metric from illuminance history
 *
 * Visibility error measures how much the illuminance falls below
 * the setpoint over time. It's the average of (setpoint - measured)
 * when measured < setpoint, otherwise 0.
 *
 * This metric represents insufficient lighting conditions.
 *
 * @return Average visibility error in lux
 */
float computeVisibilityErrorFromBuffer()
{
  int count = getLogCount();
  if (count == 0)
    return 0.0; // No data available

  // Get access to the log buffer
  LogEntry *logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;

  // Variables for error computation
  float totalErr = 0.0;
  int sampleCount = 0;

  // Iterate through all samples in the buffer
  for (int i = 0; i < count; i++)
  {
    // Calculate the actual index in the circular buffer
    int realIndex = (startIndex + i) % LOG_SIZE;

    // Get illuminance measurement
    float measuredLux = logBuffer[realIndex].lux;

    // Only accumulate error when below setpoint
    // (we care about insufficient lighting, not excess)
    critical_section_enter(&commStateLock);
    if (measuredLux < controlState.setpointLux)
    {
      totalErr += (controlState.setpointLux - measuredLux);
    }
    critical_section_exit(&commStateLock);

    sampleCount++;
  }

  if (sampleCount == 0)
    return 0.0;
  return (totalErr / sampleCount); // Average error
}

//-----------------------------------------------------------------------------
// Flicker Metric
//-----------------------------------------------------------------------------

/**
 * Calculate flicker metric from duty cycle history
 *
 * Flicker is computed by detecting direction changes in the
 * duty cycle signal, which indicate oscillations. The method uses
 * three consecutive points to detect when the slope changes sign
 * (indicating a potential oscillation), and measures the magnitude
 * of these changes.
 *
 * @return Average flicker magnitude when direction changes
 */
float computeFlickerFromBuffer()
{
  int count = getLogCount();
  if (count < 3)
    return 0.0; // Need at least 3 samples for flicker

  // Get access to the log buffer
  LogEntry *logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;

  // Variables for flicker computation
  float flickerSum = 0.0;
  int flickerCount = 0;

  // Sum up all non-zero flicker values
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;

    if (logBuffer[realIndex].flicker > 0.0)
    {
      flickerSum += logBuffer[realIndex].flicker;
      flickerCount++;
    }
  }

  if (flickerCount == 0)
    return 0.0;
  return (flickerSum / flickerCount); // Return average for compatibility
}

/**
 * Calculate flicker value for the current point
 * Based on the last three duty cycle values
 *
 * @param d0 First duty cycle point
 * @param d1 Second duty cycle point
 * @param d2 Third duty cycle point (current)
 * @return Flicker value or 0.0 if no direction change
 */
float calculateFlickerValue(float d0, float d1, float d2)
{
  // Calculate slopes between consecutive points
  float diff1 = d1 - d0; // Slope between first and second points
  float diff2 = d2 - d1; // Slope between second and third points

  // Detect direction change (sign change in slopes)
  if (diff1 * diff2 < 0.0)
  {
    // Return the magnitude of the changes
    return (fabs(diff1) + fabs(diff2));
  }

  // No direction change detected
  return 0.0;
}

//=============================================================================
// ADDITIONAL METRICS
//=============================================================================

/**
 * Calculate the duty cycle stability metric
 * Measures how stable the duty cycle remains over time
 * Lower values indicate better stability
 *
 * @return Standard deviation of duty cycle
 */
float computeDutyStabilityFromBuffer()
{
  int count = getLogCount();
  if (count < 2)
    return 0.0;

  // Get access to the log buffer
  LogEntry *logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;

  // First pass: calculate mean
  float sum = 0.0;
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    sum += logBuffer[realIndex].duty;
  }
  float mean = sum / count;

  // Second pass: calculate variance
  float variance = 0.0;
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    float diff = logBuffer[realIndex].duty - mean;
    variance += diff * diff;
  }
  variance /= count;

  // Return standard deviation
  return sqrt(variance);
}

/**
 * Calculate overall lighting quality index
 * Combines energy, visibility error, and flicker into a single metric
 * Higher values indicate better overall performance
 *
 * @return Quality index from 0 (worst) to 100 (best)
 */
float computeQualityIndex()
{
  // Get individual metrics
  float energy = computeEnergyFromBuffer();
  float visibilityError = computeVisibilityErrorFromBuffer();
  float flicker = computeFlickerFromBuffer();

  // Normalize energy (lower is better)
  // Assuming typical range of 0-10 joules for a minute of operation
  float energyScore = 100 * (1.0 - constrain(energy / 10.0, 0.0, 1.0));

  // Normalize visibility error (lower is better)
  // Assuming typical range of 0-10 lux error
  float visibilityScore = 100 * (1.0 - constrain(visibilityError / 10.0, 0.0, 1.0));

  // Normalize flicker (lower is better)
  // Assuming typical range of 0-0.2 flicker
  float flickerScore = 100 * (1.0 - constrain(flicker / 0.2, 0.0, 1.0));

  // Weighted average (prioritize visibility, then flicker, then energy)
  return (0.5 * visibilityScore + 0.3 * flickerScore + 0.2 * energyScore);
}