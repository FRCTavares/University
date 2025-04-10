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

//==========================================================================================================================================================
// BUFFER CONFIGURATION AND VARIABLES
//==========================================================================================================================================================

// Circular buffer for storing time-series data
LogEntry logBuffer[LOG_SIZE];

// Current write position in the buffer
int logIndex = 0;

// Flag indicating if buffer has wrapped around (contains LOG_SIZE entries)
bool bufferFull = false;

// Sample counter for downsampling
unsigned int sampleCounter = 0;

// Timestamp tracking for jitter calculation
static unsigned long lastSampleMicros = 0;

// Downsampling rate - only store every Nth sample
const unsigned int DOWNSAMPLE_RATE = 10;

// Flicker calculation variables
static float lastDuty = -1.0; // -1.0 indicates that no previous value exists yet
static float prevDuty1 = 0.0;
static float prevDuty2 = 0.0;
static bool enoughSamplesForFlicker = false;
static float cumulativeFlicker = 0.0; // Track the running sum of flicker values

//==========================================================================================================================================================
// BUFFER INITIALIZATION
//==========================================================================================================================================================

/**
 * Initialize the storage system
 * Resets buffer position and full flag
 */
void initStorage()
{
  logIndex = 0;
  bufferFull = false;
  sampleCounter = 0;
  cumulativeFlicker = 0.0;
  lastDuty = -1.0;
  lastSampleMicros = 0;
}

//==========================================================================================================================================================
// DATA LOGGING AND ACCESS
//==========================================================================================================================================================

static float totalEnergy = 0.0f;
static float totalVisError = 0.0f;
static float totalFlicker = 0.0f;
static int metricSampleCount = 0;

/**
 * Log a data point to the circular buffer
 * Stores timestamp, illuminance, duty cycle, and calculated metrics
 */
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
  float flickerError = 0.0;
  if (lastDuty >= 0.0)
  {
    // Simply compute the absolute difference between the current and last duty cycle.
    flickerError = fabs(duty - lastDuty);
  }
  // Update lastDuty for next computation
  lastDuty = duty;

  // Get external illuminance
  float externalLux = getExternalIlluminance();

  // Calculate metrics for running averages
  metricSampleCount++;

  // Energy calculation (approximate incremental update)
  critical_section_enter_blocking(&commStateLock);
  float currentSetpoint = controlState.setpointLux;
  critical_section_exit(&commStateLock);

  // Add to energy accumulation (approximate from current duty)
  // Using MAX_POWER_WATTS for power conversion
  float deltaT = DOWNSAMPLE_RATE * 0.01f; // Assuming 10ms control loop × downsampling
  float incrementalEnergy = duty * MAX_POWER_WATTS * deltaT;
  totalEnergy += incrementalEnergy;

  // Calculate visibility error - only when lux is below setpoint
  float visibilityError = 0.0f;
  if (lux < currentSetpoint)
  {
    visibilityError = currentSetpoint - lux;
  }
  totalVisError += visibilityError;

  // Add to flicker accumulation
  totalFlicker += flickerError;

  // Calculate running averages
  float avgEnergy = totalEnergy; // Total energy consumed so far
  float avgVisError = totalVisError / metricSampleCount;
  float avgFlicker = totalFlicker / metricSampleCount;

  // Save the data into the log buffer:
  critical_section_enter_blocking(&commStateLock);
  logBuffer[logIndex].timestamp = timestampMs;
  logBuffer[logIndex].lux = lux;
  logBuffer[logIndex].duty = duty;
  logBuffer[logIndex].setpoint = currentSetpoint;
  logBuffer[logIndex].flicker = flickerError;
  logBuffer[logIndex].avgEnergy = avgEnergy;
  logBuffer[logIndex].avgVisError = avgVisError;
  logBuffer[logIndex].avgFlicker = avgFlicker;
  logBuffer[logIndex].extLux = externalLux;
  critical_section_exit(&commStateLock);

  // Advance circular buffer index
  logIndex++;
  if (logIndex >= LOG_SIZE)
  {
    logIndex = 0;
    bufferFull = true;
  }
}

/**
 * Output all logged data as CSV to the serial port
 */
void dumpBufferToSerial()
{
  int count = bufferFull ? LOG_SIZE : logIndex;
  int startIndex = bufferFull ? logIndex : 0;

  // Print timestamp_ms row
  Serial.print("timestamp_ms");
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    Serial.print(",");
    Serial.print(logBuffer[realIndex].timestamp);
  }
  Serial.println();

  // Print lux row
  Serial.print("lux");
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    Serial.print(",");
    Serial.print(logBuffer[realIndex].lux, 2);
  }
  Serial.println();

  // Print duty row
  Serial.print("duty");
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    Serial.print(",");
    Serial.print(logBuffer[realIndex].duty, 4);
  }
  Serial.println();

  // Print setpoint row
  Serial.print("setpoint");
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    Serial.print(",");
    Serial.print(logBuffer[realIndex].setpoint, 2);
  }
  Serial.println();

  // Print flicker row
  Serial.print("flicker");
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    Serial.print(",");
    Serial.print(logBuffer[realIndex].flicker, 4);
  }
  Serial.println();

  // Print avg_energy row (replaced jitter_us)
  Serial.print("avg_energy");
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    Serial.print(",");
    Serial.print(logBuffer[realIndex].avgEnergy, 6);
  }
  Serial.println();

  // Print avg_vis_error row
  Serial.print("avg_vis_error");
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    Serial.print(",");
    Serial.print(logBuffer[realIndex].avgVisError, 4);
  }
  Serial.println();

  // Print avg_flicker row
  Serial.print("avg_flicker");
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    Serial.print(",");
    Serial.print(logBuffer[realIndex].avgFlicker, 6);
  }
  Serial.println();

  // Print ext_lux row
  Serial.print("ext_lux");
  for (int i = 0; i < count; i++)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;
    Serial.print(",");
    Serial.print(logBuffer[realIndex].extLux, 2);
  }
  Serial.println();
}

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

/**
 * Clear all data in the buffer
 * Resets buffer to empty state
 */
void clearBuffer()
{
  logIndex = 0;
  bufferFull = false;
}
