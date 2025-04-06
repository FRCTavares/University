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
  critical_section_enter_blocking(&commStateLock);
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

/**
 * Output all logged data as CSV to the serial port
 */
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
