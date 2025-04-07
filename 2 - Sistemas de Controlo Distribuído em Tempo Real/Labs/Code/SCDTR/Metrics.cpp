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
// METRICS CONFIGURATION
//==========================================================================================================================================================

// Power consumption parameters
const float Pmax = 0.08755; // Maximum LED power in Watts

//==========================================================================================================================================================
// METRICS CALCULATION FUNCTIONS
//==========================================================================================================================================================

/**
 * Compute and display all lighting quality metrics
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
 */
float computeEnergyFromBuffer()
{
  int count = getLogCount();
  if (count < 2)
    return 0.0f;

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
    int idx = (startIndex + i) % LOG_SIZE;
    unsigned long currentTime = logBuffer[idx].timestamp;
    float currentDuty = logBuffer[idx].duty;

    if (first)
    {
      // Skip first sample (need two samples to calculate interval)
      first = false;
    }
    else
    {
      // Calculate time delta in seconds
      float dt = (currentTime - prevTime) / 1000.0f;

      // Calculate average power during this interval (W)
      float avgPower = Pmax * (prevDuty + currentDuty) / 2.0f;

      // Energy = Power * Time
      float energy = avgPower * dt;

      // Add to total
      totalE += energy;
    }

    // Store current values for next iteration
    prevTime = currentTime;
    prevDuty = currentDuty;
  }

  return totalE;
}

//-----------------------------------------------------------------------------
// Visibility Error Metric
//-----------------------------------------------------------------------------

/**
 * Calculate visibility error metric from illuminance history
 */
float computeVisibilityErrorFromBuffer()
{
  int count = getLogCount();
  if (count == 0)
    return 0.0f;

  // Get access to the log buffer
  LogEntry *logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;

  // Variables for error computation
  float totalErr = 0.0;
  int sampleCount = 0;

  // Iterate through all samples in the buffer
  for (int i = 0; i < count; i++)
  {
    int idx = (startIndex + i) % LOG_SIZE;
    float setpoint = logBuffer[idx].setpoint;
    float measured = logBuffer[idx].lux;

    // Only count samples where illuminance is below setpoint
    if (measured < setpoint)
    {
      totalErr += (setpoint - measured);
      sampleCount++;
    }
  }

  if (sampleCount == 0)
    return 0.0f;

  return (totalErr / sampleCount);
}

//-----------------------------------------------------------------------------
// Flicker Metric
//-----------------------------------------------------------------------------

/**
 * Calculate flicker metric from duty cycle history
 */
float computeFlickerFromBuffer()
{
  int count = getLogCount();
  if (count < 3)
    return 0.0f;

  // Get access to the log buffer
  LogEntry *logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;

  // Variables for flicker computation
  float flickerSum = 0.0;
  int flickerCount = 0;

  // Sum up all non-zero flicker values already stored in the buffer
  for (int i = 0; i < count; i++)
  {
    int idx = (startIndex + i) % LOG_SIZE;
    if (logBuffer[idx].flicker > 0.0f)
    {
      flickerSum += logBuffer[idx].flicker;
      flickerCount++;
    }
  }

  if (flickerCount == 0)
    return 0.0f;

  return (flickerSum / flickerCount);
}