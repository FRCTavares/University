#include "Metrics.h"
#include "Storage.h"
#include <Arduino.h>
#include <math.h>

//=============================================================================
// PERFORMANCE METRICS CONFIGURATION
//=============================================================================

// Power consumption parameters
const float Pmax = 1.0;  // Maximum LED power in Watts

// External references
extern float setpointLux; // Reference illuminance target from main.ino

//=============================================================================
// METRICS COMPUTATION AND REPORTING
//=============================================================================

/**
 * Compute and display all lighting quality metrics
 * Calculates energy usage, visibility error, and flicker from logged data
 * and outputs the results to the serial console
 */
void computeAndPrintMetrics() {
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
float computeEnergyFromBuffer() {
  int count = getLogCount();
  if (count < 2) return 0.0;  // Need at least 2 samples for time difference

  // Get access to the log buffer
  LogEntry* logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;
  
  // Variables for energy computation
  unsigned long prevTime = 0;
  float prevDuty = 0.0;
  bool first = true;
  float totalE = 0.0;

  // Iterate through all samples in the buffer
  for (int i = 0; i < count; i++) {
    // Calculate the actual index in the circular buffer
    int realIndex = (startIndex + i) % LOG_SIZE;
    
    // Get timestamp and duty cycle for current sample
    unsigned long t = logBuffer[realIndex].timestamp;
    float d = logBuffer[realIndex].duty;
    
    if (!first) {
      // Calculate time delta in seconds
      float dt = (t - prevTime) / 1000.0;
      
      // Energy = Power × Time
      // Power = Pmax × Duty cycle
      totalE += (Pmax * prevDuty * dt);
    } else {
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
float computeVisibilityErrorFromBuffer() {
  int count = getLogCount();
  if (count == 0) return 0.0;  // No data available

  // Get access to the log buffer
  LogEntry* logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;
  
  // Variables for error computation
  float totalErr = 0.0;
  int sampleCount = 0;

  // Iterate through all samples in the buffer
  for (int i = 0; i < count; i++) {
    // Calculate the actual index in the circular buffer
    int realIndex = (startIndex + i) % LOG_SIZE;
    
    // Get illuminance measurement
    float measuredLux = logBuffer[realIndex].lux;
    
    // Only accumulate error when below setpoint
    // (we care about insufficient lighting, not excess)
    if (measuredLux < setpointLux) {
      totalErr += (setpointLux - measuredLux);
    }
    
    sampleCount++;
  }
  
  if (sampleCount == 0) return 0.0;
  return (totalErr / sampleCount);  // Average error
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
float computeFlickerFromBuffer() {
  int count = getLogCount();
  if (count < 3) return 0.0;  // Need at least 3 samples for flicker

  // Get access to the log buffer
  LogEntry* logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;
  
  // Variables for flicker computation
  float flickerSum = 0.0;  // Sum of flicker magnitudes
  int flickerCount = 0;    // Count of flicker events

  // Need three consecutive points to detect direction changes
  bool first = true, second = false;
  float d0, d1;  // First two points in sliding window

  // Iterate through all samples in the buffer
  for (int i = 0; i < count; i++) {
    // Calculate the actual index in the circular buffer
    int realIndex = (startIndex + i) % LOG_SIZE;
    
    // Get duty cycle for current sample (third point in window)
    float d2 = logBuffer[realIndex].duty;
    
    if (first) {
      // Initialize first point
      d0 = d2;
      first = false;
      second = false;
    }
    else if (!second) {
      // Initialize second point
      d1 = d2;
      second = true;
    }
    else {
      // Calculate slopes between consecutive points
      float diff1 = d1 - d0;  // Slope between first and second points
      float diff2 = d2 - d1;  // Slope between second and third points
      
      // Detect direction change (sign change in slopes)
      // This indicates an oscillation/flicker
      if (diff1 * diff2 < 0.0) {
        // Add the magnitude of the changes to flicker sum
        flickerSum += (fabs(diff1) + fabs(diff2));
        flickerCount++;
      }
      
      // Slide window: shift points for next iteration
      d0 = d1;
      d1 = d2;
    }
  }
  
  if (flickerCount == 0) return 0.0;
  return (flickerSum / flickerCount);  // Average flicker magnitude
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
float computeDutyStabilityFromBuffer() {
  int count = getLogCount();
  if (count < 2) return 0.0;

  // Get access to the log buffer
  LogEntry* logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;
  
  // First pass: calculate mean
  float sum = 0.0;
  for (int i = 0; i < count; i++) {
    int realIndex = (startIndex + i) % LOG_SIZE;
    sum += logBuffer[realIndex].duty;
  }
  float mean = sum / count;
  
  // Second pass: calculate variance
  float variance = 0.0;
  for (int i = 0; i < count; i++) {
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
float computeQualityIndex() {
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