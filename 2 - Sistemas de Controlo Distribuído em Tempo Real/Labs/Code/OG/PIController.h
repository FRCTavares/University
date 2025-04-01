#ifndef PICONTROLLER_H
#define PICONTROLLER_H

#include <Arduino.h>
#include <math.h>

/**
 * PI Controller Class
 *
 * Implements a discrete-time PI controller with:
 * - Setpoint weighting for improved disturbance rejection
 * - Back-calculation anti-windup to prevent integral saturation
 * - Support for setpoint coordination between nodes
 * - Configurable gains and sampling time
 *
 * This controller is designed for real-time control applications where
 * robustness and predictable performance are required.
 */
class PIController
{
public:
  //=========================================================================
  // CONSTRUCTOR AND CONFIGURATION
  //=========================================================================

  /**
   * PI Controller Constructor
   * Initializes a PI controller with anti-windup capabilities
   *
   * @param kp Proportional gain coefficient
   * @param ki Integral gain coefficient
   * @param beta Setpoint weighting factor (0.0-1.0)
   * @param samplingTime Control loop sampling time in seconds
   */
  PIController(float kp, float ki, float beta, float samplingTime);

  /**
   * Update controller gain parameters
   * Allows runtime modification of controller behavior
   *
   * @param kp New proportional gain coefficient
   * @param ki New integral gain coefficient
   */
  void setGains(float kp, float ki);

  /**
   * Set setpoint weighting factor
   * Controls how setpoint changes affect proportional action
   *
   * @param beta New setpoint weighting factor (0.0-1.0)
   */
  void setWeighting(float beta);

  /**
   * Get the controller's sampling time
   *
   * @return Sampling time in seconds
   */
  float getSamplingTime() const;

  //=========================================================================
  // CONTROL COMPUTATION
  //=========================================================================

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
  float compute(float setpoint, float measurement);

  /**
   * Reset controller state
   * Clears integral term and resets error history
   * Should be called when control is re-enabled after being off or when
   * making large changes to setpoint to prevent integral windup
   */
  void reset();

  //=========================================================================
  // COORDINATION SUPPORT
  //=========================================================================

  /**
   * Set an internal coordination target
   * This allows coordination algorithms to temporarily adjust the target
   * without changing the user-defined setpoint
   *
   * @param newTarget New internally managed setpoint value
   */
  void setTarget(float newTarget);

  /**
   * Restore using the external setpoint
   * Disables the internal target and reverts to using the setpoint
   * passed directly to the compute() method
   */
  void clearInternalTarget();

  /**
   * Check if controller is currently using an internal target
   *
   * @return true if using internal target, false if using external setpoint
   */
  bool isUsingInternalTarget() const;

  //=========================================================================
  // DIAGNOSTIC FUNCTIONS
  //=========================================================================

  /**
   * Get current values of PI terms for diagnostics
   *
   * @param p Output parameter for proportional term
   * @param i Output parameter for integral term
   */
  void getTerms(float &p, float &i) const;

  /**
   * Enable or disable feedforward control
   *
   * @param enable True to enable feedforward, false to disable
   * @param ffGain Feedforward gain to be used when enabled
   */
  void enableFeedforward(bool enable, float ffGain);

private:
  //=========================================================================
  // CONTROLLER PARAMETERS
  //=========================================================================

  float Kp;   // Proportional gain
  float Ki;   // Integral gain
  float Beta; // Setpoint weighting factor
  float h;    // Sampling time in seconds

  //=========================================================================
  // CONTROLLER STATE
  //=========================================================================

  float Iterm; // Accumulated integral term
  float e_old; // Previous error for integration
  float Pterm; // Last proportional term (for diagnostics)

  //=========================================================================
  // COORDINATION VARIABLES
  //=========================================================================

  float internalTarget;   // Target value used for coordination
  bool useInternalTarget; // Flag to use internal or external target

  // Add these new variables
  float Kff;           // Feedforward gain
  float prevSetpoint;  // Previous setpoint for detecting changes
  bool useFeedforward; // Enable/disable feedforward
};

/******************************************************************************************************************************************************************************************************************************************************************************** */
// CIRCULAR BUFFER SECTION
/******************************************************************************************************************************************************************************************************************************************************************************** */

//=============================================================================
// CONFIGURATION
//=============================================================================

/**
 * Maximum number of data points to store in the circular buffer
 * Defines the memory usage and time window of historical data
 */
#define LOG_SIZE 1000

//=============================================================================
// DATA STRUCTURES
//=============================================================================

/**
 * Data structure for a single time-series log entry
 * Each entry stores a timestamp and associated sensor/control values
 */
struct LogEntry
{
  unsigned long timestamp; // Time when entry was recorded (ms)
  float lux;               // Measured illuminance value (lux)
  float duty;              // LED duty cycle (0.0-1.0)
  float setpoint;          // Reference illuminance target
  float flicker;           // Individual flicker value at this point
  float jitter;            // Jitter in µs (difference from nominal period)
  float extLux;            // External illuminance
};

//=============================================================================
// INITIALIZATION FUNCTIONS
//=============================================================================

/**
 * Initialize the storage system
 * Resets buffer position and full flag
 */
void initStorage();

//=============================================================================
// DATA LOGGING FUNCTIONS
//=============================================================================

/**
 * Log a data point to the circular buffer
 * Stores timestamp, illuminance, and duty cycle values
 *
 * @param timestamp Millisecond timestamp when data was captured
 * @param lux Measured illuminance in lux
 * @param duty LED duty cycle (0.0-1.0)
 */
void logData(unsigned long timestamp, float lux, float duty);

/**
 * Output all logged data as CSV to the serial port
 * Formats: timestamp_ms,rawLux,duty
 */
void dumpBufferToSerial();

//=============================================================================
// DATA ACCESS FUNCTIONS
//=============================================================================

/**
 * Get direct access to the log buffer array
 * Use with caution - returns pointer to the actual buffer
 *
 * @return Pointer to the log buffer array
 */
LogEntry *getLogBuffer();

/**
 * Get the number of valid entries in the buffer
 *
 * @return Number of entries (maximum LOG_SIZE)
 */
int getLogCount();

/**
 * Check if the buffer has filled completely at least once
 *
 * @return true if buffer has wrapped around, false otherwise
 */
bool isBufferFull();

/**
 * Get current write position in buffer
 *
 * @return Current buffer index
 */
int getCurrentIndex();

//=============================================================================
// ADDITIONAL UTILITY FUNCTIONS
//=============================================================================

/**
 * Clear all data in the buffer
 * Resets buffer to empty state
 */
void clearBuffer();

/**
 * Get the oldest timestamp in the buffer
 *
 * @return Timestamp of oldest entry or 0 if buffer is empty
 */
unsigned long getOldestTimestamp();

/**
 * Get the newest timestamp in the buffer
 *
 * @return Timestamp of newest entry or 0 if buffer is empty
 */
unsigned long getNewestTimestamp();

/**
 * Calculate buffer duration in milliseconds
 *
 * @return Time span covered by buffer entries or 0 if fewer than 2 entries
 */
unsigned long getBufferDuration();

/**
 * Find the closest data entry to a given timestamp
 *
 * @param timestamp Target timestamp to search for
 * @return Index of closest entry or -1 if buffer is empty
 */
int findClosestEntry(unsigned long timestamp);

/******************************************************************************************************************************************************************************************************************************************************************************** */
// PERFORMANCE METRICS SECTION
/******************************************************************************************************************************************************************************************************************************************************************************** */

/**
 * Lighting System Performance Metrics Module
 *
 * This module provides tools for evaluating lighting system performance through
 * various metrics that capture energy efficiency, lighting quality, and comfort.
 * It processes historical data from the circular buffer to calculate:
 * - Energy consumption (power over time)
 * - Visibility error (insufficient illuminance detection)
 * - Flicker (unwanted lighting oscillations)
 * - Overall quality metrics
 *
 * These metrics enable objective evaluation of system performance and can guide
 * parameter tuning and optimization.
 */

//=============================================================================
// PRIMARY METRICS FUNCTIONS
//=============================================================================

/**
 * Compute and display all lighting quality metrics
 * Calculates energy usage, visibility error, and flicker from logged data
 * and outputs the results to the serial console
 */
void computeAndPrintMetrics();

//=============================================================================
// ENERGY EFFICIENCY METRICS
//=============================================================================

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
float computeEnergyFromBuffer();

/**
 * Calculate average power consumption over the logged period
 *
 * @return Average power consumption in watts
 */
float computeAveragePowerFromBuffer();

/**
 * Calculate peak power consumption in the logged period
 *
 * @return Maximum power consumption in watts
 */
float computePeakPowerFromBuffer();

//=============================================================================
// LIGHTING QUALITY METRICS
//=============================================================================

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
float computeVisibilityErrorFromBuffer();

/**
 * Calculate illuminance stability metric
 *
 * Measures how stable the illuminance level remains over time
 * Lower values indicate more stable illuminance
 *
 * @return Standard deviation of illuminance
 */
float computeIlluminanceStabilityFromBuffer();

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
float computeFlickerFromBuffer();

//=============================================================================
// COMBINED QUALITY METRICS
//=============================================================================

/**
 * Calculate duty cycle stability metric
 *
 * Measures how stable the duty cycle remains over time
 * Lower values indicate better stability
 *
 * @return Standard deviation of duty cycle
 */
float computeDutyStabilityFromBuffer();

/**
 * Calculate overall lighting quality index
 *
 * Combines energy, visibility error, and flicker into a single metric
 * Higher values indicate better overall performance
 *
 * @return Quality index from 0 (worst) to 100 (best)
 */
float computeQualityIndex();

/**
 * Calculate comfort metric based on illuminance stability and flicker
 *
 * @return Comfort rating from 0 (poor) to 100 (excellent)
 */
float computeComfortMetric();

//=============================================================================
// STATISTICAL ANALYSIS FUNCTIONS
//=============================================================================

/**
 * Calculate how closely illuminance matches the setpoint over time
 *
 * @return Mean absolute error between setpoint and measured illuminance
 */
float computeSetpointTrackingError();

/**
 * Calculate illuminance dip ratio
 * Measures the frequency and magnitude of illuminance drops below the setpoint
 *
 * @return Ratio of samples where illuminance is below setpoint
 */
float computeIlluminanceDipRatio();

#endif // PICONTROLLER_H