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
    void getTerms(float& p, float& i) const;

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
    
    float Kp;    // Proportional gain
    float Ki;    // Integral gain
    float Beta;  // Setpoint weighting factor
    float h;     // Sampling time in seconds
    
    //=========================================================================
    // CONTROLLER STATE
    //=========================================================================
    
    float Iterm; // Accumulated integral term
    float e_old; // Previous error for integration
    float Pterm; // Last proportional term (for diagnostics)

    //=========================================================================
    // COORDINATION VARIABLES
    //=========================================================================
    
    float internalTarget; // Target value used for coordination
    bool useInternalTarget; // Flag to use internal or external target

    // Add these new variables
    float Kff;          // Feedforward gain
    float prevSetpoint; // Previous setpoint for detecting changes
    bool useFeedforward; // Enable/disable feedforward
};

//=======================================================================================================================================================================================================================================
// Metric Calculation Functions
//=======================================================================================================================================================================================================================================


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


//=======================================================================================================================================================================================================================================
// CIRCULAR BUFFER DATA STORAGE
//=======================================================================================================================================================================================================================================
//=============================================================================
// CONFIGURATION
//=============================================================================


//=============================================================================
// DATA STRUCTURES
//=============================================================================

/**
 * Data structure for a single time-series log entry
 * Each entry stores a timestamp and associated sensor/control values
 */
struct LogEntry {
  unsigned long timestamp;  // Time when entry was recorded (ms)
  float lux;               // Measured illuminance value (lux)
  float duty;              // LED duty cycle (0.0-1.0)
  float setpoint;          // Reference illuminance target
  float flicker;           // Individual flicker value at this point
  float jitter; // Jitter in µs (difference from nominal period)
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
LogEntry* getLogBuffer();

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

//=======================================================================================================================================================================================================================================
// LED Driver Functions
//=======================================================================================================================================================================================================================================

/**
 * LED Driver Module
 * 
 * Provides a comprehensive interface for controlling LED brightness with:
 * - Properly configured PWM frequency (30kHz) for flicker-free operation
 * - High resolution (12-bit) brightness control
 * - Multiple control interfaces (duty cycle, percentage, PWM value, power)
 * - Smooth transitions and special lighting effects
 * - Energy-efficient operation with accurate power modeling
 * 
 * This module abstracts the hardware-specific details of LED control and
 * provides a consistent API for the rest of the system to use.
 */

//=============================================================================
// INITIALIZATION FUNCTIONS
//=============================================================================


//=============================================================================
// BASIC CONTROL FUNCTIONS
//=============================================================================

/**
 * Set LED brightness using PWM duty cycle
 * This is the primary control function that other methods call
 * 
 * @param dutyCycle Duty cycle value between 0.0 (off) and 1.0 (fully on)
 */
void setLEDDutyCycle(float dutyCycle);

/**
 * Set LED brightness using percentage
 * Converts percentage to duty cycle and calls setLEDDutyCycle
 * 
 * @param percentage Brightness percentage between 0.0 (off) and 100.0 (fully on)
 */
void setLEDPercentage(float percentage);

/**
 * Set LED brightness using direct PWM value
 * Bypasses duty cycle calculation for direct hardware control
 * 
 * @param pwmValue PWM value between 0 (off) and PWM_MAX (fully on)
 */
void setLEDPWMValue(int pwmValue);

/**
 * Set LED brightness based on desired power consumption
 * Maps power in watts to appropriate duty cycle
 * 
 * @param powerWatts Desired power in watts from 0.0 to MAX_POWER_WATTS
 */
void setLEDPower(float powerWatts);

//=============================================================================
// STATUS QUERY FUNCTIONS
//=============================================================================

/**
 * Get current LED duty cycle setting
 * 
 * @return Current duty cycle value (0.0 to 1.0)
 */
float getLEDDutyCycle();

/**
 * Get current LED brightness as percentage
 * 
 * @return Current brightness percentage (0.0 to 100.0)
 */
float getLEDPercentage();

/**
 * Get current LED PWM value
 * 
 * @return Current PWM value (0 to PWM_MAX)
 */
int getLEDPWMValue();

/**
 * Get estimated current LED power consumption
 * 
 * @return Estimated power consumption in watts
 */
float getLEDPower();

//=============================================================================
// ADVANCED CONTROL FUNCTIONS
//=============================================================================

/**
 * Smoothly transition LED from current to target brightness
 * Implements a gradual change to avoid abrupt lighting changes
 * 
 * @param targetDutyCycle Target duty cycle to transition to (0.0 to 1.0)
 * @param transitionTimeMs Duration of transition in milliseconds
 */
void smoothTransition(float targetDutyCycle, int transitionTimeMs);

/**
 * Create a pulsing effect by varying LED brightness
 * Implements a sinusoidal brightness variation
 * 
 * @param durationMs Total duration of the pulse effect in milliseconds
 * @param minDuty Minimum duty cycle during pulse (0.0 to 1.0)
 * @param maxDuty Maximum duty cycle during pulse (0.0 to 1.0)
 */
void pulseEffect(int durationMs, float minDuty, float maxDuty);
#endif // PICONTROLLER_H