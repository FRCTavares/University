#ifndef LEDDRIVER_H
#define LEDDRIVER_H

#include <Arduino.h>

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

/**
 * Initialize LED driver with the specified GPIO pin
 * Configures PWM parameters and sets initial state to off
 * 
 * @param pin GPIO pin number connected to the LED
 */
void initLEDDriver(int pin);

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

#endif // LEDDRIVER_H