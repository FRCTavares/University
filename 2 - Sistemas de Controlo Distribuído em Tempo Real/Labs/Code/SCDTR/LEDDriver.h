#ifndef LEDDRIVER_H
#define LEDDRIVER_H

#include <Arduino.h>

/**
 * Initialize the LED driver with the specified GPIO pin
 * Configures PWM parameters and sets initial state to off
 *
 * @param pin GPIO pin number connected to the LED
 */
void initLEDDriver(int pin);

/**
 * Set LED brightness using PWM duty cycle (0.0 to 1.0)
 * This is the primary control function that other methods call
 *
 * @param newDutyCycle Duty cycle value between 0.0 (off) and 1.0 (fully on)
 */
void setLEDDutyCycle(float newDutyCycle);

/**
 * Set LED brightness using percentage (0% to 100%)
 * Converts percentage to duty cycle and calls setLEDDutyCycle
 *
 * @param percentage Brightness percentage between 0.0 (off) and 100.0 (fully on)
 */
void setLEDPercentage(float percentage);

/**
 * Set LED brightness using direct PWM value (0 to pwmMax)
 * Bypasses duty cycle calculation for direct hardware control
 *
 * @param pwmValue PWM value between 0 (off) and pwmMax (fully on)
 */
void setLEDPWMValue(int pwmValue);

/**
 * Set LED brightness based on desired power consumption
 * Maps power in watts to appropriate duty cycle
 *
 * @param powerWatts Desired power in watts from 0.0 to MAX_POWER_WATTS
 */
void setLEDPower(float powerWatts);

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
 * @return Current PWM value (0 to pwmMax)
 */
int getLEDPWMValue();

/**
 * Get estimated current LED power consumption
 *
 * @return Estimated power consumption in watts
 */
float getLEDPower();

#endif // LEDDRIVER_H