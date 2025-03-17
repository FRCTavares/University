#ifndef LEDDRIVER_H
#define LEDDRIVER_H

#include <Arduino.h>

/**
 * LED Driver Module
 * Provides multiple interfaces for controlling LED brightness with:
 * - Properly configured PWM frequency and resolution
 * - Multiple unit formats (duty cycle, percentage, PWM value, power)
 * - Power efficiency and flicker-free operation
 */

// Initialize LED driver with appropriate pin and settings
void initLEDDriver(int pin);

// Set LED brightness using duty cycle [0.0-1.0]
void setLEDDutyCycle(float dutyCycle);

// Set LED brightness using percentage [0-100]
void setLEDPercentage(float percentage);

// Set LED brightness using direct PWM value [0-PWM_MAX]
void setLEDPWMValue(int pwmValue);

// Set LED brightness using power in watts [0-MAX_POWER_WATTS]
void setLEDPower(float powerWatts);

// Get current LED settings in different units
float getLEDDutyCycle();
float getLEDPercentage();
int getLEDPWMValue();
float getLEDPower();

// Advanced control functions
void smoothTransition(float targetDutyCycle, int transitionTimeMs);
void pulseEffect(int durationMs, float minDuty, float maxDuty);

#endif