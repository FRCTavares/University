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
//-----------------------------------------------------------------------------
// CONFIGURATION AND VARIABLES
//-----------------------------------------------------------------------------

// Static module variables
static int ledPin = -1;
static int pwmMax = 4095; // 12-bit PWM resolution
static int pwmMin = 0;

// PWM configuration constants
const unsigned int PWM_FREQUENCY = 30000;
const float MAX_POWER_WATTS = 0.08755; // Maximum power consumption at 100% duty

/**
 * Initialize the LED driver with the specified GPIO pin
 * Configures PWM parameters and sets initial state to off
 *
 * @param pin GPIO pin number connected to the LED
 */
void initLEDDriver(int pin)
{
  // Store pin and configure it as output
  ledPin = pin;
  pinMode(ledPin, OUTPUT);

  // Configure PWM with optimal settings
  // - Sets resolution to 12-bit (0-4095)
  // - Sets frequency to 30kHz for flicker-free operation
  analogWriteRange(pwmMax);
  analogWriteFreq(PWM_FREQUENCY);

  // Start with LED off
  analogWrite(ledPin, pwmMin);
  critical_section_enter_blocking(&commStateLock);
  controlState.dutyCycle = 0.0; // Use the global duty cycle variable
  critical_section_exit(&commStateLock);
}

/**
 * Set LED brightness using PWM duty cycle (0.0 to 1.0)
 * This is the primary control function that other methods call
 *
 * @param newDutyCycle Duty cycle value between 0.0 (off) and 1.0 (fully on)
 */
void setLEDDutyCycle(float newDutyCycle)
{
  // Validate and constrain input
  if (isnan(newDutyCycle) || isinf(newDutyCycle))
  {
    return; // Protect against invalid inputs
  }

  // Constrain to valid range
  newDutyCycle = constrain(newDutyCycle, 0.0f, 1.0f);

  // Apply duty cycle by converting to appropriate PWM value
  int pwmValue = (int)(newDutyCycle * pwmMax);
  analogWrite(ledPin, pwmValue);

  // Update the global duty cycle
  critical_section_enter_blocking(&commStateLock);
  controlState.dutyCycle = newDutyCycle;
  critical_section_exit(&commStateLock);
}

/**
 * Set LED brightness using percentage (0% to 100%)
 * Converts percentage to duty cycle and calls setLEDDutyCycle
 *
 * @param percentage Brightness percentage between 0.0 (off) and 100.0 (fully on)
 */
void setLEDPercentage(float percentage)
{
  // Constrain to valid percentage range
  percentage = constrain(percentage, 0.0f, 100.0f);

  // Convert percentage to duty cycle
  float newDutyCycle = percentage / 100.0f;

  // Set the LED using the calculated duty cycle
  setLEDDutyCycle(newDutyCycle);
}

/**
 * Set LED brightness using direct PWM value (0 to pwmMax)
 * Bypasses duty cycle calculation for direct hardware control
 *
 * @param pwmValue PWM value between 0 (off) and pwmMax (fully on)
 */
void setLEDPWMValue(int pwmValue)
{
  // Constrain to valid PWM range
  pwmValue = constrain(pwmValue, pwmMin, pwmMax);

  // Apply PWM value directly
  analogWrite(ledPin, pwmValue);

  // Update global duty cycle to maintain state consistency
  critical_section_enter_blocking(&commStateLock);
  controlState.dutyCycle = (float)pwmValue / pwmMax;
  critical_section_exit(&commStateLock);
}

/**
 * Set LED brightness based on desired power consumption
 * Maps power in watts to appropriate duty cycle
 *
 * @param powerWatts Desired power in watts from 0.0 to MAX_POWER_WATTS
 */
void setLEDPower(float powerWatts)
{
  // Constrain to valid power range
  powerWatts = constrain(powerWatts, 0.0f, MAX_POWER_WATTS);

  // Convert power to duty cycle (assumes linear relationship)
  float newDutyCycle = powerWatts / MAX_POWER_WATTS;

  // Set the LED using the calculated duty cycle
  setLEDDutyCycle(newDutyCycle);
}

/**
 * Get current LED duty cycle setting
 *
 * @return Current duty cycle value (0.0 to 1.0)
 */
float getLEDDutyCycle()
{
  return controlState.dutyCycle;
}
