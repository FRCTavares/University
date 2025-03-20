#pragma once

// Initialize LED driver
void initLEDDriver(int pin);

// Control functions
void setLEDDutyCycle(float dutyCycle);
void setLEDPercentage(float percentage);
void setLEDPWMValue(int pwmValue);
void setLEDPower(float powerWatts);

// Query functions
float getLEDDutyCycle();
float getLEDPercentage();
int getLEDPWMValue();
float getLEDPower();

// Test LED functionality
void testLED();