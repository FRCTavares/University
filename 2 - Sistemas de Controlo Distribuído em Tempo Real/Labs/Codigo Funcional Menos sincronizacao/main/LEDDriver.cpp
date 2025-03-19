#include "LEDDriver.h"
#include "Globals.h"  

// Static variables for the module
static int ledPin = -1;
static int pwmMax = 4095;
static int pwmMin = 0;

// --- PWM Configuration Constants ---
const unsigned int PWM_FREQUENCY = 30000;


void initLEDDriver(int pin) {
    ledPin = pin;
    pinMode(ledPin, OUTPUT);
    
    // Configure PWM with optimal settings
    analogWriteRange(pwmMax);
    analogWriteFreq(PWM_FREQUENCY);
    
    // Start with LED off
    analogWrite(ledPin, pwmMin);
    dutyCycle = 0.0; // Use the global duty cycle variable
    
    // Debug message only if debug enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("LED driver initialized on pin ");
        Serial.println(pin);
    }
}

void setLEDDutyCycle(float newDutyCycle) {
    // Validate and constrain input
    if (isnan(newDutyCycle) || isinf(newDutyCycle)) {
        return; // Protect against invalid inputs
    }
    
    // Constrain to valid range
    newDutyCycle = constrain(newDutyCycle, 0.0f, 1.0f);
    
    // Apply duty cycle
    int pwmValue = (int)(newDutyCycle * pwmMax);
    analogWrite(ledPin, pwmValue);
    
    // Update the global duty cycle
    dutyCycle = newDutyCycle;
    
    // Debug message only if debug enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("LED duty cycle set to: ");
        Serial.println(newDutyCycle, 3);
    }
}

void setLEDPercentage(float percentage) {
    percentage = constrain(percentage, 0.0f, 100.0f);
    float newDutyCycle = percentage / 100.0f;
    setLEDDutyCycle(newDutyCycle);
    
    // Debug message only if debug enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("LED percentage set to: ");
        Serial.println(percentage, 1);
    }
}

void setLEDPWMValue(int pwmValue) {
    pwmValue = constrain(pwmValue, pwmMin, pwmMax);
    analogWrite(ledPin, pwmValue);
    
    // Update global duty cycle
    dutyCycle = (float)pwmValue / pwmMax;
    
    // Debug message only if debug enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("LED PWM value set to: ");
        Serial.println(pwmValue);
    }
}

void setLEDPower(float powerWatts) {
    powerWatts = constrain(powerWatts, 0.0f, MAX_POWER_WATTS);
    float newDutyCycle = powerWatts / MAX_POWER_WATTS;
    setLEDDutyCycle(newDutyCycle);
    
    // Debug message only if debug enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("LED power set to: ");
        Serial.println(powerWatts, 3);
    }
}

float getLEDDutyCycle() {
    return dutyCycle;
}

float getLEDPercentage() {
    return dutyCycle * 100.0f;
}

int getLEDPWMValue() {
    return (int)(dutyCycle * pwmMax);
}

float getLEDPower() {
    return dutyCycle * MAX_POWER_WATTS;
}