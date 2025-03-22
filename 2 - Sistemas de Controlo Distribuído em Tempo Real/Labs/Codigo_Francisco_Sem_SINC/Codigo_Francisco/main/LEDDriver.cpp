#include "LEDDriver.h"
#include "Globals.h"  

//-----------------------------------------------------------------------------
// CONFIGURATION AND VARIABLES
//-----------------------------------------------------------------------------

// Static module variables
static int ledPin = -1;       // GPIO pin connected to the LED
static int pwmMax = 4095;     // Maximum PWM value (12-bit resolution)
static int pwmMin = 0;        // Minimum PWM value (off)

// PWM configuration constants
const unsigned int PWM_FREQUENCY = 30000; // 30 kHz frequency

//-----------------------------------------------------------------------------
// INITIALIZATION FUNCTIONS
//-----------------------------------------------------------------------------

/**
 * Initialize the LED driver with the specified GPIO pin
 * Configures PWM parameters and sets initial state to off
 * 
 * @param pin GPIO pin number connected to the LED
 */
void initLEDDriver(int pin) {
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
    dutyCycle = 0.0; // Use the global duty cycle variable
    
    // Debug message only if debug enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("LED driver initialized on pin ");
        Serial.println(pin);
    }
}

//-----------------------------------------------------------------------------
// LED CONTROL FUNCTIONS
//-----------------------------------------------------------------------------

/**
 * Set LED brightness using PWM duty cycle (0.0 to 1.0)
 * This is the primary control function that other methods call
 * 
 * @param newDutyCycle Duty cycle value between 0.0 (off) and 1.0 (fully on)
 */
void setLEDDutyCycle(float newDutyCycle) {
    // Validate and constrain input
    if (isnan(newDutyCycle) || isinf(newDutyCycle)) {
        return; // Protect against invalid inputs
    }
    
    // Constrain to valid range
    newDutyCycle = constrain(newDutyCycle, 0.0f, 1.0f);
    
    // Apply duty cycle by converting to appropriate PWM value
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

/**
 * Set LED brightness using percentage (0% to 100%)
 * Converts percentage to duty cycle and calls setLEDDutyCycle
 * 
 * @param percentage Brightness percentage between 0.0 (off) and 100.0 (fully on)
 */
void setLEDPercentage(float percentage) {
    // Constrain to valid percentage range
    percentage = constrain(percentage, 0.0f, 100.0f);
    
    // Convert percentage to duty cycle
    float newDutyCycle = percentage / 100.0f;
    
    // Set the LED using the calculated duty cycle
    setLEDDutyCycle(newDutyCycle);
    
    // Debug message only if debug enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("LED percentage set to: ");
        Serial.println(percentage, 1);
    }
}

/**
 * Set LED brightness using direct PWM value (0 to pwmMax)
 * Bypasses duty cycle calculation for direct hardware control
 * 
 * @param pwmValue PWM value between 0 (off) and pwmMax (fully on)
 */
void setLEDPWMValue(int pwmValue) {
    // Constrain to valid PWM range
    pwmValue = constrain(pwmValue, pwmMin, pwmMax);
    
    // Apply PWM value directly
    analogWrite(ledPin, pwmValue);
    
    // Update global duty cycle to maintain state consistency
    dutyCycle = (float)pwmValue / pwmMax;
    
    // Debug message only if debug enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("LED PWM value set to: ");
        Serial.println(pwmValue);
    }
}

/**
 * Set LED brightness based on desired power consumption
 * Maps power in watts to appropriate duty cycle
 * 
 * @param powerWatts Desired power in watts from 0.0 to MAX_POWER_WATTS
 */
void setLEDPower(float powerWatts) {
    // Constrain to valid power range
    powerWatts = constrain(powerWatts, 0.0f, MAX_POWER_WATTS);
    
    // Convert power to duty cycle (assumes linear relationship)
    float newDutyCycle = powerWatts / MAX_POWER_WATTS;
    
    // Set the LED using the calculated duty cycle
    setLEDDutyCycle(newDutyCycle);
    
    // Debug message only if debug enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("LED power set to: ");
        Serial.println(powerWatts, 3);
    }
}

//-----------------------------------------------------------------------------
// LED STATUS QUERY FUNCTIONS
//-----------------------------------------------------------------------------

/**
 * Get current LED duty cycle setting
 * 
 * @return Current duty cycle value (0.0 to 1.0)
 */
float getLEDDutyCycle() {
    return dutyCycle;
}

/**
 * Get current LED brightness as percentage
 * 
 * @return Current brightness percentage (0.0 to 100.0)
 */
float getLEDPercentage() {
    return dutyCycle * 100.0f;
}

/**
 * Get current LED PWM value
 * 
 * @return Current PWM value (0 to pwmMax)
 */
int getLEDPWMValue() {
    return (int)(dutyCycle * pwmMax);
}

/**
 * Get estimated current LED power consumption
 * 
 * @return Estimated power consumption in watts
 */
float getLEDPower() {
    return dutyCycle * MAX_POWER_WATTS;
}

//-----------------------------------------------------------------------------
// ADVANCED CONTROL FUNCTIONS
//-----------------------------------------------------------------------------

/**
 * Smoothly transition LED from current to target brightness
 * Implements a gradual change to avoid abrupt lighting changes
 * 
 * @param targetDutyCycle Target duty cycle to transition to (0.0 to 1.0)
 * @param transitionTimeMs Duration of transition in milliseconds
 */
void smoothTransition(float targetDutyCycle, int transitionTimeMs) {
    // Validate input parameters
    targetDutyCycle = constrain(targetDutyCycle, 0.0f, 1.0f);
    transitionTimeMs = max(transitionTimeMs, 10); // Minimum 10ms transition
    
    // Get current duty cycle as starting point
    float startDutyCycle = getLEDDutyCycle();
    
    // No transition needed if already at target
    if (fabs(targetDutyCycle - startDutyCycle) < 0.001) {
        return;
    }
    
    // Calculate total change in duty cycle
    float deltaDuty = targetDutyCycle - startDutyCycle;
    
    // Set up timing
    unsigned long startTime = millis();
    unsigned long currentTime;
    float progress;
    
    // Debug message if enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("Starting transition from ");
        Serial.print(startDutyCycle, 3);
        Serial.print(" to ");
        Serial.print(targetDutyCycle, 3);
        Serial.print(" over ");
        Serial.print(transitionTimeMs);
        Serial.println("ms");
    }
    
    // Transition loop
    do {
        // Calculate current progress (0.0 to 1.0)
        currentTime = millis();
        progress = constrain((float)(currentTime - startTime) / transitionTimeMs, 0.0f, 1.0f);
        
        // Apply easing function for more natural transitions
        // Using cubic easing: progress = progress^3
        float easedProgress = progress * progress * progress;
        
        // Calculate and set current duty cycle using linear interpolation
        float currentDuty = startDutyCycle + (deltaDuty * easedProgress);
        setLEDDutyCycle(currentDuty);
        
        // Small delay to prevent overwhelming the CPU
        delay(5);
        
    } while (progress < 1.0);
    
    // Ensure we end exactly at the target value
    setLEDDutyCycle(targetDutyCycle);
    
    // Debug message if enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.println("Transition complete");
    }
}

/**
 * Create a pulsing effect by varying LED brightness
 * Implements a sinusoidal brightness variation
 * 
 * @param durationMs Total duration of the pulse effect in milliseconds
 * @param minDuty Minimum duty cycle during pulse (0.0 to 1.0)
 * @param maxDuty Maximum duty cycle during pulse (0.0 to 1.0)
 */
void pulseEffect(int durationMs, float minDuty, float maxDuty) {
    // Validate input parameters
    minDuty = constrain(minDuty, 0.0f, 1.0f);
    maxDuty = constrain(maxDuty, 0.0f, 1.0f);
    durationMs = max(durationMs, 100); // Minimum 100ms duration
    
    // Ensure min is less than max
    if (minDuty > maxDuty) {
        float temp = minDuty;
        minDuty = maxDuty;
        maxDuty = temp;
    }
    
    // Calculate the amplitude and midpoint of the duty cycle variation
    float amplitude = (maxDuty - minDuty) / 2.0;
    float midpoint = minDuty + amplitude;
    
    // Debug message if enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("Starting pulse effect: min=");
        Serial.print(minDuty, 3);
        Serial.print(", max=");
        Serial.print(maxDuty, 3);
        Serial.print(", duration=");
        Serial.print(durationMs);
        Serial.println("ms");
    }
    
    // Set up timing
    unsigned long startTime = millis();
    unsigned long currentTime;
    float progress;
    
    // Target 50 updates per second for smooth animation
    const int updateIntervalMs = 20;
    unsigned long lastUpdateTime = 0;
    
    // Pulse loop
    do {
        currentTime = millis();
        
        // Only update at the specified interval
        if (currentTime - lastUpdateTime >= updateIntervalMs) {
            lastUpdateTime = currentTime;
            
            // Calculate progress through the effect (0.0 to 1.0)
            progress = constrain((float)(currentTime - startTime) / durationMs, 0.0f, 1.0f);
            
            // Calculate current duty cycle using sine function
            // sin() expects radians, so we convert progress to 0-2π range
            // We multiply by 2 to get a complete sine cycle
            float dutyCycle = midpoint + amplitude * sin(progress * 2 * PI);
            
            // Set the LED brightness
            setLEDDutyCycle(dutyCycle);
        }
        
        // Small delay to prevent overwhelming the CPU
        delay(1);
        
    } while (progress < 1.0);
    
    // Leave LED at midpoint brightness after effect completes
    setLEDDutyCycle(midpoint);
    
    // Debug message if enabled
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.println("Pulse effect complete");
    }
}