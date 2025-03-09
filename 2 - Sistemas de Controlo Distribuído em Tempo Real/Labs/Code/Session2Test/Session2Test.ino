#include <Arduino.h>
#include <math.h>

// Hardware configuration
#define LED_PIN 15           // GPIO pin connected to LED
#define LDR_PIN 10           // GPIO pin for LDR voltage divider (ADC)
#define FIXED_RESISTOR 10000 // Value of fixed resistor in the voltage divider (10kΩ)
#define PWM_FREQ 60000       // PWM frequency: 60 kHz
#define PWM_MAX 4095         // 12-bit PWM resolution (0-4095)

// LDR characteristics - using improved parameters
// Formula: log10(LDR) = m * log10(LUX) + b
#define LDR_M -1.2           // Slope parameter from new code
const float R10 = 225000.0;  // Reference resistance at 10 lux
const float LDR_B = log10(R10) - LDR_M * log10(10.0); // Calculate B from reference point

// Measurement configuration
#define NUM_MEASUREMENTS 11   // Number of measurements to take
#define STABILIZE_MS 3000     // Increased time to wait for light level to stabilize (ms)
#define NUM_SAMPLES 30        // Number of ADC samples to average per measurement

// Convert ADC reading to LDR resistance
float adc_to_resistance(int adc_value) {
    // Convert 12-bit ADC reading to voltage (0-3.3V)
    float voltage = (adc_value * 3.3f) / 4095.0f;
    
    // Use voltage divider formula to calculate resistance
    // R_LDR = R_fixed * (Vcc/V_out - 1)
    if (voltage < 0.01f) return 1000000.0f; // Avoid division by very small values
    return FIXED_RESISTOR * (3.3f / voltage - 1.0f);
}

// Convert LDR resistance to LUX using log-log relationship
float resistance_to_lux(float resistance) {
    // Guard against negative or zero values
    if (resistance <= 0) return 0.0f;
    
    // Solve for LUX: log10(LUX) = (log10(LDR) - b) / m
    float lux = pow(10.0f, (log10(resistance) - LDR_B) / LDR_M);
    return lux;
}

void setup() {
    // Initialize serial communication
    Serial.begin(115200);
    delay(2000); // Wait for serial connection
    
    Serial.println("LED PWM and LUX Measurement System");
    
    // Initialize pins
    pinMode(LED_PIN, OUTPUT);
    analogReadResolution(12);  // Set ADC resolution to 12-bit
    analogWriteResolution(12); // Set PWM resolution to 12-bit
    analogWriteFreq(PWM_FREQ); // Set PWM frequency to 60kHz
    
    // Table header
    Serial.println("\nDuty Cycle (0-1),PWM Level,LDR Resistance (Ohms),Illuminance (LUX)");
}

void loop() {
    // Take measurements at different PWM duty cycles
    for (int i = 0; i < NUM_MEASUREMENTS; i++) {
        // Calculate PWM level (evenly spaced from 0 to MAX)
        int pwm_level = i * (PWM_MAX / (NUM_MEASUREMENTS - 1));
        float duty_cycle = (float)pwm_level / PWM_MAX;  // 0.0-1.0 range
        
        // Set LED PWM duty cycle
        analogWrite(LED_PIN, pwm_level);
        
        // Wait for the light level to stabilize
        delay(STABILIZE_MS);
        
        // Take multiple readings and average them to reduce noise
        uint32_t adc_sum = 0;
        for (int j = 0; j < NUM_SAMPLES; j++) {
            adc_sum += analogRead(LDR_PIN);
            delay(10);
        }
        int adc_value = adc_sum / NUM_SAMPLES;
        
        // Calculate LDR resistance and convert to LUX
        float resistance = adc_to_resistance(adc_value);
        float lux = resistance_to_lux(resistance);
        
        // Print results
        Serial.print(duty_cycle, 4);  // Print with 4 decimal places
        Serial.print(",");
        Serial.print(pwm_level);
        Serial.print(",");
        Serial.print(resistance, 2);
        Serial.print(",");
        Serial.println(lux, 2);
    }
    
    // Turn off LED when measurements are complete
    analogWrite(LED_PIN, 0);
    Serial.println("\nMeasurement complete");
    
    // Wait before repeating the measurements
    delay(10000);
}