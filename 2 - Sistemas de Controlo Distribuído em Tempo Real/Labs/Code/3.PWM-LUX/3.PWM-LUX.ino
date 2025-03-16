#include <Arduino.h>
#include <math.h>

// Define constants for voltage, ADC, and resistor values
#define VCC 3.3                    // Supply voltage in volts
#define MY_ADC_RESOLUTION 4095.0   // 12-bit ADC maximum value (0 to 4095)
#define FIXED_RESISTOR 10000.0     // 10k ohm resistor in the voltage divider
const float R10 = 225000.0;          // Nominal LDR resistance at a reference lux

// Calibration parameter for the LDR conversion equation
// The relationship is: log10(R_LDR) = m * log10(Lux) + b
// We are calibrating m (the slope); b is then computed from R10.
float LDR_M = -1;    // Initial guess for m; adjust this for calibration
float LDR_B = log10(R10) - LDR_M; // Derived intercept from nominal value

// PWM and LED configuration
const int LED_PIN = 15;    // LED pin driven by PWM
const int DAC_RANGE = 4096; // PWM range (0 to 4095)
const int NUM_STEPS = 11;  // We want 11 steps (0, 1, 2, ..., 10)

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogWriteResolution(12);
}

void loop() {
  int pwmValues[NUM_STEPS];
  float luxValues[NUM_STEPS];
  
  // Loop over 11 steps to sweep the PWM duty cycle from 0 to maximum
  for (int step = 0; step < NUM_STEPS; step++) {
    // Calculate the PWM value for this step
    int pwmValue = (DAC_RANGE - 1) * step / (NUM_STEPS - 1);
    pwmValues[step] = pwmValue;
    
    // Write PWM value to the LED
    analogWrite(LED_PIN, pwmValue);
    delay(500);  // Allow time for the LED and RC filter to settle
    
    // Read the ADC value from the voltage divider (LDR and fixed resistor)
    int adcValue = analogRead(A0);
    
    // Convert ADC reading to voltage
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
    if (voltage <= 0) {
      Serial.println("Error: Voltage reading is 0.");
      luxValues[step] = 0;
      continue;
    }
    
    // Calculate the LDR resistance using the voltage divider formula:
    // V_out = VCC * (R_fixed / (R_fixed + R_LDR))
    // => R_LDR = R_fixed * (VCC/V_out - 1)
    float rLDR = FIXED_RESISTOR * (VCC / voltage - 1);
    
    // Compute lux using the log-log conversion formula:
    // log10(Lux) = (log10(R_LDR) - b) / m  ==>  Lux = 10^((log10(R_LDR) - b) / m)
    float lux = pow(10, (log10(rLDR) - LDR_B) / LDR_M);
    luxValues[step] = lux;
  }
  
  // Print all PWM (duty cycle) values first, one per line
  Serial.println("PWM Duty Cycle Values:");
  for (int i = 0; i < NUM_STEPS; i++) {
    Serial.println(pwmValues[i]);
  }
  
  // Then print all corresponding Lux values, one per line
  Serial.println("Lux Values:");
  for (int i = 0; i < NUM_STEPS; i++) {
    Serial.println(luxValues[i], 2);
  }
  
  Serial.println("Calibration cycle complete. Restarting in 5 seconds...");
  delay(5000);
}
