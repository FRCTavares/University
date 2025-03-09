#include <Arduino.h>
#include <math.h>

// Circuit Constants
#define VCC 3.3                    // Supply voltage (volts)
#define MY_ADC_RESOLUTION 4095.0   // 12-bit ADC max value (0 to 4095)
#define FIXED_RESISTOR 10000.0     // 10kΩ resistor in voltage divider

// Reference Resistance (LDR at 10 lux)
const float R10 = 225000.0;

// LDR Equation Parameters (log-log relationship)
const float LDR_M = -1.2;
float LDR_B = log10(R10) - LDR_M;

// Calibration Constants
const int NUM_PONTOS = 11;          // Number of calibration points
const int NUM_READINGS = 20;        // Increased number of ADC readings to improve accuracy
const int STABILIZATION_TIME = 3000; // Time (ms) to allow LED brightness to stabilize

// Arrays to store measured values
float dutyCycles[NUM_PONTOS];
float luxValues[NUM_PONTOS];

// Pin Definitions
#define LED_PIN 15    // PWM pin for LED control
#define SENSOR_PIN A0 // Analog pin for LDR reading

void setup() {
  Serial.begin(115200);
  
  // Configure 12-bit ADC resolution (if supported)
  #if defined(analogReadResolution)
  analogReadResolution(12);
  #endif

  // Set a high PWM frequency (e.g., 60kHz to avoid flickering)
  analogWriteFreq(60000);      // Set PWM frequency to 60kHz
  analogWriteRange(4096);      // Use full 12-bit range

  Serial.println("Duty Cycle vs. Lux Measurements:");
  delay(2000); // Allow system stabilization

  // Calibration loop: Sweep through different duty cycles and measure Lux
  for (int i = 0; i < NUM_PONTOS; i++) {
    // Compute duty cycle (linearly spaced between 1 and 4096)
    float u = 1 + (4095.0 * i) / (NUM_PONTOS - 1);
    dutyCycles[i] = u;

    // Set PWM output
    analogWrite(LED_PIN, (int)u);

    // Ensure steady-state by waiting before measuring
    delay(STABILIZATION_TIME); 

    // Take multiple ADC readings and average them
    float totalVoltage = 0;
    for (int j = 0; j < NUM_READINGS; j++) {
      int adcValue = analogRead(SENSOR_PIN);
      float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
      totalVoltage += voltage;
      delay(20);  // Short delay between readings to avoid immediate noise
    }
    float avgVoltage = totalVoltage / NUM_READINGS;

    // Avoid division by zero
    if (avgVoltage <= 0) avgVoltage = 0.0001;

    // Compute LDR resistance
    float rLDR = FIXED_RESISTOR * (VCC / avgVoltage - 1);

    // Compute Lux using log-log equation
    float lux = pow(10, (log10(rLDR) - LDR_B) / LDR_M);

    // Store Lux value
    luxValues[i] = lux;
  }

  // Print Duty Cycle values
  Serial.println("Duty Cycle:");
  for (int i = 0; i < NUM_PONTOS; i++) {
    Serial.print(dutyCycles[i], 2);
    Serial.println();
  }
  Serial.println(); // New line

  // Print Lux values
  Serial.println("Lux:");
  for (int i = 0; i < NUM_PONTOS; i++) {
    Serial.print(luxValues[i], 2);
    Serial.println();
  }
  Serial.println(); // New line
}

void loop() {
  // Calibration runs only once in setup()
}