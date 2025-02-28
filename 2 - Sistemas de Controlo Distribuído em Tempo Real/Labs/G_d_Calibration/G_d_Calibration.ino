#include <Arduino.h>
#include <math.h>

// Define constants for sensor conversion
#define VCC 3.3                    // Supply voltage in volts
#define MY_ADC_RESOLUTION 4095.0   // 12-bit ADC max value (0 to 4095)
#define FIXED_RESISTOR 10000.0     // 10k ohm resistor in the voltage divider

// Calibrated LDR parameters from datasheet (for PGM5659D)
// Model: log10(R_LDR) = m * log10(Lux) + b, where m = -γ and b = log10(A)
// Example: at 10 lux, R ≈ 225 kΩ and γ ≈ 0.8 gives:
#define LDR_M -0.8               
#define LDR_B 6.15

// LED and sensor pins (adjust according to your wiring)
#define LED_PIN 15    // PWM pin driving the LED
#define SENSOR_PIN A0 // Analog pin reading the LDR voltage

// Define maximum PWM value (8-bit resolution)
#define MAX_PWM 255

// Number of calibration points
const int N = 11;  // Duty cycle values from 0 to MAX_PWM

// Arrays to store duty cycle (u) and measured lux (L) values
float dutyCycles[N];
float luxValues[N];

void setup() {
  Serial.begin(115200);
  
  // Set analog resolution to 12 bits for the ADC if supported
  #if defined(analogReadResolution)
  analogReadResolution(12);
  #endif

  Serial.println("Calibration of Box Gain Started");
  Serial.println("DutyCycle, Lux"); // CSV header for plotting

  // Allow the system to settle
  delay(2000);

  // Sweep through duty cycles from 0 to MAX_PWM
  for (int i = 0; i < N; i++) {
    // Calculate the duty cycle value for this step
    float u = (MAX_PWM * i) / (N - 1);  // u in [0, MAX_PWM]
    dutyCycles[i] = u;
    
    // Set the LED PWM to the duty cycle value using analogWrite()
    analogWrite(LED_PIN, (int)u);
    // Wait for the LED and sensor reading to settle
    delay(1000);
    
    // Read the sensor value from the LDR circuit
    int adcValue = analogRead(SENSOR_PIN);
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
    if(voltage <= 0) voltage = 0.0001; // Prevent division by zero
    
    // Compute LDR resistance using the voltage divider formula:
    // V_out = VCC * (R_fixed / (R_fixed + R_LDR))
    // => R_LDR = R_fixed * (VCC/V_out - 1)
    float rLDR = FIXED_RESISTOR * (VCC / voltage - 1);
    
    // Convert LDR resistance to lux using the relationship:
    // log10(R_LDR) = m * log10(Lux) + b  ->  log10(Lux) = (log10(rLDR) - b) / m
    float lux = pow(10, (log10(rLDR) - LDR_B) / LDR_M);
    luxValues[i] = lux;
    
    // Print the duty cycle and lux value in CSV format for plotting
    Serial.print(u, 2);
    Serial.print(",");
    Serial.println(lux, 2);
  }
  
  // Now perform linear regression to fit L = d + G * u
  float sumU = 0, sumL = 0, sumUU = 0, sumUL = 0;
  for (int i = 0; i < N; i++) {
    sumU += dutyCycles[i];
    sumL += luxValues[i];
    sumUU += dutyCycles[i] * dutyCycles[i];
    sumUL += dutyCycles[i] * luxValues[i];
  }
  float G_cal = (N * sumUL - sumU * sumL) / (N * sumUU - sumU * sumU);
  float d_cal = (sumL - G_cal * sumU) / N;
  
  Serial.println("----- Regression Result -----");
  Serial.print("Calibrated Gain (G): ");
  Serial.println(G_cal, 4);
  Serial.print("Offset (d): ");
  Serial.println(d_cal, 4);
  
  // Optionally, you can leave the loop() empty or implement further functionality.
}

void loop() {
  // Calibration is done in setup.
}
