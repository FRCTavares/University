#include <Arduino.h>
#include <math.h>

#define MY_ADC_RESOLUTION 4095.0  // 12-bit ADC maximum value (0 to 4095)

// Calibration parameters for the ADC-to-Lux conversion
// Using the relationship: LUX = G * (ADC value) + d
const float G = 0.0046;
const float d = 0.0115;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  Serial.println("ADC Value to Lux Measurement Initialized");
}

void loop() {
  // Read the ADC value from the analog input (e.g., from the filtered PWM signal)
  int adcValue = analogRead(A0);
  
  // Use the ADC value directly as u in the calibration equation
  float lux = G * adcValue + d;

  // Print the ADC value and the calculated Lux value
  Serial.print("ADC Value: ");
  Serial.print(adcValue);
  Serial.print("   Lux: ");
  Serial.println(lux, 3);

  delay(1000); // Wait 1 second before the next reading
}
