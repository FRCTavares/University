#include <Arduino.h>
#include <math.h>

// Define constants
#define VCC 3.3                    // Supply voltage in volts
#define MY_ADC_RESOLUTION 4095.0   // 12-bit ADC max value (0 to 4095)
#define FIXED_RESISTOR 10000.0     // 10k ohm resistor in the voltage divider
const float R10 = 225000.0; 

// Calibrated parameters for the LDR conversion equation
// Derived from the datasheet for PGM5659D:
// log10(R_LDR) = m * log10(Lux) + b, where m = -γ and b = log10(A)
// For a nominal R10 of ~225kΩ and γ ≈ 0.8, we get:
const float LDR_M = -1.2;             
float LDR_B = log10(R10) - LDR_M;

// Gain Calibration Constants (if needed)
const float G = 1; //0.2045; // Fator de ganho (ajustar experimentalmente)
const float d = 0; // 0.01; // Offset (ajustar experimentalmente)

void setup() {
  Serial.begin(115200);
  // Set analog resolution to 12 bits (if supported)
  #if defined(analogReadResolution)
  analogReadResolution(12);
  #endif

  Serial.println("LUX Measurement Initialized");
}

void loop() {
  // Read the ADC value from the analog pin connected to the voltage divider
  int adcValue = analogRead(A0);

  // Convert the ADC reading to voltage using the renamed constant
  float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;

  // Prevent division by zero if voltage is 0
  if (voltage <= 0) {
    Serial.println("Error: Voltage reading is 0.");
    delay(1000);
    return;
  }

  // Calculate the resistance of the LDR using the voltage divider formula:
  // V_out = VCC * (R_fixed / (R_fixed + R_LDR))
  // => R_LDR = R_fixed * (VCC/V_out - 1)
  float rLDR = FIXED_RESISTOR * (VCC / voltage - 1);

  // Compute LUX using the relationship:
  // log10(R_LDR) = m * log10(Lux) + b  ->  log10(Lux) = (log10(rLDR) - b) / m
  float lux = pow(10, (log10(rLDR) - LDR_B) / LDR_M);

  // Apply Gain Calibration (if needed)
  //float correctedLux = d + G * ;

  // Print the values to the Serial Monitor
  Serial.print("ADC Value: ");
  Serial.print(adcValue);
  Serial.print("\tVoltage: ");
  Serial.print(voltage, 3);
  Serial.print(" V\tLDR Resistance: ");
  Serial.print(rLDR, 2);
  Serial.print(" ohms\tLux: ");
  Serial.println(lux, 2);

  delay(1000); // Wait 1 second before the next reading
}
