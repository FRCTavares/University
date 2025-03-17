#include <Arduino.h>
#include <math.h>

// ---------------- Sensor Calibration Constants ----------------
#define VCC 3.3                    // Supply voltage (volts)
#define MY_ADC_RESOLUTION 4095.0   // 12-bit ADC max
#define FIXED_RESISTOR 10000.0     // 10 kΩ in voltage divider

// Example LDR reference at 10 lux (from datasheet)
const float R10 = 225000.0;
// Using: log10(R_LDR) = m*log10(Lux) + b, with m = -1 (example)
const float LDR_M = -1.0;
float LDR_B = log10(R10) - LDR_M;  // Offset for the LDR equation

// ---------------- Pins ----------------
const int LED_PIN = 15;   // PWM output pin to the LED
const int LDR_PIN = A0;   // ADC input for LDR

// ---------------- PID Gains ----------------
float Kp = 200.0;  // Proportional gain
float Ki = 20.0;   // Integral gain
float Kd = 5.0;    // Derivative gain
// Derivative filter parameter (optional)
float N  = 10.0;

// ---------------- Sampling ----------------
float h = 0.01;  // 10 ms → 100 Hz sampling rate

// ---------------- Operating Variables ----------------
float setpointLux = 15.0; // Desired lux (effective lux from LED only)
float Iterm = 0.0;        // Integral accumulator
float Dterm = 0.0;        // Derivative state (filtered)
float e_old = 0.0;        // Previous error

// PWM limits
const int PWM_MAX = 4095;
const int PWM_MIN = 0;

// ---------------- Ambient Lux ----------------
// Measured once at startup (with LED off)
float ambientLux = 0.0;

void setup() {
  // Hardware setup
  analogReadResolution(12);         // 12-bit ADC
  analogWriteFreq(30000);           // 30 kHz PWM frequency
  analogWriteRange(PWM_MAX);        // 0..4095 PWM range
  Serial.begin(115200);
  Serial.println("PID Controller with Anti-Windup Started.");
  
  // --- Measure ambient lux (with LED off) ---
  analogWrite(LED_PIN, 0);          // Ensure LED is off
  delay(500);                       // Wait for sensor to settle
  int adcAmbient = analogRead(LDR_PIN);
  float voltageAmbient = (adcAmbient / MY_ADC_RESOLUTION) * VCC;
  float rLDR_ambient = FIXED_RESISTOR * (VCC / voltageAmbient - 1);
  ambientLux = pow(10, (log10(rLDR_ambient) - LDR_B) / LDR_M);
  Serial.print("Ambient lux measured: ");
  Serial.println(ambientLux, 2);
}

void loop() {
  // -------- (1) Read/Update the Setpoint from Serial --------
  if (Serial.available() > 0) {
    String inStr = Serial.readStringUntil('\n');
    inStr.trim(); // Remove extra whitespace/newlines.
    if (inStr.length() > 0) {
      float newSetpoint = inStr.toFloat();
      // Allow 0 if explicitly entered
      if (newSetpoint != 0.0 || inStr == "0") { 
        setpointLux = newSetpoint;
        Serial.print("New setpoint: ");
        Serial.println(setpointLux);
      }
    }
  }

  // -------- (2) Sensor Reading + Lux Conversion --------
  int adcValue = analogRead(LDR_PIN);
  float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
  if (voltage <= 0.0) {
    Serial.println("Error: Voltage reading is 0 or negative.");
    delay(1000);
    return;
  }
  float rLDR = FIXED_RESISTOR * (VCC / voltage - 1);
  float lux = pow(10, (log10(rLDR) - LDR_B) / LDR_M);

  // -------- (3) Compute Effective Lux and Error --------
  // Effective lux is due solely to the LED (subtract ambient)
  float effectiveLux = lux - ambientLux;
  if(effectiveLux < 0) effectiveLux = 0;
  // Error = desired effective lux - measured effective lux
  float e = setpointLux - effectiveLux;

  // -------- (4) PID Computation --------
  // Proportional term
  float Pterm = Kp * e;
  
  // Derivative term (filtered on error)
  float derivative = (e - e_old) / h;
  float alpha = N * h;  // filter constant
  // Apply a simple first-order filter
  Dterm = (alpha * derivative + Dterm) / (1 + alpha);
  float D_out = Kd * Dterm;
  
  // Compute unsaturated control output
  float u_unsat = Pterm + Iterm + D_out;
  
  // -------- (5) Anti-Windup Integral Update --------
  // Update the integral term only if u_unsat is NOT saturated in the same direction as the error.
  if ( (u_unsat < PWM_MAX || e < 0) && (u_unsat > PWM_MIN || e > 0) ) {
    Iterm += Ki * e * h;
  }
  
  // Recompute control output after updating Iterm:
  float u = Pterm + Iterm + D_out;
  
  // -------- (6) Saturate Output --------
  if (u > PWM_MAX) u = PWM_MAX;
  if (u < PWM_MIN) u = PWM_MIN;
  
  // Write the PWM signal to the LED
  analogWrite(LED_PIN, (int)u);
  
  // -------- (7) Save Past Error --------
  e_old = e;
  
  // -------- (8) Debug Output --------
  // Print: SP | Measured Lux | Effective Lux | PWM
  Serial.print("SP: ");
  Serial.print(setpointLux);
  Serial.print(" | Measured Lux: ");
  Serial.print(lux, 2);
  Serial.print(" | Effective Lux: ");
  Serial.print(effectiveLux, 2);
  Serial.print(" | PWM: ");
  Serial.println(u, 2);
  
  // Wait for next sampling period (10 ms for 100 Hz)
  delay((int)(h * 1000));
}
