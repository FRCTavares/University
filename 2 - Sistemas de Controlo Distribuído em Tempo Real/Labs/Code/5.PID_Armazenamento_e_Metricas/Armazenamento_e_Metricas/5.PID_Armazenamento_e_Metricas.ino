#include <Arduino.h>
#include <math.h>

// --------------------- Original PID + Sensor Calibration ---------------------
#define VCC 3.3
#define MY_ADC_RESOLUTION 4095.0
#define FIXED_RESISTOR 10000.0

// Example LDR reference at ~10 lux
const float R10 = 225000.0;
// log10(R_LDR) = m * log10(Lux) + b, with m = -1 (example)
const float LDR_M = -1.0;
float LDR_B = log10(R10) - LDR_M;  // offset for LDR eqn

// Pins
const int LED_PIN = 15;
const int LDR_PIN = A0;

// PID Gains (unchanged from your snippet)
float Kp = 200.0;
float Ki = 20.0;
float Kd = 5.0;
float N  = 10.0;  // derivative filter

// Sampling: 10 ms → 100 Hz
float h = 0.01;

// Operating Vars
float setpointLux = 15.0;
float Iterm = 0.0;
float Dterm = 0.0;
float e_old = 0.0;

// PWM limits
const int PWM_MAX = 4095;
const int PWM_MIN = 0;

// Ambient lux measured once at startup (LED off)
float ambientLux = 0.0;

// --------------------- For Energy Computation ---------------------
// A placeholder “max LED power,” in Watts. 
// If your LED can be 0.2A at 5V, that's 1.0W, etc.
const float Pmax = 1.0; 

// -----------------------------------------------------------------
// (1) CIRCULAR BUFFER FOR LOGGING
// -----------------------------------------------------------------
#define LOG_SIZE 1000

struct LogEntry {
  unsigned long timestamp; // [ms] from millis()
  float lux;               // total measured lux (raw)
  float duty;              // final duty cycle [0..1]
};
LogEntry logBuffer[LOG_SIZE];
int logIndex = 0;
bool isBufferFull = false;

// -----------------------------------------------------------------
// Forward Declarations for Additional Functions
// -----------------------------------------------------------------
void processSerialCommands();
void dumpBufferToSerial();
void computeAndPrintMetrics();
float computeEnergyFromBuffer();
float computeVisibilityErrorFromBuffer();
float computeFlickerFromBuffer();

// -----------------------------------------------------------------
void setup() {
  // EXACT PART OF YOUR PID SETUP
  analogReadResolution(12);
  analogWriteFreq(30000);
  analogWriteRange(PWM_MAX);

  Serial.begin(115200);
  Serial.println("PID Controller with Anti-Windup + Circular Buffer Logging.");

  // Measure ambient lux (LED off)
  analogWrite(LED_PIN, 0);
  delay(500);
  int adcAmbient = analogRead(LDR_PIN);
  float voltageAmbient = (adcAmbient / MY_ADC_RESOLUTION) * VCC;
  if (voltageAmbient > 0.0) {
    float rLDR_ambient = FIXED_RESISTOR * (VCC / voltageAmbient - 1);
    ambientLux = pow(10, (log10(rLDR_ambient) - LDR_B) / LDR_M);
  } else {
    ambientLux = 0.0;
  }
  Serial.print("Ambient lux measured: ");
  Serial.println(ambientLux, 2);

  Serial.println("Commands:");
  Serial.println("  <number>  -> new lux setpoint (e.g. 20)");
  Serial.println("  dump      -> dump circular buffer as CSV");
  Serial.println("  metrics   -> compute & print performance metrics");
  Serial.println();
}

// -----------------------------------------------------------------
void loop() {
  // (A) Process user commands from Serial
  processSerialCommands();

  // (B) EXACT COPY of your PID control code from snippet (minus some comments)
  // --------------------------------------------------------------------------
  // (2) Sensor Reading
  int adcValue = analogRead(LDR_PIN);
  float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
  if (voltage <= 0.0) {
    Serial.println("Error: Voltage reading is 0 or negative.");
    delay(1000);
    return;
  }
  float rLDR = FIXED_RESISTOR * (VCC / voltage - 1);
  float lux = pow(10, (log10(rLDR) - LDR_B) / LDR_M);

  // (3) Effective Lux, error
  float effectiveLux = lux - ambientLux;
  if (effectiveLux < 0) effectiveLux = 0;
  float e = setpointLux - effectiveLux;

  // (4) PID Computation
  float Pterm = Kp * e;
  float derivative = (e - e_old) / h;
  float alpha = N * h;
  Dterm = (alpha * derivative + Dterm) / (1 + alpha);
  float D_out = Kd * Dterm;

  float u_unsat = Pterm + Iterm + D_out;

  // (5) Anti-Windup Integral
  if ((u_unsat < PWM_MAX || e < 0) && (u_unsat > PWM_MIN || e > 0)) {
    Iterm += Ki * e * h;
  }

  // Recompute after integral
  float u = Pterm + Iterm + D_out;

  // (6) Saturation
  if (u > PWM_MAX) u = PWM_MAX;
  if (u < PWM_MIN) u = PWM_MIN;

  // (7) Write PWM
  analogWrite(LED_PIN, (int)u);

  // (8) Save Past Error
  e_old = e;

  // Debug
  Serial.print("SP: ");
  Serial.print(setpointLux);
  Serial.print(" | Measured Lux: ");
  Serial.print(lux, 2);
  Serial.print(" | Effective Lux: ");
  Serial.print(effectiveLux, 2);
  Serial.print(" | PWM: ");
  Serial.println(u, 2);

  // (C) Circular Buffer logging
  static unsigned long now;
  now = millis();
  float dutyCycle = u / PWM_MAX; // [0..1]

  logBuffer[logIndex].timestamp = now;
  logBuffer[logIndex].lux       = lux;      // raw measured
  logBuffer[logIndex].duty      = dutyCycle;

  logIndex++;
  if (logIndex >= LOG_SIZE) {
    logIndex = 0;
    isBufferFull = true;
  }

  // (D) Wait next sampling (10 ms => 100 Hz)
  delay((int)(h * 1000));
}

// -----------------------------------------------------------------
// (3) Additional Functions: Commands, Dump, Metrics
// -----------------------------------------------------------------
void processSerialCommands() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.equalsIgnoreCase("dump")) {
      dumpBufferToSerial();
    } 
    else if (input.equalsIgnoreCase("metrics")) {
      computeAndPrintMetrics();
    } 
    else {
      // Try to parse setpoint
      float newSetpoint = input.toFloat();
      if ((newSetpoint != 0.0) || input.equals("0")) {
        setpointLux = newSetpoint;
        Serial.print("New setpoint: ");
        Serial.println(setpointLux);
      }
    }
  }
}

void dumpBufferToSerial() {
  Serial.println("timestamp_ms,rawLux,duty");
  int count = isBufferFull ? LOG_SIZE : logIndex;
  int startIndex = isBufferFull ? logIndex : 0;
  for (int i = 0; i < count; i++) {
    int realIndex = (startIndex + i) % LOG_SIZE;
    unsigned long t = logBuffer[realIndex].timestamp;
    float lx        = logBuffer[realIndex].lux;
    float d         = logBuffer[realIndex].duty;
    Serial.print(t);
    Serial.print(",");
    Serial.print(lx, 2);
    Serial.print(",");
    Serial.println(d, 4);
  }
  Serial.println("End of dump.\n");
}

void computeAndPrintMetrics() {
  float E  = computeEnergyFromBuffer();
  float VE = computeVisibilityErrorFromBuffer();
  float F  = computeFlickerFromBuffer();

  Serial.println("----- Metrics from Circular Buffer -----");
  Serial.print("Energy (J approx): ");
  Serial.println(E, 4);
  Serial.print("Visibility Error (lux): ");
  Serial.println(VE, 2);
  Serial.print("Flicker: ");
  Serial.println(F, 4);
  Serial.println("----------------------------------------\n");
}

// -----------------------------------------------------------------
// (4) METRICS from buffer
// -----------------------------------------------------------------
float computeEnergyFromBuffer() {
  int count = isBufferFull ? LOG_SIZE : logIndex;
  if (count < 2) return 0.0;

  int startIndex = isBufferFull ? logIndex : 0;
  unsigned long prevTime = 0;
  float prevDuty = 0.0;
  bool first = true;
  float totalE = 0.0;

  for (int i = 0; i < count; i++) {
    int realIndex = (startIndex + i) % LOG_SIZE;
    unsigned long t = logBuffer[realIndex].timestamp;
    float d = logBuffer[realIndex].duty;
    if (!first) {
      float dt = (t - prevTime) / 1000.0;
      totalE += (Pmax * prevDuty * dt);
    } else {
      first = false;
    }
    prevTime = t;
    prevDuty = d;
  }
  return totalE;
}

float computeVisibilityErrorFromBuffer() {
  int count = isBufferFull ? LOG_SIZE : logIndex;
  if (count == 0) return 0.0;

  int startIndex = isBufferFull ? logIndex : 0;
  float totalErr = 0.0;
  int sampleCount= 0;

  for (int i = 0; i < count; i++) {
    int realIndex = (startIndex + i) % LOG_SIZE;
    float measuredLux = logBuffer[realIndex].lux;
    if (measuredLux < setpointLux) {
      totalErr += (setpointLux - measuredLux);
    }
    sampleCount++;
  }
  if (sampleCount == 0) return 0.0;
  return (totalErr / sampleCount);
}

float computeFlickerFromBuffer() {
  int count = isBufferFull ? LOG_SIZE : logIndex;
  if (count < 3) return 0.0;

  int startIndex = isBufferFull ? logIndex : 0;
  float flickerSum = 0.0;
  int flickerCount = 0;

  bool first = true, second = false;
  float d0, d1;

  for (int i = 0; i < count; i++) {
    int realIndex = (startIndex + i) % LOG_SIZE;
    float d2 = logBuffer[realIndex].duty;
    if (first) {
      d0 = d2;
      first = false;
      second= false;
    }
    else if (!second) {
      d1 = d2;
      second = true;
    }
    else {
      float diff1 = d1 - d0;
      float diff2 = d2 - d1;
      if (diff1 * diff2 < 0.0) {
        flickerSum += (fabs(diff1) + fabs(diff2));
        flickerCount++;
      }
      d0 = d1;
      d1 = d2;
    }
  }
  if (flickerCount == 0) return 0.0;
  return (flickerSum / flickerCount);
}
