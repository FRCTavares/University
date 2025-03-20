#include "Metrics.h"
#include "Storage.h"
#include <Arduino.h>
#include <math.h>

// For Energy Computation
const float Pmax = 1.0;  // Max LED power in Watts
extern float setpointLux; // Declared in main.ino

void computeAndPrintMetrics() {
  float E = computeEnergyFromBuffer();
  float VE = computeVisibilityErrorFromBuffer();
  float F = computeFlickerFromBuffer();

  Serial.println("----- Metrics from Circular Buffer -----");
  Serial.print("Energy (J approx): ");
  Serial.println(E, 4);
  Serial.print("Visibility Error (lux): ");
  Serial.println(VE, 2);
  Serial.print("Flicker: ");
  Serial.println(F, 4);
  Serial.println("----------------------------------------\n");
}

float computeEnergyFromBuffer() {
  int count = getLogCount();
  if (count < 2) return 0.0;

  LogEntry* logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;
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
  int count = getLogCount();
  if (count == 0) return 0.0;

  LogEntry* logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;
  float totalErr = 0.0;
  int sampleCount = 0;

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
  int count = getLogCount();
  if (count < 3) return 0.0;

  LogEntry* logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;
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
      second = false;
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
