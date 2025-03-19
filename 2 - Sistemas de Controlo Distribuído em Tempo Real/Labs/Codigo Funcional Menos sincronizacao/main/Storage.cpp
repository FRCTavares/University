#include "Storage.h"
#include <Arduino.h>

LogEntry logBuffer[LOG_SIZE];
int logIndex = 0;
bool bufferFull = false;

void initStorage() {
  logIndex = 0;
  bufferFull = false;
}

void logData(unsigned long timestamp, float lux, float duty) {
  logBuffer[logIndex].timestamp = timestamp;
  logBuffer[logIndex].lux = lux;
  logBuffer[logIndex].duty = duty;
  
  logIndex++;
  if (logIndex >= LOG_SIZE) {
    logIndex = 0;
    bufferFull = true;
  }
}

void dumpBufferToSerial() {
  Serial.println("timestamp_ms,rawLux,duty");
  int count = bufferFull ? LOG_SIZE : logIndex;
  int startIndex = bufferFull ? logIndex : 0;
  
  for (int i = 0; i < count; i++) {
    int realIndex = (startIndex + i) % LOG_SIZE;
    unsigned long t = logBuffer[realIndex].timestamp;
    float lx = logBuffer[realIndex].lux;
    float d = logBuffer[realIndex].duty;
    
    Serial.print(t);
    Serial.print(",");
    Serial.print(lx, 2);
    Serial.print(",");
    Serial.println(d, 4);
  }
  
  Serial.println("End of dump.\n");
}

LogEntry* getLogBuffer() {
  return logBuffer;
}

int getLogCount() {
  return bufferFull ? LOG_SIZE : logIndex;
}

bool isBufferFull() {
  return bufferFull;
}

int getCurrentIndex() {
  return logIndex;
}
