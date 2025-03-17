#ifndef STORAGE_H
#define STORAGE_H

#define LOG_SIZE 1000

struct LogEntry {
  unsigned long timestamp; // [ms] from millis()
  float lux;              // measured lux (raw)
  float duty;             // duty cycle [0..1]
};

void initStorage();
void logData(unsigned long timestamp, float lux, float duty);
void dumpBufferToSerial();
LogEntry* getLogBuffer();
int getLogCount();
bool isBufferFull();
int getCurrentIndex();

#endif
