#pragma once

struct LogEntry {
    unsigned long timestamp;
    float lux;
    float duty;
};

// Initialize storage
void initStorage();

// Log data point
void logData(unsigned long timestamp, float lux, float duty);

// Data retrieval
LogEntry* getLogBuffer();
int getLogCount();
bool isBufferFull();
int getCurrentIndex();

// Utility functions
void dumpBufferToSerial();
String getLastMinuteBuffer(const String &var, int index);

// Metrics calculation
float computeEnergyFromBuffer();
float computeVisibilityErrorFromBuffer();
float computeFlickerFromBuffer();
void computeAndPrintMetrics();