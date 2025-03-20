#pragma once
#include <Arduino.h>

// Define structure for circular buffer entries
struct LogEntry
{
    unsigned long timestamp;
    float lux;
    float duty;
};

// Buffer size
#define LOG_SIZE 1200

// Initialize storage
void initStorage();

// Log a new data point
void logData(unsigned long timestamp, float lux, float duty);

// Output buffer to serial
void dumpBufferToSerial();

// Access buffer for computations
LogEntry *getLogBuffer();
int getLogCount();
bool isBufferFull();
int getCurrentIndex();

// Compute and output metrics
void computeAndPrintMetrics();
float computeEnergyFromBuffer();
float computeVisibilityErrorFromBuffer();
float computeFlickerFromBuffer();

// STREAMING FUNCTIONS
void startStream(const String &var, int index);
void stopStream(const String &var, int index);
void handleStreaming();
String getLastMinuteBuffer(const String &var, int index);