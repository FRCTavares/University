#include "Storage.h"
#include <Arduino.h>

//=============================================================================
// CIRCULAR BUFFER DATA STORAGE
//=============================================================================

// Circular buffer for storing time-series data
LogEntry logBuffer[LOG_SIZE];

// Current write position in the buffer
int logIndex = 0;

// Flag indicating if buffer has wrapped around (contains LOG_SIZE entries)
bool bufferFull = false;

//=============================================================================
// INITIALIZATION FUNCTIONS
//=============================================================================

/**
 * Initialize the storage system
 * Resets buffer position and full flag
 */
void initStorage() {
  logIndex = 0;
  bufferFull = false;
}

//=============================================================================
// DATA LOGGING FUNCTIONS
//=============================================================================

/**
 * Log a data point to the circular buffer
 * Stores timestamp, illuminance, and duty cycle values
 * 
 * @param timestamp Millisecond timestamp when data was captured
 * @param lux Measured illuminance in lux
 * @param duty LED duty cycle (0.0-1.0)
 */
void logData(unsigned long timestamp, float lux, float duty) {
  // Input validation (optional)
  if (isnan(lux) || isnan(duty)) {
    // Skip invalid data
    return;
  }

  // Store data in current buffer position
  logBuffer[logIndex].timestamp = timestamp;
  logBuffer[logIndex].lux = lux;
  logBuffer[logIndex].duty = duty;
  
  // Advance write position
  logIndex++;
  
  // Wrap around if we reach the end of the buffer
  if (logIndex >= LOG_SIZE) {
    logIndex = 0;
    bufferFull = true;  // Mark that buffer has wrapped around
  }
}

/**
 * Output all logged data as CSV to the serial port
 * Formats: timestamp_ms,rawLux,duty
 */
void dumpBufferToSerial() {
  // Print CSV header
  Serial.println("timestamp_ms,rawLux,duty");
  
  // Calculate how many entries to output and where to start
  int count = bufferFull ? LOG_SIZE : logIndex;
  int startIndex = bufferFull ? logIndex : 0;
  
  // Output each entry in chronological order
  for (int i = 0; i < count; i++) {
    // Calculate real index accounting for circular buffer wrapping
    int realIndex = (startIndex + i) % LOG_SIZE;
    
    // Get data values
    unsigned long t = logBuffer[realIndex].timestamp;
    float lx = logBuffer[realIndex].lux;
    float d = logBuffer[realIndex].duty;
    
    // Output as CSV line
    Serial.print(t);
    Serial.print(",");
    Serial.print(lx, 2);  // 2 decimal places for lux
    Serial.print(",");
    Serial.println(d, 4); // 4 decimal places for duty cycle
  }
  
  Serial.println("End of dump.\n");
}

//=============================================================================
// DATA ACCESS FUNCTIONS
//=============================================================================

/**
 * Get direct access to the log buffer array
 * Use with caution - returns pointer to the actual buffer
 * 
 * @return Pointer to the log buffer array
 */
LogEntry* getLogBuffer() {
  return logBuffer;
}

/**
 * Get the number of valid entries in the buffer
 * 
 * @return Number of entries (maximum LOG_SIZE)
 */
int getLogCount() {
  return bufferFull ? LOG_SIZE : logIndex;
}

/**
 * Check if the buffer has filled completely at least once
 * 
 * @return true if buffer has wrapped around, false otherwise
 */
bool isBufferFull() {
  return bufferFull;
}

/**
 * Get current write position in buffer
 * 
 * @return Current buffer index
 */
int getCurrentIndex() {
  return logIndex;
}

//=============================================================================
// ADDITIONAL UTILITY FUNCTIONS
//=============================================================================

/**
 * Clear all data in the buffer
 * Resets buffer to empty state
 */
void clearBuffer() {
  logIndex = 0;
  bufferFull = false;
}

/**
 * Get the oldest timestamp in the buffer
 * 
 * @return Timestamp of oldest entry or 0 if buffer is empty
 */
unsigned long getOldestTimestamp() {
  if (getLogCount() == 0) {
    return 0;  // Buffer is empty
  }
  
  int startIndex = bufferFull ? logIndex : 0;
  return logBuffer[startIndex].timestamp;
}

/**
 * Get the newest timestamp in the buffer
 * 
 * @return Timestamp of newest entry or 0 if buffer is empty
 */
unsigned long getNewestTimestamp() {
  if (getLogCount() == 0) {
    return 0;  // Buffer is empty
  }
  
  int newestIndex = (logIndex == 0) ? (bufferFull ? LOG_SIZE - 1 : 0) : (logIndex - 1);
  return logBuffer[newestIndex].timestamp;
}

/**
 * Calculate buffer duration in milliseconds
 * 
 * @return Time span covered by buffer entries or 0 if fewer than 2 entries
 */
unsigned long getBufferDuration() {
  if (getLogCount() < 2) {
    return 0;  // Need at least 2 entries to calculate duration
  }
  
  return getNewestTimestamp() - getOldestTimestamp();
}

/**
 * Find the closest data entry to a given timestamp
 * 
 * @param timestamp Target timestamp to search for
 * @return Index of closest entry or -1 if buffer is empty
 */
int findClosestEntry(unsigned long timestamp) {
  int count = getLogCount();
  if (count == 0) {
    return -1;  // Buffer is empty
  }
  
  int startIndex = bufferFull ? logIndex : 0;
  int closestIndex = -1;
  unsigned long closestDiff = ULONG_MAX;
  
  for (int i = 0; i < count; i++) {
    int realIndex = (startIndex + i) % LOG_SIZE;
    unsigned long diff = abs((long)(logBuffer[realIndex].timestamp - timestamp));
    
    if (diff < closestDiff) {
      closestDiff = diff;
      closestIndex = realIndex;
    }
  }
  
  return closestIndex;
}