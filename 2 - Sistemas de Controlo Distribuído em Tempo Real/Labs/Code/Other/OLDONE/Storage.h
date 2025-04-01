#ifndef STORAGE_H
#define STORAGE_H

//=============================================================================
// CONFIGURATION
//=============================================================================

/**
 * Maximum number of data points to store in the circular buffer
 * Defines the memory usage and time window of historical data
 */
#define LOG_SIZE 1000

//=============================================================================
// DATA STRUCTURES
//=============================================================================

/**
 * Data structure for a single time-series log entry
 * Each entry stores a timestamp and associated sensor/control values
 */
struct LogEntry {
  unsigned long timestamp;  // Time when entry was recorded (ms)
  float lux;               // Measured illuminance value (lux)
  float duty;              // LED duty cycle (0.0-1.0)
  float setpoint;          // Reference illuminance target
  float flicker;           // Individual flicker value at this point
  float jitter; // Jitter in µs (difference from nominal period)
  float extLux;            // External illuminance 
};

//=============================================================================
// INITIALIZATION FUNCTIONS
//=============================================================================

/**
 * Initialize the storage system
 * Resets buffer position and full flag
 */
void initStorage();

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
void logData(unsigned long timestamp, float lux, float duty);

/**
 * Output all logged data as CSV to the serial port
 * Formats: timestamp_ms,rawLux,duty
 */
void dumpBufferToSerial();

//=============================================================================
// DATA ACCESS FUNCTIONS
//=============================================================================

/**
 * Get direct access to the log buffer array
 * Use with caution - returns pointer to the actual buffer
 * 
 * @return Pointer to the log buffer array
 */
LogEntry* getLogBuffer();

/**
 * Get the number of valid entries in the buffer
 * 
 * @return Number of entries (maximum LOG_SIZE)
 */
int getLogCount();

/**
 * Check if the buffer has filled completely at least once
 * 
 * @return true if buffer has wrapped around, false otherwise
 */
bool isBufferFull();

/**
 * Get current write position in buffer
 * 
 * @return Current buffer index
 */
int getCurrentIndex();

//=============================================================================
// ADDITIONAL UTILITY FUNCTIONS
//=============================================================================

/**
 * Clear all data in the buffer
 * Resets buffer to empty state
 */
void clearBuffer();

/**
 * Get the oldest timestamp in the buffer
 * 
 * @return Timestamp of oldest entry or 0 if buffer is empty
 */
unsigned long getOldestTimestamp();

/**
 * Get the newest timestamp in the buffer
 * 
 * @return Timestamp of newest entry or 0 if buffer is empty
 */
unsigned long getNewestTimestamp();

/**
 * Calculate buffer duration in milliseconds
 * 
 * @return Time span covered by buffer entries or 0 if fewer than 2 entries
 */
unsigned long getBufferDuration();

/**
 * Find the closest data entry to a given timestamp
 * 
 * @param timestamp Target timestamp to search for
 * @return Index of closest entry or -1 if buffer is empty
 */
int findClosestEntry(unsigned long timestamp);

#endif // STORAGE_H