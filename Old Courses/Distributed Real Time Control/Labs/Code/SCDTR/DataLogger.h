#ifndef DATALOGGER_H
#define DATALOGGER_H

#include <Arduino.h>

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
struct LogEntry
{
  unsigned long timestamp; // Timestamp in milliseconds
  float lux;               // Illuminance in lux
  float duty;              // LED duty cycle
  float setpoint;          // Setpoint illuminance
  float flicker;           // Instantaneous flicker
  float avgEnergy;         // Running average energy consumption
  float avgVisError;       // Running average visibility error
  float avgFlicker;        // Running average flicker value
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
 * Formats: timestamp_ms,rawLux,duty,jitter_us
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
LogEntry *getLogBuffer();

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

#endif // DATALOGGER_H