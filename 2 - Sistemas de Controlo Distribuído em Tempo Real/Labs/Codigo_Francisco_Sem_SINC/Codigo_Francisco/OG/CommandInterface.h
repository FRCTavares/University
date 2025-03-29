#ifndef COMMANDINTERFACE_H
#define COMMANDINTERFACE_H

#include <Arduino.h>

//=============================================================================
// COMMAND INTERFACE MODULE
//=============================================================================
/**
 * Command Interface Module
 *
 * Provides a serial command interface for controlling the lighting system.
 * Handles command parsing, execution, and response handling for functions like:
 * - LED control (brightness, power, etc.)
 * - System state control (occupancy, modes, etc.)
 * - Metrics and data query
 * - CAN bus communication between nodes
 * - Data streaming
 */

//=============================================================================
// SERIAL COMMAND PROCESSING
//=============================================================================

/**
 * Process any pending serial commands
 * This function should be called regularly in the main loop to check for
 * and execute commands received via the serial interface
 */
void processSerialCommands();

//=============================================================================
// DATA STREAMING FUNCTIONS
//=============================================================================

/**
 * Start streaming a variable to serial port
 *
 * @param var Variable to stream (y=illuminance, u=duty, etc.)
 * @param index Node index to stream
 */
void startStream(const char *var, int index);

/**
 * Stop streaming a variable
 *
 * @param var Variable to stop streaming
 * @param index Node index
 */
void stopStream(const char *var, int index);

/**
 * Get historical data buffer as CSV string
 *
 * @param var Variable type (y=illuminance, u=duty cycle)
 * @param index Node index
 * @return CSV string of historical values
 */
void getLastMinuteBuffer(const char *var, int index, char *buffer, size_t bufferSize);

#endif // COMMANDINTERFACE_H