#ifndef CANCOMM_H
#define CANCOMM_H

#include <Arduino.h>
#include "mcp2515.h"

/**
 * CAN Communication Module
 *
 * Provides a comprehensive interface for CAN bus communication in a distributed
 * lighting control system. This module manages:
 * - Hardware initialization and configuration of the MCP2515 CAN controller
 * - Message formatting and parsing according to application protocol
 * - Higher-level semantic messaging functions for sensors and control
 * - Data type conversions between CAN frames and application data
 * - Statistical monitoring of communication performance
 *
 * The module abstracts away the complexity of CAN communication, providing
 * simple interfaces for sending sensor data, control commands, and status
 * updates between lighting nodes.
 */

//=============================================================================
// TYPE DEFINITIONS
//=============================================================================

/**
 * CAN message callback function type
 * Function pointer type for processing received CAN messages
 *
 * @param msg Reference to received CAN frame
 */
typedef void (*CANMessageCallback)(const can_frame &msg);

//=============================================================================
// INITIALIZATION FUNCTIONS
//=============================================================================

/**
 * Initialize the CAN communication subsystem
 * Configures the MCP2515 controller with appropriate speed and filters
 * Sets up interrupt handling and initializes statistics counters
 */
void initCANComm();

//=============================================================================
// MESSAGE BUILDING AND PARSING
//=============================================================================

/**
 * Build a CAN identifier from component fields
 * Combines message type, destination address, and priority into a standard
 * CAN identifier according to the application protocol format
 *
 * @param msgType Message type code (see CAN_TYPE_* constants)
 * @param destAddr Destination node address (0 for broadcast)
 * @param priority Message priority level (see CAN_PRIO_* constants)
 * @return Assembled 29-bit CAN identifier
 */
uint32_t buildCANId(uint8_t msgType, uint8_t destAddr, uint8_t priority);

/**
 * Parse a CAN identifier into component fields
 * Extracts message type, destination address, and priority from a CAN ID
 *
 * @param canId CAN identifier to parse
 * @param msgType Output parameter for message type
 * @param destAddr Output parameter for destination address
 * @param priority Output parameter for priority level
 */
void parseCANId(uint32_t canId, uint8_t &msgType, uint8_t &destAddr, uint8_t &priority);

//=============================================================================
// DATA CONVERSION UTILITIES
//=============================================================================

/**
 * Convert a float value to 4 bytes for CAN transmission
 * Handles endianness and IEEE-754 representation
 *
 * @param value Float value to convert
 * @param bytes Output array to store the 4 bytes (must be pre-allocated)
 */
void floatToBytes(float value, uint8_t *bytes);

/**
 * Convert 4 bytes from CAN message back to float
 * Inverse of floatToBytes function
 *
 * @param bytes Array of 4 bytes to convert
 * @return Reconstructed float value
 */
float bytesToFloat(const uint8_t *bytes);

//=============================================================================
// HIGH-LEVEL MESSAGE FUNCTIONS
//=============================================================================

/**
 * Send a sensor reading over the CAN bus
 * Formats and transmits a sensor value with appropriate metadata
 *
 * @param destAddr Destination node address (0 for broadcast)
 * @param sensorType Type of sensor data (0=lux, 1=duty, 2=state, etc.)
 * @param value Sensor reading value
 * @return true if message was successfully queued, false otherwise
 */
bool sendSensorReading(uint8_t destAddr, uint8_t sensorType, float value);

/**
 * Send a control command over the CAN bus
 * Formats and transmits a control instruction to one or all nodes
 *
 * @param destAddr Destination node address (0 for broadcast)
 * @param controlType Type of control command (see CAN_CTRL_* constants)
 * @param value Command parameter value
 * @return true if message was successfully queued, false otherwise
 */
bool sendControlCommand(uint8_t destAddr, uint8_t controlType, float value);

/**
 * Send a heartbeat message to indicate node presence
 * Periodically broadcast to maintain network awareness
 *
 * @return true if message was successfully queued, false otherwise
 */
bool sendHeartbeat();

//=============================================================================
// CORE CAN FUNCTIONS
//=============================================================================

/**
 * Process CAN message reception/transmission
 * Non-blocking function that should be called regularly in the main loop
 * Checks for received messages and processes them
 */
void canCommLoop();

/**
 * Send a raw CAN message frame
 * Low-level function for direct access to the CAN controller
 *
 * @param frame CAN frame structure with ID, length, and data
 * @return Error code (0 for success)
 */
MCP2515::ERROR sendCANMessage(const can_frame &frame);

/**
 * Read a CAN message from the controller
 * Low-level function to directly access incoming messages
 *
 * @param frame Pointer to CAN frame structure to store the message
 * @return Error code (0 for success)
 */
MCP2515::ERROR readCANMessage(struct can_frame *frame);

/**
 * Set a callback function for received CAN messages
 * The callback will be invoked for each received message
 *
 * @param callback Function pointer to the callback function
 */
void setCANMessageCallback(CANMessageCallback callback);

//=============================================================================
// STATISTICS AND DIAGNOSTICS
//=============================================================================

/**
 * Get communication statistics
 * Retrieves counters for monitoring network performance
 *
 * @param sent Output parameter for sent message count
 * @param received Output parameter for received message count
 * @param errors Output parameter for error count
 * @param avgLatency Output parameter for average message latency
 */
void getCANStats(uint32_t &sent, uint32_t &received, uint32_t &errors, float &avgLatency);

/**
 * Reset all communication statistics counters
 * Useful for beginning a new monitoring period
 */
void resetCANStats();

//=============================================================================
// HARDWARE ACCESS
//=============================================================================

/**
 * Global CAN controller instance
 * Provides direct access to the MCP2515 hardware driver
 */
extern MCP2515 can0;

#endif // CANCOMM_H