#ifndef CANCOMM_H
#define CANCOMM_H

#include <Arduino.h>
#include "mcp2515.h"
#include "Globals.h"

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
// CONSTANTS AND DEFINITIONS
//=============================================================================

/**
 * Maximum number of stream requests that can be tracked simultaneously
 * Limits the number of variables that can be streamed to other nodes
 */
#define MAX_STREAM_REQUESTS 5

/**
 * Special address for broadcasting messages to all nodes
 * Used when sending messages that should be received by every node
 */
#define CAN_ADDR_BROADCAST 0

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
 *
 * This function:
 * - Initializes SPI communication for the CAN controller
 * - Configures CAN bus speed to 1Mbps
 * - Sets the controller to normal operating mode
 * - Assigns a unique node ID based on the device's unique identifier
 */
void initCANComm();

//=============================================================================
// MESSAGE BUILDING AND PARSING
//=============================================================================

/**
 * Build a CAN identifier from component fields
 * Combines message type and destination address into a standard
 * CAN identifier (11-bit) according to the application protocol format
 *
 * Format: [ msgType(3 bits) | destAddr(6 bits) | fixed bits(2) ]
 *
 * @param msgType Message type code (3 bits, 0-7)
 * @param destAddr Destination node address (6 bits, 0-63)
 * @return Assembled 11-bit CAN identifier
 */
uint32_t buildCANId(uint8_t msgType, uint8_t destAddr);

/**
 * Parse a CAN identifier into component fields
 * Extracts message type and destination address from a CAN ID
 *
 * @param canId CAN identifier to parse
 * @param msgType Output parameter for message type
 * @param destAddr Output parameter for destination address
 */
void parseCANId(uint32_t canId, uint8_t &msgType, uint8_t &destAddr);

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
 * Payload format:
 * [0] = Source node ID
 * [1] = Sensor type (0=lux, 1=duty cycle, etc.)
 * [2-5] = Float value (4 bytes)
 * [6-7] = Timestamp (16-bit milliseconds)
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
 * Payload format:
 * [0] = Source node ID
 * [1] = Control type
 * [2-5] = Float value (4 bytes)
 * [6-7] = Sequence number (16-bit)
 *
 * Control types:
 * 0 = Set setpoint
 * 2 = Echo request (responds with same value)
 * 3 = Ping/discovery
 * 4 = Set duty cycle directly (0.0-1.0)
 * 5 = Set LED percentage (0-100%)
 * 6 = Set LED power in watts
 * 7 = Set luminaire state (0=off, 1=unoccupied, 2=occupied)
 * 8 = Set anti-windup (0=off, 1=on)
 * 9 = Set feedback control (0=off, 1=on)
 * 10 = Set reference illuminance (lux)
 * 11 = Start streaming specified variable
 * 12 = Stop streaming specified variable
 * 13 = Set luminaire state by name (0=off, 1=unoccupied, 2=occupied)
 * 14 = Set filter enable/disable (0=off, 1=on)
 * 20-32 = Various query commands (see implementation)
 *
 * @param destAddr Destination node address (0 for broadcast)
 * @param controlType Type of control command
 * @param value Command parameter value
 * @return true if message was successfully queued, false otherwise
 */
bool sendControlCommand(uint8_t destAddr, uint8_t controlType, float value);

/**
 * Send a response to a query message
 * Used to reply to data requests from other nodes
 *
 * Payload format:
 * [0] = Source node ID
 * [1] = Response type (2 = query response)
 * [2-5] = Float value (4 bytes)
 * [6-7] = Reserved (set to 0)
 *
 * @param destNode Destination node address
 * @param value Response value
 * @return True if message sent successfully
 */
bool sendQueryResponse(uint8_t destNode, float value);

/**
 * Send a heartbeat message to indicate node presence
 * Periodically broadcast to maintain network awareness
 *
 * Payload format:
 * [0] = Source node ID
 * [1] = Status flags (bit0=feedback, bit1-2=luminaireState)
 * [2-5] = Node uptime in seconds (32-bit)
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
 * @param avgLatency Output parameter for average message latency (microseconds)
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