#include <Arduino.h>
#include <math.h>
#include "pico/multicore.h"
#include <SPI.h>

#include "Globals.h"
#include "CANComm.h"
#include "CommandInterface.h"
#include "DataLogger.h"
#include "LEDDriver.h"
#include "Metrics.h"
#include "PIController.h"
#include "SensorManager.h"
#include "Calibration.h"

extern float readLux();
extern float getPowerConsumption();
extern float getVoltageAtLDR();
extern float getExternalIlluminance();

// Add near the top with other external declarations:
extern const float SETPOINT_OFF;
extern const float SETPOINT_UNOCCUPIED;
extern const float SETPOINT_OCCUPIED;

//==========================================================================================================================================================
// CONFIGURATION AND INITIALIZATION
//==========================================================================================================================================================

// CAN controller pin configuration
const int CAN_CS_PIN = 17;
const int CAN_MOSI_PIN = 19;
const int CAN_MISO_PIN = 16;
const int CAN_SCK_PIN = 18;

// MCP2515 CAN controller instance with 10MHz clock
MCP2515 can0(spi0, CAN_CS_PIN, CAN_MOSI_PIN, CAN_MISO_PIN, CAN_SCK_PIN, 10000000);

// Callback for custom message handling
static CANMessageCallback messageCallback = nullptr;

// Statistics for network performance monitoring
static uint32_t msgSent = 0;
static uint32_t msgReceived = 0;
static uint32_t msgErrors = 0;
static unsigned long lastLatencyMeasure = 0;
static unsigned long totalLatency = 0;
static uint32_t latencySamples = 0;
bool responseReceived = false;
uint8_t responseSourceNode = 0;
float responseValue = 0.0f;

extern bool canMonitorEnabled;

// Node discovery and tracking constants
const unsigned long NODE_TIMEOUT_MS = 15000; // 15 seconds without heartbeat marks node as inactive

// Structure to track discovered nodes
struct DiscoveredNode
{
  uint8_t nodeId;              // Node identifier
  unsigned long lastHeartbeat; // Time of last heartbeat (milliseconds)
  bool isActive;               // Is the node currently active
  uint8_t status;              // Status flags from last heartbeat
  unsigned long uptime;        // Node's reported uptime in seconds
};

// Array to store discovered nodes
static DiscoveredNode discoveredNodes[MAX_TRACKED_NODES];
static int numDiscoveredNodes = 0;

/**
 * Initialize the CAN communication interface
 * - Sets up SPI
 * - Configures CAN controller
 * - Assigns unique node ID based on board ID
 */
void initCANComm()
{
  // Initialize SPI for CAN controller communication
  SPI.begin();

  // Reset CAN controller to clear any previous state
  can0.reset();

  // Set CAN bus speed to 1Mbps
  can0.setBitrate(CAN_1000KBPS);

  // Set normal mode (not loopback or listen-only)
  can0.setNormalMode();

  initNodeDiscovery();

  Serial.print("CANComm: Node ID assigned: ");
  critical_section_enter_blocking(&commStateLock);
  Serial.println(deviceConfig.nodeId);
  critical_section_exit(&commStateLock);
}

//==========================================================================================================================================================
// MESSAGE FORMATTING AND PARSING
//==========================================================================================================================================================

/**
 * Create a CAN message ID from component fields
 *
 * @param msgType Message type (3 bits, 0-7)
 * @param destAddr Destination address (6 bits, 0-63)
 * @return Combined CAN ID
 */
uint32_t buildCANId(uint8_t msgType, uint8_t destAddr)
{
  // Format: [ msgType(3 bits) | destAddr(6 bits) | fixed bits(2) ]
  return ((uint32_t)msgType << 8) | ((uint32_t)destAddr << 2);
}

/**
 * Extract component fields from a CAN message ID
 *
 * @param canId CAN ID to parse
 * @param msgType Output parameter for message type
 * @param destAddr Output parameter for destination address
 */
void parseCANId(uint32_t canId, uint8_t &msgType, uint8_t &destAddr)
{
  // Extract message type (3 bits) and destination address (6 bits)
  msgType = (canId >> 8) & 0x07;
  destAddr = (canId >> 2) & 0x3F;
}

/**
 * Convert float to byte array for CAN transmission (little-endian)
 *
 * @param value Float value to convert
 * @param bytes Output byte array (must be at least 4 bytes)
 */
void floatToBytes(float value, uint8_t *bytes)
{
  memcpy(bytes, &value, 4);
}

/**
 * Convert byte array back to float (little-endian)
 *
 * @param bytes Input byte array (must be at least 4 bytes)
 * @return Reconstructed float value
 */
float bytesToFloat(const uint8_t *bytes)
{
  float value;
  memcpy(&value, bytes, 4);
  return value;
}

//==========================================================================================================================================================
// MESSAGE SENDING FUNCTIONS
//==========================================================================================================================================================

/**
 * Send a CAN message and track performance metrics
 *
 * @param frame CAN message to send
 * @return Error code from CAN controller
 */
MCP2515::ERROR sendCANMessage(const can_frame &frame)
{
  // Record send time for latency measurement
  lastLatencyMeasure = micros();

  // Send the message
  MCP2515::ERROR err = can0.sendMessage(&frame);

  // Update latency statistics if successful
  if (err == MCP2515::ERROR_OK)
  {
    unsigned long latency = micros() - lastLatencyMeasure;
    totalLatency += latency;
    latencySamples++;
  }

  return err;
}

/**
 * Send sensor reading to a specific node or broadcast
 *
 * @param destAddr Destination node address (0 for broadcast)
 * @param sensorType Type of sensor data (0=lux, 1=duty cycle, etc.)
 * @param value Sensor reading value
 * @return True if message sent successfully
 */
bool sendSensorReading(uint8_t destAddr, uint8_t sensorType, float value)
{
  can_frame frame;

  // Configure message ID with normal priority
  frame.can_id = buildCANId(CAN_TYPE_SENSOR, destAddr);
  frame.can_dlc = 8;

  // Payload format:
  // [0] = Source node ID
  // [1] = Sensor type
  // [2-5] = Float value (4 bytes)
  // [6-7] = Timestamp (16-bit milliseconds)
  critical_section_enter_blocking(&commStateLock);
  frame.data[0] = deviceConfig.nodeId;
  critical_section_exit(&commStateLock);
  frame.data[1] = sensorType;

  floatToBytes(value, &frame.data[2]);

  // Include millisecond timestamp for timing analysis
  uint16_t timestamp = (uint16_t)(millis() & 0xFFFF);
  frame.data[6] = timestamp & 0xFF;
  frame.data[7] = (timestamp >> 8) & 0xFF;

  // Send the message and update statistics
  MCP2515::ERROR result = sendCANMessage(frame);

  if (result == MCP2515::ERROR_OK)
  {
    msgSent++;
    return true;
  }
  else
  {
    msgErrors++;
    return false;
  }
}

/**
 * Send a control command to another node
 *
 * @param destAddr Destination node address
 * @param controlType Control command type
 * @param value Command parameter value
 * @return True if message sent successfully
 */
bool sendControlCommand(uint8_t destAddr, uint8_t controlType, float value)
{
  can_frame frame;

  // Configure message ID with high priority
  frame.can_id = buildCANId(CAN_TYPE_CONTROL, destAddr);
  frame.can_dlc = 8;

  // Payload format:
  // [0] = Source node ID
  // [1] = Control type
  // [2-5] = Float value (4 bytes)
  // [6-7] = Sequence number (16-bit)
  critical_section_enter_blocking(&commStateLock);
  frame.data[0] = deviceConfig.nodeId;
  critical_section_exit(&commStateLock);
  frame.data[1] = controlType;

  floatToBytes(value, &frame.data[2]);

  // Include sequence number for detecting lost messages
  static uint16_t seqNum = 0;
  frame.data[6] = seqNum & 0xFF;
  frame.data[7] = (seqNum >> 8) & 0xFF;
  seqNum++;

  // Send the message and update statistics
  MCP2515::ERROR result = sendCANMessage(frame);

  if (result == MCP2515::ERROR_OK)
  {
    msgSent++;
    return true;
  }
  else
  {
    msgErrors++;
    return false;
  }
}

/**
 * Send response to a query message
 *
 * @param destNode Destination node address
 * @param value Response value
 * @return True if message sent successfully
 */
bool sendQueryResponse(uint8_t destNode, float value)
{
  can_frame frame;

  // Configure message ID with normal priority
  frame.can_id = buildCANId(CAN_TYPE_RESPONSE, destNode);
  frame.can_dlc = 8;

  // Payload format:
  // [0] = Source node ID
  // [1] = Response type (2 = query response)
  // [2-5] = Float value (4 bytes)
  // [6-7] = Reserved (set to 0)
  critical_section_enter_blocking(&commStateLock);
  frame.data[0] = deviceConfig.nodeId;
  critical_section_exit(&commStateLock);
  frame.data[1] = 2; // Type 2 = query response

  floatToBytes(value, &frame.data[2]);

  frame.data[6] = 0;
  frame.data[7] = 0;

  // Send the message and log debug info
  MCP2515::ERROR result = sendCANMessage(frame);

  Serial.print("DEBUG: Sent query response to node ");
  Serial.print(destNode);
  Serial.print(", value: ");
  Serial.println(value);

  // Update statistics based on result
  if (result == MCP2515::ERROR_OK)
  {
    msgSent++;
    return true;
  }
  else
  {
    msgErrors++;
    return false;
  }
}

/**
 * Send a heartbeat message to indicate node presence
 * Periodically broadcast to maintain network awareness
 *
 * @return true if message was successfully queued, false otherwise
 */
bool sendHeartbeat()
{
  can_frame frame;

  // Configure message ID with heartbeat type and broadcast address
  frame.can_id = buildCANId(CAN_TYPE_HEARTBEAT, CAN_ADDR_BROADCAST);
  frame.can_dlc = 8;

  // Read current state safely
  critical_section_enter_blocking(&commStateLock);
  uint8_t nodeId = deviceConfig.nodeId;
  uint8_t statusFlags = 0;

  // Pack status flags: bit0=feedback, bit1-2=luminaireState
  statusFlags |= (controlState.feedbackControl ? 0x01 : 0x00);
  statusFlags |= ((uint8_t)controlState.luminaireState << 1) & 0x06;
  critical_section_exit(&commStateLock);

  // Payload format:
  // [0] = Source node ID
  // [1] = Status flags
  // [2-5] = Node uptime in seconds (32-bit little-endian)
  // [6-7] = Reserved (set to 0)
  frame.data[0] = nodeId;
  frame.data[1] = statusFlags;

  // Include uptime in seconds
  unsigned long uptime = millis() / 1000;
  frame.data[2] = uptime & 0xFF;
  frame.data[3] = (uptime >> 8) & 0xFF;
  frame.data[4] = (uptime >> 16) & 0xFF;
  frame.data[5] = (uptime >> 24) & 0xFF;

  // Reserved bytes
  frame.data[6] = 0;
  frame.data[7] = 0;

  // Send the message
  MCP2515::ERROR result = sendCANMessage(frame);

  if (result == MCP2515::ERROR_OK)
  {
    msgSent++;
    return true;
  }
  else
  {
    msgErrors++;
    return false;
  }
}

/**
 * Initialize node discovery system
 * Clears the tracked nodes array
 */
void initNodeDiscovery()
{
  for (int i = 0; i < MAX_TRACKED_NODES; i++)
  {
    discoveredNodes[i].isActive = false;
  }
  numDiscoveredNodes = 0;
}

/**
 * Process incoming heartbeat message
 * Updates node tracking data with received heartbeat information
 *
 * @param nodeId Source node ID
 * @param statusFlags Status flags from heartbeat message
 * @param uptime Node uptime in seconds
 */
void handleHeartbeatMessage(uint8_t nodeId, uint8_t statusFlags, unsigned long uptime)
{
  // Don't track our own node
  critical_section_enter_blocking(&commStateLock);
  bool isOurNode = (nodeId == deviceConfig.nodeId);
  critical_section_exit(&commStateLock);

  if (isOurNode)
  {
    return;
  }

  // This prevents phantom nodes from appearing in the list
  bool isValidNode = (nodeId > 0 && nodeId <= 63);

  // Also reject nodes with suspicious data (like zero uptime combined with status flags)
  if (!isValidNode || (uptime == 0 && statusFlags == 0))
  {
    if (canMonitorEnabled)
    {
      Serial.print("WARNING: Ignoring heartbeat from invalid node ID: ");
      Serial.println(nodeId);
    }
    return;
  }

  // Check if node already exists
  int nodeIndex = -1;
  for (int i = 0; i < numDiscoveredNodes; i++)
  {
    if (discoveredNodes[i].nodeId == nodeId)
    {
      nodeIndex = i;
      break;
    }
  }

  unsigned long currentTime = millis();

  // If node not found, add it
  if (nodeIndex == -1)
  {
    if (numDiscoveredNodes < MAX_TRACKED_NODES)
    {
      nodeIndex = numDiscoveredNodes++;

      if (canMonitorEnabled)
      {
        Serial.print("CAN: Discovered new node: ");
        Serial.println(nodeId);
      }
    }
    else
    {
      // Find oldest inactive node to replace
      unsigned long oldestTime = ULONG_MAX;
      for (int i = 0; i < MAX_TRACKED_NODES; i++)
      {
        if (!discoveredNodes[i].isActive && discoveredNodes[i].lastHeartbeat < oldestTime)
        {
          nodeIndex = i;
          oldestTime = discoveredNodes[i].lastHeartbeat;
        }
      }

      // If no inactive nodes, ignore new node
      if (nodeIndex == -1)
      {
        return;
      }
    }

    // Initialize new node entry
    discoveredNodes[nodeIndex].nodeId = nodeId;
    discoveredNodes[nodeIndex].isActive = true;
  }

  // Update node information
  discoveredNodes[nodeIndex].lastHeartbeat = currentTime;
  discoveredNodes[nodeIndex].status = statusFlags;
  discoveredNodes[nodeIndex].uptime = uptime;
  discoveredNodes[nodeIndex].isActive = true;
}

/**
 * Update status of discovered nodes
 * Marks nodes as inactive if no heartbeat received within timeout period
 * Should be called periodically
 */
void updateNodeStatus()
{
  unsigned long currentTime = millis();

  for (int i = 0; i < numDiscoveredNodes; i++)
  {
    if (discoveredNodes[i].isActive &&
        (currentTime - discoveredNodes[i].lastHeartbeat) > NODE_TIMEOUT_MS)
    {
      discoveredNodes[i].isActive = false;

      if (canMonitorEnabled)
      {
        Serial.print("CAN: Node ");
        Serial.print(discoveredNodes[i].nodeId);
        Serial.println(" marked inactive (timeout)");
      }
    }
  }
}

/**
 * Get list of active nodes
 *
 * @param nodeList Array to store active node IDs
 * @param maxNodes Maximum number of nodes to return
 * @return Number of active nodes found
 */
int getActiveNodes(uint8_t *nodeList, int maxNodes)
{
  int count = 0;

  for (int i = 0; i < numDiscoveredNodes && count < maxNodes; i++)
  {
    if (discoveredNodes[i].isActive)
    {
      nodeList[count++] = discoveredNodes[i].nodeId;
    }
  }

  return count;
}

/**
 * Display discovered nodes and their status
 * Prints a table of all discovered nodes with their status
 */
void displayDiscoveredNodes()
{
  Serial.print("\nDiscovered Nodes: of node ");
  Serial.println(deviceConfig.nodeId);
  Serial.println("----------------");
  Serial.println("ID | Active | Status | Uptime (s)");

  for (int i = 0; i < numDiscoveredNodes; i++)
  {
    Serial.print(discoveredNodes[i].nodeId);
    Serial.print(" | ");
    Serial.print(discoveredNodes[i].isActive ? "Yes" : "No");
    Serial.print(" | 0x");
    Serial.print(discoveredNodes[i].status, HEX);
    Serial.print(" | ");
    Serial.println(discoveredNodes[i].uptime);
  }

  Serial.println("----------------");
}

//==========================================================================================================================================================
// MESSAGE PROCESSING
//==========================================================================================================================================================

/**
 * Process an incoming CAN message
 *
 * @param msg The received CAN message
 */
void processIncomingMessage(const can_frame &msg)
{
  // Parse CAN ID into message type and destination address
  uint8_t msgType, destAddr;
  parseCANId(msg.can_id, msgType, destAddr);
  // Extract sender node ID from first byte
  uint8_t sourceNodeID = msg.data[0];

  // Validate the source node ID
  if (sourceNodeID == 0 || sourceNodeID > 63)
  {
    if (canMonitorEnabled)
    {
      Serial.print("WARNING: Invalid source node ID: ");
      Serial.println(sourceNodeID);
    }
    return;
  }

  // Process messages from other nodes
  critical_section_enter_blocking(&commStateLock);
  bool isOurNode = (sourceNodeID == deviceConfig.nodeId);
  critical_section_exit(&commStateLock);

  if (msgType == CAN_TYPE_HEARTBEAT)
  {
    // Extract heartbeat information
    uint8_t statusFlags = msg.data[1];

    // Extract node uptime from bytes 2-5 (little-endian)
    unsigned long nodeUptime =
        ((unsigned long)msg.data[5] << 24) |
        ((unsigned long)msg.data[4] << 16) |
        ((unsigned long)msg.data[3] << 8) |
        msg.data[2];

    // Handle the heartbeat message for discovery
    handleHeartbeatMessage(sourceNodeID, statusFlags, nodeUptime);
  }

  if (!isOurNode && msgType == CAN_TYPE_SENSOR)
  {
    uint8_t sensorType = msg.data[1];
    float sensorValue = bytesToFloat(&msg.data[2]);

    // Check if this is part of a streaming request
    bool isPartOfStream = false;
    const char *varName = nullptr;

    // NEW CODE: Check if we're in calibration mode
    critical_section_enter_blocking(&commStateLock);
    bool inCalibration = commState.calibrationInProgress;
    if (commState.streamingEnabled && commState.streamingVar != nullptr &&
        commState.streamingIndex == sourceNodeID)
    {

      // Map sensor type to variable name for output formatting
      if ((strcmp(commState.streamingVar, "y") == 0 && sensorType == 0) ||
          (strcmp(commState.streamingVar, "u") == 0 && sensorType == 1) ||
          (strcmp(commState.streamingVar, "p") == 0 && sensorType == 2) ||
          (strcmp(commState.streamingVar, "d") == 0 && sensorType == 3))
      {
        isPartOfStream = true;
        varName = commState.streamingVar;
      }
    }
    critical_section_exit(&commStateLock);

    // Also check REMOTE stream requests
    for (int i = 0; i < MAX_STREAM_REQUESTS; i++)
    {
      critical_section_enter_blocking(&commStateLock);
      bool isActiveRequest = commState.remoteStreamRequests[i].active &&
                             commState.remoteStreamRequests[i].requesterNode == deviceConfig.nodeId &&
                             commState.remoteStreamRequests[i].variableType == sensorType;
      critical_section_exit(&commStateLock);

      if (isActiveRequest)
      {
        isPartOfStream = true; // SET THE FLAG!

        // Map sensor type to variable name
        switch (sensorType)
        {
        case 0:
          varName = "y";
          break;
        case 1:
          varName = "u";
          break;
        case 2:
          varName = "p";
          break;
        case 3:
          varName = "d";
          break;
        default:
          varName = "?";
        }
        break;
      }
    }

    // If this is part of a stream, format and print in streaming format
    if ((isPartOfStream && varName != nullptr) && !inCalibration)
    {
      Serial.print(varName);
      Serial.print(" ");
      Serial.print(sourceNodeID);
      Serial.print(" ");
      Serial.println(sensorValue, 2);
    }
    // Otherwise print standard format if monitoring enabled or not part of stream
    else if ((!isPartOfStream && !inCalibration) || canMonitorEnabled)
    {
      /*Serial.print("Node ");
      Serial.print(sourceNodeID);
      Serial.print(" sent sensor type ");
      Serial.print(sensorType);
      Serial.print(" = ");
      Serial.println(sensorValue, 2);*/
    }
  }

  // Debug output if monitoring is enabled
  if (canMonitorEnabled && msgType != CAN_TYPE_HEARTBEAT)
  {
    Serial.print("DEBUG: Received CAN message, type ");
    Serial.print(msgType);
    Serial.print(", source ");
    Serial.println(msg.data[0]);
  }

  // Check if message is addressed to this node or is broadcast
  critical_section_enter_blocking(&commStateLock);
  if (destAddr != deviceConfig.nodeId && destAddr != CAN_ADDR_BROADCAST)
  {
    critical_section_exit(&commStateLock);
    return; // Message not for us, ignore
  }
  else
  {
    critical_section_exit(&commStateLock);
  }

  // Extract source node ID from first byte of payload
  uint8_t sourceNode = msg.data[0];

  // Process message based on type
  switch (msgType)
  {

  //-------------------------------------------------------------------------
  // SENSOR MESSAGE HANDLING
  //-------------------------------------------------------------------------
  case CAN_TYPE_SENSOR:
  {
    uint8_t sensorType = msg.data[1];
    float value = bytesToFloat(&msg.data[2]);
    uint16_t timestamp = ((uint16_t)msg.data[7] << 8) | msg.data[6];

    // If we're in calibration process, store readings from other nodes
    critical_section_enter_blocking(&commStateLock);
    bool inCalibration = commState.calibrationInProgress && commState.isCalibrationMaster;
    critical_section_exit(&commStateLock);

    if (inCalibration && sensorType == 0)
    { // Lux reading during calibration
      processCalibrationReading(sourceNode, value);

      if (canMonitorEnabled)
      {
        Serial.print("CALIB: Received lux reading from node ");
        Serial.print(sourceNode);
        Serial.print(": ");
        Serial.println(value);
      }
    }
    else if (sensorType == 1) // Duty cycle update
    {
      // Get the source node ID and duty cycle value
      uint8_t sourceNodeId = msg.data[0];
      float dutyCycle = bytesToFloat(&msg.data[2]);

      // Update neighbor info
      bool neighborFound = false;
      critical_section_enter_blocking(&commStateLock);
      // Try to update an existing neighbor
      for (int i = 0; i < MAX_NEIGHBORS; i++)
      {
        if (neighbors[i].nodeId == sourceNodeId)
        {
          neighbors[i].lastDuty = dutyCycle;
          neighbors[i].lastUpdate = millis();
          neighbors[i].isActive = true;
          neighborFound = true;
          break;
        }
      }

      // If not found, add as a new neighbor
      if (!neighborFound)
      {
        for (int i = 0; i < MAX_NEIGHBORS; i++)
        {
          if (!neighbors[i].isActive)
          {
            neighbors[i].nodeId = sourceNodeId;
            neighbors[i].lastDuty = dutyCycle;
            neighbors[i].lastUpdate = millis();
            neighbors[i].isActive = true;
            break;
          }
        }
      }
      critical_section_exit(&commStateLock);

      if (canMonitorEnabled)
      {
        Serial.print("Updated neighbor duty cycle: Node ");
        Serial.print(sourceNodeId);
        Serial.print(" = ");
        Serial.println(dutyCycle, 3);
      }
    }

    // Output debug info if monitoring enabled
    if (canMonitorEnabled)
    {
      Serial.print("CAN: Node ");
      Serial.print(sourceNode);
      Serial.print(" sensor ");
      Serial.print(sensorType);
      Serial.print(" = ");
      Serial.print(value);
      Serial.print(" (ts: ");
      Serial.print(timestamp);
      Serial.println(")");
    }
    break;
  }

  //-------------------------------------------------------------------------
  // CONTROL MESSAGE HANDLING
  //-------------------------------------------------------------------------
  case CAN_TYPE_CONTROL:
  {
    uint8_t sourceNodeID = msg.data[0];
    uint8_t controlType = msg.data[1];
    float value = bytesToFloat(&msg.data[2]);
    uint16_t sequence = ((uint16_t)msg.data[7] << 8) | msg.data[6];

    // Output debug info if monitoring enabled
    if (canMonitorEnabled)
    {
      Serial.print("CAN: Node ");
      Serial.print(sourceNode);
      Serial.print(" control ");
      Serial.print(controlType);
      Serial.print(" = ");
      Serial.print(value);
      Serial.print(" (seq: ");
      Serial.print(sequence);
      Serial.println(")");
    }

    // Handle different control commands based on type

    // Basic setpoint control
    if (controlType == 0)
    {
      critical_section_enter_blocking(&commStateLock);
      controlState.setpointLux = value;
      critical_section_exit(&commStateLock);
    }
    // Echo request - respond with same value
    else if (controlType == 2)
    {
      can_frame response;
      response.can_id = buildCANId(CAN_TYPE_RESPONSE, sourceNode);
      response.can_dlc = 8;
      critical_section_enter_blocking(&commStateLock);
      response.data[0] = deviceConfig.nodeId;
      critical_section_exit(&commStateLock);
      response.data[1] = 0;                   // Response type 0 = echo
      floatToBytes(value, &response.data[2]); // Echo back the same value
      response.data[6] = msg.data[6];         // Copy sequence numbers
      response.data[7] = msg.data[7];
      sendCANMessage(response);
    }
    // Ping/discovery - respond with node ID
    else if (controlType == 3)
    {
      can_frame response;
      response.can_id = buildCANId(CAN_TYPE_RESPONSE, sourceNode);
      response.can_dlc = 8;
      critical_section_enter_blocking(&commStateLock);
      response.data[0] = deviceConfig.nodeId;
      critical_section_exit(&commStateLock);
      response.data[1] = 1; // Response type 1 = discovery
      floatToBytes(0, &response.data[2]);
      response.data[6] = 0;
      response.data[7] = 0;
      sendCANMessage(response);
    }
    // LED output control commands
    else if (controlType == 4)
    { // Set duty cycle directly
      setLEDDutyCycle(value);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting duty cycle to ");
        Serial.println(value);
      }
    }
    else if (controlType == 5)
    { // Set LED percentage
      setLEDPercentage(value);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting LED percentage to ");
        Serial.println(value);
      }
    }
    else if (controlType == 6)
    { // Set LED power in watts
      setLEDPower(value);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting LED power to ");
        Serial.println(value);
      }
    }
    // System state control commands
    else if (controlType == 7)
    { // Set luminaireState
      critical_section_enter_blocking(&commStateLock);
      int stateVal = (int)value;
      if (stateVal >= 0 && stateVal <= 2)
      {
        controlState.luminaireState = static_cast<LuminaireState>(stateVal);
      }
      critical_section_exit(&commStateLock);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting luminaireState to ");
        Serial.println(controlState.luminaireState ? "true" : "false");
      }
    }
    else if (controlType == 8)
    { // Set anti-windup
      critical_section_enter_blocking(&commStateLock);
      controlState.antiWindup = (value != 0.0f);
      critical_section_exit(&commStateLock);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting anti-windup to ");
        Serial.println(controlState.antiWindup ? "true" : "false");
      }
    }
    else if (controlType == 9)
    { // Set feedback control
      critical_section_enter_blocking(&commStateLock);
      controlState.feedbackControl = (value != 0.0f);
      critical_section_exit(&commStateLock);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting feedback control to ");
        Serial.println(controlState.feedbackControl ? "true" : "false");
      }
    }
    else if (controlType == 10)
    { // Reference illuminance
      critical_section_enter_blocking(&commStateLock);
      controlState.setpointLux = value;
      critical_section_exit(&commStateLock);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting reference illuminance to ");
        Serial.print(value);
        Serial.println(" lux");
      }
    }
    // Stream control commands
    else if (controlType == 11)
    { // Start streaming
      float varCode = value;
      // Find an empty slot to store this streaming request
      int emptySlot = -1;
      for (int i = 0; i < MAX_STREAM_REQUESTS; i++)
      {
        if (!commState.remoteStreamRequests[i].active)
        {
          emptySlot = i;
          break;
        }
      }

      if (emptySlot >= 0)
      {
        critical_section_enter_blocking(&commStateLock);
        commState.remoteStreamRequests[emptySlot].requesterNode = sourceNode;
        commState.remoteStreamRequests[emptySlot].variableType = (int)varCode;
        commState.remoteStreamRequests[emptySlot].active = true;
        commState.remoteStreamRequests[emptySlot].lastSent = 0;
        critical_section_exit(&commStateLock);
      }
    }
    else if (controlType == 12)
    { // Stop streaming
      float varCode = value;
      // Find and deactivate the matching stream request
      for (int i = 0; i < MAX_STREAM_REQUESTS; i++)
      {
        if (commState.remoteStreamRequests[i].active &&
            commState.remoteStreamRequests[i].requesterNode == sourceNode &&
            commState.remoteStreamRequests[i].variableType == (int)varCode)
        {
          critical_section_enter_blocking(&commStateLock);
          commState.remoteStreamRequests[i].active = false;
          critical_section_exit(&commStateLock);
        }
      }
    }
    // Luminaire state control
    else if (controlType == 13)
    {
      int stateVal = (int)value;
      if (stateVal == 0)
        changeState(STATE_OFF);
      else if (stateVal == 1)
        changeState(STATE_UNOCCUPIED);
      else if (stateVal == 2)
        changeState(STATE_OCCUPIED);
    }
    else if (controlType == 14)
    { // Set filter enable/disable
      critical_section_enter_blocking(&commStateLock);
      sensorState.filterEnabled = (value != 0.0f);
      critical_section_exit(&commStateLock);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting sensor filtering to ");
        Serial.println(sensorState.filterEnabled ? "enabled" : "disabled");
      }
    }
    else if (controlType == 15)
    { // WakeUp command
      // Exit standby mode
      critical_section_enter_blocking(&commStateLock);
      controlState.standbyMode = false;

      // If remote wake-up, we're not the master but need to prepare for calibration
      if (!controlState.systemAwake)
      {
        controlState.systemAwake = true;
        controlState.discoveryStartTime = millis();
        commState.isCalibrationMaster = false; // We're not the master, just woken up
      }
      critical_section_exit(&commStateLock);

      // Send acknowledgment
      if (canMonitorEnabled)
      {
        Serial.println("Received wakeup command - exiting standby mode");
      }

      // Respond with acknowledgment
      sendControlCommand(sourceNodeID, 16, 1.0f); // 16 = wakeup acknowledgment
    }
    // Query commands (types 20-32)
    else if (controlType >= 20 && controlType <= 32)
    {
      // Get requested value based on query type
      float responseValue = 0.0f;

      switch (controlType)
      {
      case 20: // Visibility error
        responseValue = computeVisibilityErrorFromBuffer();
        break;
      case 21: // Flicker
        responseValue = computeFlickerFromBuffer();
        break;
      case 22: // Energy
        responseValue = computeEnergyFromBuffer();
        break;
      case 23: // Duty cycle
        critical_section_enter_blocking(&commStateLock);
        responseValue = controlState.dutyCycle;
        critical_section_exit(&commStateLock);
        break;
      case 24: // luminaireState state
        critical_section_enter_blocking(&commStateLock);
        responseValue = static_cast<float>(controlState.luminaireState);
        critical_section_exit(&commStateLock);
        break;
      case 25: // Anti-windup state
        critical_section_enter_blocking(&commStateLock);
        responseValue = controlState.antiWindup ? 1.0f : 0.0f;
        critical_section_exit(&commStateLock);
        break;
      case 26: // Feedback control state
        critical_section_enter_blocking(&commStateLock);
        responseValue = controlState.feedbackControl ? 1.0f : 0.0f;
        critical_section_exit(&commStateLock);
        break;
      case 27: // Reference illuminance
        critical_section_enter_blocking(&commStateLock);
        responseValue = controlState.setpointLux;
        critical_section_exit(&commStateLock);
        break;
      case 28: // Current illuminance
        Serial.println("Query for current illuminance");
        responseValue = readLux();
        break;
      case 29: // Power consumption
        responseValue = getPowerConsumption();
        break;
      case 30: // Elapsed time
        responseValue = getElapsedTime();
        break;
      case 31: // LDR voltage
        responseValue = getVoltageAtLDR();
        break;
      case 32: // External illuminance
        responseValue = getExternalIlluminance();
        break;
      default:
        return; // Unknown query type
      }

      // Send response with requested value
      sendQueryResponse(sourceNode, responseValue);
    }
    else if (controlType >= 100 && controlType < 110)
    {
      // This is a calibration control message
      handleCalibrationMessage(sourceNodeID, controlType, value);
    }
    else if (controlType >= CAL_CMD_INIT && controlType <= CAL_CMD_COMPLETE ||
             controlType == 120 || controlType == 121)
    {
      handleCalibrationMessage(sourceNodeID, controlType, value);
      return;
    }
    else if (controlType == 4)
    { // Set duty cycle
      // Handle duty cycle commands in calibration mode
      critical_section_enter_blocking(&commStateLock);
      bool isInCalibration = commState.calibrationInProgress;
      critical_section_exit(&commStateLock);

      if (isInCalibration)
      {
        // During calibration, always follow LED commands directly
        setLEDDutyCycle(value);
      }
      // ... existing duty cycle handling ...
    }
    break;
  }

    //-------------------------------------------------------------------------
    // RESPONSE MESSAGE HANDLING
    //-------------------------------------------------------------------------

  case CAN_TYPE_RESPONSE:
  {
    uint8_t responseType = msg.data[1];
    float value = bytesToFloat(&msg.data[2]);

    if (responseType == 2) // Type 2 = query response
    {
      // Print response for monitoring
      // Serial.print("CAN: Received query response from node ");
      // Serial.print(sourceNode);
      // Serial.print(", value: ");
      // Serial.println(value);

      // Set global flag that we got a response
      responseReceived = true;
      responseSourceNode = sourceNode;
      responseValue = value;
    }
    break;
  }
  }
}

/**
 * Main CAN communication processing loop
 * - Checks for incoming messages
 * - Processes messages and triggers callbacks
 */
void canCommLoop()
{
  can_frame msg;

  // Check if there's a message waiting
  MCP2515::ERROR err = can0.readMessage(&msg);
  if (err == MCP2515::ERROR_OK)
  {
    // Add debug prints to see ALL messages
    uint8_t msgType, destAddr;
    parseCANId(msg.can_id, msgType, destAddr);
    if (canMonitorEnabled && msgType != CAN_TYPE_HEARTBEAT)
    {
      Serial.print("RAW CAN: type=");
      Serial.print(msgType);
      Serial.print(", dest=");
      Serial.print(destAddr);
      Serial.print(", src=");
      Serial.println(msg.data[0]);
    }

    // Update statistics
    msgReceived++;

    // Process the received message
    processIncomingMessage(msg);

    // Call user-provided callback if registered
    if (messageCallback)
    {
      messageCallback(msg);
    }
  }
}

/**
 * Read a CAN message directly from the controller
 *
 * @param frame Pointer to store the received message
 * @return Error code from CAN controller
 */
MCP2515::ERROR readCANMessage(struct can_frame *frame)
{
  return can0.readMessage(frame);
}

//==========================================================================================================================================================
// UTILITY FUNCTIONS
//==========================================================================================================================================================

/**
 * Get communication statistics
 *
 * @param sent Output parameter for sent message count
 * @param received Output parameter for received message count
 * @param errors Output parameter for error count
 * @param avgLatency Output parameter for average message latency (microseconds)
 */
void getCANStats(uint32_t &sent, uint32_t &received, uint32_t &errors, float &avgLatency)
{
  sent = msgSent;
  received = msgReceived;
  errors = msgErrors;
  avgLatency = latencySamples > 0 ? (float)totalLatency / latencySamples : 0.0f;
}

/**
 * Reset all communication statistics
 */
void resetCANStats()
{
  msgSent = 0;
  msgReceived = 0;
  msgErrors = 0;
  totalLatency = 0;
  latencySamples = 0;
}

/**
 * Display CAN communication statistics
 * Shows message counts, error rates, and latency information
 */
void displayCANStatistics()
{
  uint32_t sent, received, errors;
  float avgLatency;

  // Get the statistics
  getCANStats(sent, received, errors, avgLatency);

  // Display statistics
  Serial.println("CAN Bus Statistics:");
  Serial.print("  Node ID: ");
  critical_section_enter_blocking(&commStateLock);
  Serial.println(deviceConfig.nodeId);
  critical_section_exit(&commStateLock);
  Serial.print("  Messages sent: ");
  Serial.println(sent);
  Serial.print("  Messages received: ");
  Serial.println(received);
  Serial.print("  Errors: ");
  Serial.println(errors);
  Serial.print("  Average latency: ");
  Serial.print(avgLatency);
  Serial.println(" µs");

  // Calculate error rate if messages were sent
  if (sent > 0)
  {
    float errorRate = (float)errors / (float)sent * 100.0f;
    Serial.print("  Error rate: ");
    Serial.print(errorRate, 2);
    Serial.println("%");
  }

  Serial.println("ack");
}

/**
 * Measure round-trip latency to a specific node
 *
 * @param numTokens Number of tokens in command
 * @param tokens Command tokens (tokens[2] should contain node ID if present)
 */
void measureCANLatency(int numTokens, char tokens[][TOKEN_MAX_LENGTH])
{
  // Default is to test current node
  uint8_t targetNode = 1;
  int samples = 10;

  // Parse target node if provided
  if (numTokens >= 3)
  {
    int node;
    if (parseIntParam(tokens[2], &node) && node > 0 && node < 64)
    {
      targetNode = (uint8_t)node;
    }
  }

  // Parse sample count if provided
  if (numTokens >= 4)
  {
    int count;
    if (parseIntParam(tokens[3], &count) && count > 0 && count <= 100)
    {
      samples = count;
    }
  }

  Serial.print("Measuring latency to node ");
  Serial.print(targetNode);
  Serial.print(" with ");
  Serial.print(samples);
  Serial.println(" samples...");

  float totalLatency = 0.0f;
  int successCount = 0;
  unsigned long startTime, endTime;
  const float testValue = 12345.67f; // Unique value to identify our test

  for (int i = 0; i < samples; i++)
  {
    Serial.print("  Sample ");
    Serial.print(i + 1);
    Serial.print(": ");

    // Record time and send echo request
    startTime = micros();
    if (sendControlCommand(targetNode, 2, testValue))
    {
      bool validResponse = false;

      // Wait for response with timeout
      for (int wait = 0; wait < 50; wait++)
      { // 50ms timeout
        can_frame frame;
        if (readCANMessage(&frame) == MCP2515::ERROR_OK)
        {
          uint8_t msgType, destAddr;
          parseCANId(frame.can_id, msgType, destAddr);

          if (msgType == CAN_TYPE_RESPONSE && frame.data[0] == targetNode && frame.data[1] == 0)
          {
            float value = bytesToFloat(&frame.data[2]);

            // Make sure it's a response to our test
            if (fabs(value - testValue) < 0.01f)
            {
              endTime = micros();
              unsigned long latency = endTime - startTime;
              totalLatency += latency;
              successCount++;
              validResponse = true;

              Serial.print(latency);
              Serial.println(" µs");
              break;
            }
          }
        }
        delay(1);
      }

      if (!validResponse)
      {
        Serial.println("timeout");
      }
    }
    else
    {
      Serial.println("send failed");
    }

    // Small delay between samples
    delay(10);
  }

  // Calculate and display results
  if (successCount > 0)
  {
    float avgLatency = totalLatency / successCount;
    Serial.print("Average round-trip latency: ");
    Serial.print(avgLatency);
    Serial.println(" µs");
    Serial.print("Success rate: ");
    Serial.print((float)successCount / samples * 100.0f);
    Serial.println("%");
  }
  else
  {
    Serial.println("No successful measurements");
  }

  Serial.println("ack");
}