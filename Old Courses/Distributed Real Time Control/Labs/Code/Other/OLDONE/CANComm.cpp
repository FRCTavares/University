#include "CANComm.h"
#include "Controller.h"
#include <Arduino.h>
#include <math.h>
#include <SPI.h>


//-----------------------------------------------------------------------------
// CONFIGURATION AND INITIALIZATION
//-----------------------------------------------------------------------------

// CAN controller pin configuration
const int CAN_CS_PIN = 17;
const int CAN_MOSI_PIN = 19;
const int CAN_MISO_PIN = 16;
const int CAN_SCK_PIN = 18;

// MCP2515 CAN controller instance with 10MHz clockk
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

extern bool filterEnabled; // Add this line for sensor filtering toggle

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

  Serial.println("CANComm: CAN initialized in normal mode");

  // Generate unique node ID from board-specific identifier
  // Uses last 6 bits of unique ID to create node addresses 1-63
  pico_unique_board_id_t board_id;
  pico_get_unique_board_id(&board_id);
  nodeID = board_id.id[7] & 0x3F; // Use last 6 bits for node ID (1-63)
  if (nodeID == 0)
    nodeID = 1; // Avoid broadcast address (0)

  Serial.print("CANComm: Node ID assigned: ");
  Serial.println(nodeID);
}

//-----------------------------------------------------------------------------
// MESSAGE FORMATTING AND PARSING
//-----------------------------------------------------------------------------

/**
 * Create a CAN message ID from component fields
 *
 * @param msgType Message type (3 bits, 0-7)
 * @param destAddr Destination address (6 bits, 0-63)
 * @param priority Message priority (2 bits, 0-3)
 * @return Combined CAN ID
 */
uint32_t buildCANId(uint8_t msgType, uint8_t destAddr, uint8_t priority)
{
  return ((uint32_t)msgType << 8) | ((uint32_t)destAddr << 2) | priority;
}

/**
 * Extract component fields from a CAN message ID
 *
 * @param canId CAN ID to parse
 * @param msgType Output parameter for message type
 * @param destAddr Output parameter for destination address
 * @param priority Output parameter for message priority
 */
void parseCANId(uint32_t canId, uint8_t &msgType, uint8_t &destAddr, uint8_t &priority)
{
  msgType = (canId >> 8) & 0x07;
  destAddr = (canId >> 2) & 0x3F;
  priority = canId & 0x03;
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

//-----------------------------------------------------------------------------
// MESSAGE SENDING FUNCTIONS
//-----------------------------------------------------------------------------

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
  frame.can_id = buildCANId(CAN_TYPE_SENSOR, destAddr, CAN_PRIO_NORMAL);
  frame.can_dlc = 8;

  // Payload format:
  // [0] = Source node ID
  // [1] = Sensor type
  // [2-5] = Float value (4 bytes)
  // [6-7] = Timestamp (16-bit milliseconds)
  frame.data[0] = nodeID;
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
  frame.can_id = buildCANId(CAN_TYPE_CONTROL, destAddr, CAN_PRIO_HIGH);
  frame.can_dlc = 8;

  // Payload format:
  // [0] = Source node ID
  // [1] = Control type
  // [2-5] = Float value (4 bytes)
  // [6-7] = Sequence number (16-bit)
  frame.data[0] = nodeID;
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
  frame.can_id = buildCANId(CAN_TYPE_RESPONSE, destNode, CAN_PRIO_NORMAL);
  frame.can_dlc = 8;

  // Payload format:
  // [0] = Source node ID
  // [1] = Response type (2 = query response)
  // [2-5] = Float value (4 bytes)
  // [6-7] = Reserved (set to 0)
  frame.data[0] = nodeID;
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
 * Send heartbeat message to indicate node is active
 *
 * @return True if message sent successfully
 */
bool sendHeartbeat()
{
  can_frame frame;

  // Configure message ID for broadcast with low priority
  frame.can_id = buildCANId(CAN_TYPE_HEARTBEAT, CAN_ADDR_BROADCAST, CAN_PRIO_LOW);
  frame.can_dlc = 6;

  // Payload format:
  // [0] = Source node ID
  // [1] = Status flags (bit0=feedback, bit1=occupancy)
  // [2-5] = Node uptime in seconds (32-bit)
  frame.data[0] = nodeID;

  // Create status flags byte from control settings
  uint8_t statusFlags = 0;
  if (feedbackControl)
    statusFlags |= 0x01;
  if (occupancy)
    statusFlags |= 0x02;
  frame.data[1] = statusFlags;

  // Include node uptime in seconds
  uint32_t uptime = getElapsedTime();
  frame.data[2] = uptime & 0xFF;
  frame.data[3] = (uptime >> 8) & 0xFF;
  frame.data[4] = (uptime >> 16) & 0xFF;
  frame.data[5] = (uptime >> 24) & 0xFF;

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

//-----------------------------------------------------------------------------
// MESSAGE PROCESSING
//-----------------------------------------------------------------------------

/**
 * Process an incoming CAN message
 *
 * @param msg The received CAN message
 */
void processIncomingMessage(const can_frame &msg)
{
  // Parse CAN ID
  uint8_t msgType, destAddr, priority;
  parseCANId(msg.can_id, msgType, destAddr, priority);

  // Extract sender node ID from first byte
  uint8_t sourceNodeID = msg.data[0];

  // Always print messages from other nodes (not just when monitoring is enabled)
  if (sourceNodeID != nodeID)
  {
    if (msgType == CAN_TYPE_SENSOR)
    {
      uint8_t sensorType = msg.data[1];
      float sensorValue = bytesToFloat(&msg.data[2]);

      // Print regardless of monitor state when it's from another node
      Serial.print("Node ");
      Serial.print(sourceNodeID);
      Serial.print(" sent sensor type ");
      Serial.print(sensorType);
      Serial.print(" = ");
      Serial.println(sensorValue, 2);
    }
  }

  // Debug output if monitoring is enabled
  if (canMonitorEnabled)
  {
    Serial.print("DEBUG: Received CAN message, type ");
    Serial.print(msgType);
    Serial.print(", source ");
    Serial.print(msg.data[0]);
  }

  // Check if message is addressed to this node or is broadcast
  if (destAddr != nodeID && destAddr != CAN_ADDR_BROADCAST)
  {
    return; // Message not for us, ignore
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
  // HEARTBEAT MESSAGE HANDLING
  //-------------------------------------------------------------------------
  case CAN_TYPE_HEARTBEAT:
  {
    uint8_t statusFlags = msg.data[1];
    uint32_t uptime = ((uint32_t)msg.data[5] << 24) |
                      ((uint32_t)msg.data[4] << 16) |
                      ((uint32_t)msg.data[3] << 8) |
                      msg.data[2];

    // Output debug info if monitoring enabled
    if (canMonitorEnabled)
    {
      Serial.print("CAN: Node ");
      Serial.print(sourceNode);
      Serial.print(" heartbeat, uptime ");
      Serial.print(uptime);
      Serial.print("s, flags: ");
      Serial.println(statusFlags, BIN);
    }
    break;
  }

  //-------------------------------------------------------------------------
  // CONTROL MESSAGE HANDLING
  //-------------------------------------------------------------------------
  case CAN_TYPE_CONTROL:
  {
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
      setpointLux = value;
    }
    // Echo request - respond with same value
    else if (controlType == 2)
    {
      can_frame response;
      response.can_id = buildCANId(CAN_TYPE_RESPONSE, sourceNode, CAN_PRIO_HIGH);
      response.can_dlc = 8;
      response.data[0] = nodeID;
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
      response.can_id = buildCANId(CAN_TYPE_RESPONSE, sourceNode, CAN_PRIO_NORMAL);
      response.can_dlc = 8;
      response.data[0] = nodeID;
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
    { // Set occupancy
      occupancy = (value != 0.0f);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting occupancy to ");
        Serial.println(occupancy ? "true" : "false");
      }
    }
    else if (controlType == 8)
    { // Set anti-windup
      antiWindup = (value != 0.0f);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting anti-windup to ");
        Serial.println(antiWindup ? "true" : "false");
      }
    }
    else if (controlType == 9)
    { // Set feedback control
      feedbackControl = (value != 0.0f);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting feedback control to ");
        Serial.println(feedbackControl ? "true" : "false");
      }
    }
    else if (controlType == 10)
    { // Reference illuminance
      refIlluminance = value;
      setpointLux = value;

      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting reference illuminance to ");
        Serial.println(value);
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
        if (!remoteStreamRequests[i].active)
        {
          emptySlot = i;
          break;
        }
      }

      if (emptySlot >= 0)
      {
        remoteStreamRequests[emptySlot].requesterNode = sourceNode;
        remoteStreamRequests[emptySlot].variableType = (int)varCode;
        remoteStreamRequests[emptySlot].active = true;
        remoteStreamRequests[emptySlot].lastSent = 0;
      }
    }
    else if (controlType == 12)
    { // Stop streaming
      float varCode = value;
      // Find and deactivate the matching stream request
      for (int i = 0; i < MAX_STREAM_REQUESTS; i++)
      {
        if (remoteStreamRequests[i].active &&
            remoteStreamRequests[i].requesterNode == sourceNode &&
            remoteStreamRequests[i].variableType == (int)varCode)
        {
          remoteStreamRequests[i].active = false;
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
      filterEnabled = (value != 0.0f);
      if (canMonitorEnabled)
      {
        Serial.print("CAN: Setting sensor filtering to ");
        Serial.println(filterEnabled ? "enabled" : "disabled");
      }
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
        responseValue = dutyCycle;
        break;
      case 24: // Occupancy state
        responseValue = occupancy ? 1.0f : 0.0f;
        break;
      case 25: // Anti-windup state
        responseValue = antiWindup ? 1.0f : 0.0f;
        break;
      case 26: // Feedback control state
        responseValue = feedbackControl ? 1.0f : 0.0f;
        break;
      case 27: // Reference illuminance
        responseValue = refIlluminance;
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
    break;
  }
  }
}

//-----------------------------------------------------------------------------
// COMMUNICATION LOOP AND UTILITY FUNCTIONS
//-----------------------------------------------------------------------------

/**
 * Main CAN communication processing loop
 * - Checks for incoming messages
 * - Processes messages and triggers callbacks
 */
void canCommLoop()
{
  can_frame msg;

  // Check if there's a message waiting
  if (can0.readMessage(&msg) == MCP2515::ERROR_OK)
  {
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
 * Read a CAN message directly from the controller
 *
 * @param frame Pointer to store the received message
 * @return Error code from CAN controller
 */
MCP2515::ERROR readCANMessage(struct can_frame *frame)
{
  return can0.readMessage(frame);
}

/**
 * Register a callback function for CAN message handling
 *
 * @param callback Function to call when messages are received
 */
void setCANMessageCallback(CANMessageCallback callback)
{
  messageCallback = callback;
}

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