#include "CANComm.h"
#include <SPI.h>
#include "Globals.h"
#include "Metrics.h"
#include "LEDDriver.h"

// Use the same pin configuration as the working example
const int CAN_CS_PIN = 17;
const int CAN_MOSI_PIN = 19;
const int CAN_MISO_PIN = 16;
const int CAN_SCK_PIN = 18;

// Create the MCP2515 instance with matching pin configuration
MCP2515 can0(spi0, CAN_CS_PIN, CAN_MOSI_PIN, CAN_MISO_PIN, CAN_SCK_PIN, 10000000);

// Static variable to hold the registered callback
static CANMessageCallback messageCallback = nullptr;

// Statistics tracking
static uint32_t msgSent = 0;
static uint32_t msgReceived = 0;
static uint32_t msgErrors = 0;
static unsigned long lastLatencyMeasure = 0;
static unsigned long totalLatency = 0;
static uint32_t latencySamples = 0;

void initCANComm()
{
  // Initialize SPI
  SPI.begin();

  // Simple reset - like the working example
  can0.reset();

  // Use 1000KBPS like the example (not 125KBPS)
  can0.setBitrate(CAN_1000KBPS);

  // Set normal mode
  can0.setNormalMode();

  Serial.println("CANComm: CAN initialized in normal mode");

  // Generate a unique node ID from the last 6 bits of the board's unique ID
  pico_unique_board_id_t board_id;
  pico_get_unique_board_id(&board_id);
  nodeID = board_id.id[7] & 0x3F; // Use last 6 bits for node ID (1-63)
  if (nodeID == 0)
    nodeID = 1; // Avoid broadcast address

  Serial.print("CANComm: Node ID assigned: ");
  Serial.println(nodeID);
}

// Build a CAN message ID from components
uint32_t buildCANId(uint8_t msgType, uint8_t destAddr, uint8_t priority)
{
  return ((uint32_t)msgType << 8) | ((uint32_t)destAddr << 2) | priority;
}

// Extract components from a CAN ID
void parseCANId(uint32_t canId, uint8_t &msgType, uint8_t &destAddr, uint8_t &priority)
{
  msgType = (canId >> 8) & 0x07;
  destAddr = (canId >> 2) & 0x3F;
  priority = canId & 0x03;
}

// Convert float to bytes for CAN transmission (little-endian)
void floatToBytes(float value, uint8_t *bytes)
{
  memcpy(bytes, &value, 4);
}

// Convert bytes back to float (little-endian)
float bytesToFloat(const uint8_t *bytes)
{
  float value;
  memcpy(&value, bytes, 4);
  return value;
}

// Send a sensor reading to another node or broadcast
bool sendSensorReading(uint8_t destAddr, uint8_t sensorType, float value)
{
  can_frame frame;

  // Build the CAN ID
  frame.can_id = buildCANId(CAN_TYPE_SENSOR, destAddr, CAN_PRIO_NORMAL);

  // Set data length
  frame.can_dlc = 8;

  // Payload: source node, sensor type, value as float, timestamp
  frame.data[0] = nodeID;     // Source node
  frame.data[1] = sensorType; // Sensor type (0 = lux, 1 = duty, etc)

  // Float value (4 bytes)
  floatToBytes(value, &frame.data[2]);

  // Timestamp - 16-bit milliseconds counter
  uint16_t timestamp = (uint16_t)(millis() & 0xFFFF);
  frame.data[6] = timestamp & 0xFF;
  frame.data[7] = (timestamp >> 8) & 0xFF;

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

// Send a control command to another node
bool sendControlCommand(uint8_t destAddr, uint8_t controlType, float value)
{
  can_frame frame;

  // Build the CAN ID
  frame.can_id = buildCANId(CAN_TYPE_CONTROL, destAddr, CAN_PRIO_HIGH);

  // Set data length
  frame.can_dlc = 8;

  // Payload: source node, control type, value as float, sequence number
  frame.data[0] = nodeID;      // Source node
  frame.data[1] = controlType; // Control type

  // Float value (4 bytes)
  floatToBytes(value, &frame.data[2]);

  // Sequence number (for detecting lost messages)
  static uint16_t seqNum = 0;
  frame.data[6] = seqNum & 0xFF;
  frame.data[7] = (seqNum >> 8) & 0xFF;
  seqNum++;

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

// Send a response to a query message
bool sendQueryResponse(uint8_t destNode, float value)
{
  can_frame frame;
  frame.can_id = buildCANId(CAN_TYPE_RESPONSE, destNode, CAN_PRIO_NORMAL);
  frame.can_dlc = 8; // Use 8 bytes como todas as outras mensagens

  // Inclua o ID do nó emissor e um tipo de resposta
  frame.data[0] = nodeID; // ID do nó que está respondendo
  frame.data[1] = 2;      // Tipo 2 = resposta de consulta

  // Coloque o valor float em bytes 2-5
  floatToBytes(value, &frame.data[2]);

  // Bytes 6-7 podem ser usados para sequência ou deixados como zero
  frame.data[6] = 0;
  frame.data[7] = 0;

  // Envie a mensagem
  MCP2515::ERROR result = sendCANMessage(frame);

  Serial.print("DEBUG: Sent query response to node ");
  Serial.print(destNode);
  Serial.print(", value: ");
  Serial.println(value);

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

// Send a heartbeat message to indicate node is alive
bool sendHeartbeat()
{
  can_frame frame;

  // Build the CAN ID
  frame.can_id = buildCANId(CAN_TYPE_HEARTBEAT, CAN_ADDR_BROADCAST, CAN_PRIO_LOW);

  // Set data length
  frame.can_dlc = 6;

  // Payload: source node, status flags, uptime
  frame.data[0] = nodeID;

  // Status flags: bit0=feedback, bit1=occupancy
  uint8_t statusFlags = 0;
  if (feedbackControl)
    statusFlags |= 0x01;
  if (occupancy)
    statusFlags |= 0x02;
  frame.data[1] = statusFlags;

  // Node uptime in seconds
  uint32_t uptime = getElapsedTime();
  frame.data[2] = uptime & 0xFF;
  frame.data[3] = (uptime >> 8) & 0xFF;
  frame.data[4] = (uptime >> 16) & 0xFF;
  frame.data[5] = (uptime >> 24) & 0xFF;

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

void processIncomingMessage(const can_frame &msg)
{
  // Parse the CAN ID
  uint8_t msgType, destAddr, priority;
  parseCANId(msg.can_id, msgType, destAddr, priority);

  if (canMonitorEnabled)
  {
    Serial.print("DEBUG: Received CAN message, type ");
    Serial.print(msgType);
    Serial.print(", source ");
    Serial.print(msg.data[0]);
  }

  // Check if this message is for us (or broadcast)
  if (destAddr != nodeID && destAddr != CAN_ADDR_BROADCAST)
  {
    return; // Not for us
  }

  // Message is for us, process based on type
  uint8_t sourceNode = msg.data[0];

  switch (msgType)
  {
  case CAN_TYPE_SENSOR:
  {
    uint8_t sensorType = msg.data[1];
    float value = bytesToFloat(&msg.data[2]);
    uint16_t timestamp = ((uint16_t)msg.data[7] << 8) | msg.data[6];

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
  case CAN_TYPE_HEARTBEAT:
  {
    uint8_t statusFlags = msg.data[1];
    uint32_t uptime = ((uint32_t)msg.data[5] << 24) |
                      ((uint32_t)msg.data[4] << 16) |
                      ((uint32_t)msg.data[3] << 8) |
                      msg.data[2];

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
  case CAN_TYPE_CONTROL:
  {
    uint8_t controlType = msg.data[1];
    float value = bytesToFloat(&msg.data[2]);
    uint16_t sequence = ((uint16_t)msg.data[7] << 8) | msg.data[6];

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

    // Handle control commands
    if (controlType == 0)
    { // Setpoint
      setpointLux = value;
      // Send acknowledgment?
    }
    else if (controlType == 2)
    { // Echo request
      // Send echo response - use the received value
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
    else if (controlType == 3)
    { // Ping/discovery
      // Send a response to identify ourselves
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
    else if (controlType == 4) { // Set duty cycle
      setLEDDutyCycle(value);
      if (canMonitorEnabled) {
        Serial.print("CAN: Setting duty cycle to ");
        Serial.println(value);
      }
    }
    else if (controlType == 5) { // Set LED percentage
      setLEDPercentage(value);
      if (canMonitorEnabled) {
        Serial.print("CAN: Setting LED percentage to ");
        Serial.println(value);
      }
    }
    else if (controlType == 6) { // Set LED power in watts
      setLEDPower(value);
      if (canMonitorEnabled) {
        Serial.print("CAN: Setting LED power to ");
        Serial.println(value);
      }
    }
    else if (controlType == 7) { // Set occupancy
      occupancy = (value != 0.0f);
      if (canMonitorEnabled) {
        Serial.print("CAN: Setting occupancy to ");
        Serial.println(occupancy ? "true" : "false");
      }
    }
    else if (controlType == 8) { // Set anti-windup
      antiWindup = (value != 0.0f);
      if (canMonitorEnabled) {
        Serial.print("CAN: Setting anti-windup to ");
        Serial.println(antiWindup ? "true" : "false");
      }
    }
    else if (controlType == 9) { // Set feedback control
      feedbackControl = (value != 0.0f);
      if (canMonitorEnabled) {
        Serial.print("CAN: Setting feedback control to ");
        Serial.println(feedbackControl ? "true" : "false");
      }
    }
    else if (controlType == 10) { // Reference illuminance
      refIlluminance = value;
      setpointLux = value;
      
      if (canMonitorEnabled) {
        Serial.print("CAN: Setting reference illuminance to ");
        Serial.println(value);
      }
    }
    else if (controlType == 11)
    { // Stream start
      // Extract variable type from value
      int varCode = (int)value;
      String var = "y"; // Default

      if (varCode == 1)
        var = "u";
      else if (varCode == 2)
        var = "p";
      else if (varCode == 3)
        var = "o";
      else if (varCode == 4)
        var = "a";
      else if (varCode == 5)
        var = "f";
      else if (varCode == 6)
        var = "r";
      else if (varCode == 7)
        var = "v";
      else if (varCode == 8)
        var = "d";
      else if (varCode == 9)
        var = "t";
      else if (varCode == 10)
        var = "V";
      else if (varCode == 11)
        var = "F";
      else if (varCode == 12)
        var = "E";

      startStream(var, sourceNode);
    }
    else if (controlType == 12)
    { // Stream stop
      // Extract variable type from value
      int varCode = (int)value;
      String var = "y"; // Default

      if (varCode == 1)
        var = "u";
      else if (varCode == 2)
        var = "p";
      else if (varCode == 3)
        var = "o";
      else if (varCode == 4)
        var = "a";
      else if (varCode == 5)
        var = "f";
      else if (varCode == 6)
        var = "r";
      else if (varCode == 7)
        var = "v";
      else if (varCode == 8)
        var = "d";
      else if (varCode == 9)
        var = "t";
      else if (varCode == 10)
        var = "V";
      else if (varCode == 11)
        var = "F";
      else if (varCode == 12)
        var = "E";

      stopStream(var, sourceNode);
    }
    else if (controlType == 13)
    { // Luminaire state
      int stateVal = (int)value;
      if (stateVal == 0)
        changeState(STATE_OFF);
      else if (stateVal == 1)
        changeState(STATE_UNOCCUPIED);
      else if (stateVal == 2)
        changeState(STATE_OCCUPIED);
    }
    else if (controlType >= 20 && controlType <= 32)
    {
      // This is a query message, send back a response
      float responseValue = 0.0f;

      // Get the requested value
      switch (controlType)
      {
      case 20:
        responseValue = computeVisibilityErrorFromBuffer();
        break;
      case 21:
        responseValue = computeFlickerFromBuffer();
        break;
      case 22:
        responseValue = computeEnergyFromBuffer();
        break;
      case 23:
        responseValue = dutyCycle;
        break;
      case 24:
        responseValue = occupancy ? 1.0f : 0.0f;
        break;
      case 25:
        responseValue = antiWindup ? 1.0f : 0.0f;
        break;
      case 26:
        responseValue = feedbackControl ? 1.0f : 0.0f;
        break;
      case 27:
        responseValue = refIlluminance;
        break;
      case 28:
        Serial.println("Query for current illuminance");
        responseValue = readLux();
        break;
      case 29:
        responseValue = getPowerConsumption();
        break;
      case 30:
        responseValue = getElapsedTime();
        break;
      case 31:
        responseValue = getVoltageAtLDR();
        break;
      case 32:
        responseValue = getExternalIlluminance();
        break;
      default:
        return; // Unknown query type
      }
      // Send a response message with the value
      sendQueryResponse(sourceNode, responseValue);
    }
    break;
  }
  }
}

void canCommLoop()
{
  // Check for received messages
  can_frame msg;
  if (can0.readMessage(&msg) == MCP2515::ERROR_OK)
  {
    // Record statistics
    msgReceived++;

    // Process the message
    processIncomingMessage(msg);

    // If a callback has been registered, call it
    if (messageCallback)
    {
      messageCallback(msg);
    }
  }
}

MCP2515::ERROR sendCANMessage(const can_frame &frame)
{
  // Record send time for latency measurements
  lastLatencyMeasure = micros();

  // Send the message
  MCP2515::ERROR err = can0.sendMessage(&frame);

  // Update latency if successful (assumes hardware has sent the message)
  if (err == MCP2515::ERROR_OK)
  {
    unsigned long latency = micros() - lastLatencyMeasure;
    totalLatency += latency;
    latencySamples++;
  }

  return err;
}

MCP2515::ERROR readCANMessage(struct can_frame *frame)
{
  return can0.readMessage(frame);
}

void setCANMessageCallback(CANMessageCallback callback)
{
  messageCallback = callback;
}

// Get communication statistics
void getCANStats(uint32_t &sent, uint32_t &received, uint32_t &errors, float &avgLatency)
{
  sent = msgSent;
  received = msgReceived;
  errors = msgErrors;
  avgLatency = latencySamples > 0 ? (float)totalLatency / latencySamples : 0.0f;
}

// Reset communication statistics
void resetCANStats()
{
  msgSent = 0;
  msgReceived = 0;
  msgErrors = 0;
  totalLatency = 0;
  latencySamples = 0;
}
