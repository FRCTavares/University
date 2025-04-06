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

extern float readLux();
extern float getPowerConsumption();
extern float getVoltageAtLDR();

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
const unsigned long NODE_TIMEOUT_MS = 15000;  // 15 seconds without heartbeat marks node as inactive
const int MAX_TRACKED_NODES = 16;             // Maximum number of nodes to track

// Structure to track discovered nodes
struct DiscoveredNode {
  uint8_t nodeId;                 // Node identifier
  unsigned long lastHeartbeat;    // Time of last heartbeat (milliseconds)
  bool isActive;                  // Is the node currently active
  uint8_t status;                 // Status flags from last heartbeat
  unsigned long uptime;           // Node's reported uptime in seconds
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
bool sendHeartbeat() {
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
  
  if (result == MCP2515::ERROR_OK) {
    msgSent++;
    return true;
  } else {
    msgErrors++;
    return false;
  }
}

/**
 * Initialize node discovery system
 * Clears the tracked nodes array
 */
void initNodeDiscovery() {
  for(int i = 0; i < MAX_TRACKED_NODES; i++) {
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
void handleHeartbeatMessage(uint8_t nodeId, uint8_t statusFlags, unsigned long uptime) {
  // Don't track our own node
  critical_section_enter_blocking(&commStateLock);
  bool isOurNode = (nodeId == deviceConfig.nodeId);
  critical_section_exit(&commStateLock);
  
  if (isOurNode) {
    return;
  }

  // This prevents phantom nodes from appearing in the list
  bool isValidNode = (nodeId > 0 && nodeId <= 63);

  // Also reject nodes with suspicious data (like zero uptime combined with status flags)
  if (!isValidNode || (uptime == 0 && statusFlags == 0)) {
    if (canMonitorEnabled) {
      Serial.print("WARNING: Ignoring heartbeat from invalid node ID: ");
      Serial.println(nodeId);
    }
    return;
  }
  
  // Check if node already exists
  int nodeIndex = -1;
  for(int i = 0; i < numDiscoveredNodes; i++) {
    if (discoveredNodes[i].nodeId == nodeId) {
      nodeIndex = i;
      break;
    }
  }
  
  unsigned long currentTime = millis();
  
  // If node not found, add it
  if (nodeIndex == -1) {
    if (numDiscoveredNodes < MAX_TRACKED_NODES) {
      nodeIndex = numDiscoveredNodes++;
      
      if (canMonitorEnabled) {
        Serial.print("CAN: Discovered new node: ");
        Serial.println(nodeId);
      }
    } else {
      // Find oldest inactive node to replace
      unsigned long oldestTime = ULONG_MAX;
      for(int i = 0; i < MAX_TRACKED_NODES; i++) {
        if (!discoveredNodes[i].isActive && discoveredNodes[i].lastHeartbeat < oldestTime) {
          nodeIndex = i;
          oldestTime = discoveredNodes[i].lastHeartbeat;
        }
      }
      
      // If no inactive nodes, ignore new node
      if (nodeIndex == -1) {
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
void updateNodeStatus() {
  unsigned long currentTime = millis();
  
  for(int i = 0; i < numDiscoveredNodes; i++) {
    if (discoveredNodes[i].isActive && 
        (currentTime - discoveredNodes[i].lastHeartbeat) > NODE_TIMEOUT_MS) {
      discoveredNodes[i].isActive = false;
      
      if (canMonitorEnabled) {
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
int getActiveNodes(uint8_t* nodeList, int maxNodes) {
  int count = 0;
  
  for(int i = 0; i < numDiscoveredNodes && count < maxNodes; i++) {
    if (discoveredNodes[i].isActive) {
      nodeList[count++] = discoveredNodes[i].nodeId;
    }
  }
  
  return count;
}

/**
 * Display discovered nodes and their status
 * Prints a table of all discovered nodes with their status
 */
void displayDiscoveredNodes() {
  Serial.print("\nDiscovered Nodes: of node ");
  Serial.println(deviceConfig.nodeId);
  Serial.println("----------------");
  Serial.println("ID | Active | Status | Uptime (s)");
  
  for(int i = 0; i < numDiscoveredNodes; i++) {
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
// CALIBRATION FUNCTIONS
//==========================================================================================================================================================

/**
 * Start a calibration sequence as the calibration master
 * This node takes control of the calibration process across the network
 * 
 * @return true if calibration started successfully
 */
bool startCalibration() {
  // Check if we have active nodes before starting calibration
  Serial.println("Starting calibration as master...");
  
  // First update node status to make sure our list is current
  updateNodeStatus();
  
  // Get list of active nodes
  uint8_t activeNodes[MAX_TRACKED_NODES];
  int numActiveNodes = getActiveNodes(activeNodes, MAX_TRACKED_NODES);
  
  // Add our own node to the count
  critical_section_enter_blocking(&commStateLock);
  uint8_t masterNodeId = deviceConfig.nodeId;
  critical_section_exit(&commStateLock);
  
  Serial.print("Found ");
  Serial.print(numActiveNodes);
  Serial.println(" active nodes to calibrate");
  
  // Display all nodes that will be included in calibration
  Serial.println("Calibration will include nodes:");
  Serial.print("  ");
  Serial.print(masterNodeId);
  Serial.println(" (master)");
  
  for (int i = 0; i < numActiveNodes; i++) {
    Serial.print("  ");
    Serial.println(activeNodes[i]);
  }
  
  // Send command to turn off all LEDs (broadcast)
  Serial.println("Initializing all nodes for calibration...");
  if (!sendControlCommand(CAN_ADDR_BROADCAST, 7, STATE_OFF)) {
    Serial.println("Failed to send broadcast OFF command");
    return false;
  }
  
  // Send command to reset reference values
  if (!sendControlCommand(CAN_ADDR_BROADCAST, 10, 0.0f)) {
    Serial.println("Failed to send setpoint reset command");
    return false;
  }
  
  // Apply locally as well to ensure consistency
  critical_section_enter_blocking(&commStateLock);
  controlState.luminaireState = STATE_OFF;
  controlState.setpointLux = 0.0f;
  
  // Initialize calibration state machine
  commState.isCalibrationMaster = true;
  commState.calibrationInProgress = true;
  commState.calibrationStep = 0; // Stage 1: Wait for acknowledgments
  commState.calLastStepTime = millis();
  commState.waitingForAcks = true;
  commState.acksReceived = 0;
  
  // Initialize calibration matrix
  commState.calibMatrix.numNodes = numActiveNodes + 1; // +1 for the master
  commState.calibMatrix.nodeIds[0] = masterNodeId; // First position is always the master
  
  // Add other nodes to the calibration matrix
  for (int i = 0; i < numActiveNodes; i++) {
    commState.calibMatrix.nodeIds[i+1] = activeNodes[i];
  }
  
  // Initialize all gains to zero
  for (int i = 0; i < MAX_CALIB_NODES; i++) {
    commState.calibMatrix.externalLight[i] = 0.0f;
    for (int j = 0; j < MAX_CALIB_NODES; j++) {
      commState.calibMatrix.gains[i][j] = 0.0f;
    }
  }
  
  // Reset temporary readings array
  for (int i = 0; i < MAX_CALIB_NODES; i++) {
    commState.luxReadings[i] = 0.0f;
  }
  
  // Reset stabilization check variables
  commState.readingIndex = 0;
  commState.measurementsStable = false;
  for (int i = 0; i < 5; i++) {
    commState.previousReadings[i] = 0.0f;
  }
  
  critical_section_exit(&commStateLock);
  
  // Send initialization command to all nodes
  Serial.println("Sending calibration initialization command...");
  if (!sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_INIT, (float)masterNodeId)) {
    Serial.println("Failed to send calibration initialization command");
    return false;
  }
  
  Serial.println("Calibration initialized! Waiting for acknowledgments...");
  return true;
}

/**
 * Process a calibration acknowledgment message
 * Updates the master's tracking of which nodes have acknowledged
 * 
 * @param nodeId Node ID that sent the acknowledgment
 * @return true if all expected nodes have acknowledged
 */
bool processCalibrationAck(uint8_t nodeId) {
  critical_section_enter_blocking(&commStateLock);
  
  if (!commState.isCalibrationMaster || !commState.waitingForAcks) {
    critical_section_exit(&commStateLock);
    return false;
  }
  
  // Check if this node is part of our calibration
  bool nodeIncluded = false;
  for (int i = 1; i < commState.calibMatrix.numNodes; i++) {
    if (commState.calibMatrix.nodeIds[i] == nodeId) {
      nodeIncluded = true;
      break;
    }
  }
  
  if (nodeIncluded) {
    commState.acksReceived++;
    Serial.print("Received acknowledgment from node ");
    Serial.print(nodeId);
    Serial.print(" (");
    Serial.print(commState.acksReceived);
    Serial.print(" of ");
    Serial.print(commState.calibMatrix.numNodes - 1);
    Serial.println(")");
  }
  
  bool allAcksReceived = (commState.acksReceived >= commState.calibMatrix.numNodes - 1);
  critical_section_exit(&commStateLock);
  
  return allAcksReceived;
}

/**
 * Acknowledge a calibration initialization request
 * Sent by nodes when they receive a calibration initialization command
 * 
 * @param masterNodeId ID of the calibration master
 */
void acknowledgeCalibration(uint8_t masterNodeId) {
  critical_section_enter_blocking(&commStateLock);
  
  if (commState.calibrationInProgress) {
    // Already in calibration mode, don't send another acknowledgment
    critical_section_exit(&commStateLock);
    return;
  }
  
  commState.calibrationInProgress = true;
  commState.isCalibrationMaster = false;
  commState.calibrationStep = 0;
  
  critical_section_exit(&commStateLock);
  
  Serial.print("Entering calibration mode, master is node ");
  Serial.println(masterNodeId);
  
  // Send acknowledgment to master
  bool ackSent = sendControlCommand(masterNodeId, CAL_CMD_ACK, (float)deviceConfig.nodeId);
  Serial.print("Sent calibration acknowledgment: ");
  Serial.println(ackSent ? "SUCCESS" : "FAILED");
}

/**
 * Process and store a light reading from another node during calibration
 * 
 * @param nodeId Source node ID
 * @param reading Light reading value
 */
void processCalibrationReading(uint8_t nodeId, float reading) {
  critical_section_enter_blocking(&commStateLock);
  
  if (!commState.isCalibrationMaster || commState.calibrationStep != 1) {
    critical_section_exit(&commStateLock);
    return;
  }
  
  // Find the node's index in our calibration matrix
  int nodeIndex = -1;
  for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
    if (commState.calibMatrix.nodeIds[i] == nodeId) {
      nodeIndex = i;
      break;
    }
  }
  
  if (nodeIndex >= 0) {
    commState.luxReadings[nodeIndex] = reading;
    Serial.print("Received reading from node ");
    Serial.print(nodeId);
    Serial.print(": ");
    Serial.println(reading, 4);
  }
  
  critical_section_exit(&commStateLock);
}

/**
 * Check if light measurements have stabilized
 * Uses statistical methods to determine if readings are consistent over time
 * 
 * @return true if measurements are stable
 */
bool areMeasurementsStable() {
  static unsigned long lastCheckTime = 0;
  unsigned long currentTime = millis();
  
  // Only check at regular intervals to allow light to stabilize
  if (currentTime - lastCheckTime < CAL_CHECK_INTERVAL_MS) {
    return false;
  }
  lastCheckTime = currentTime;
  
  // Read current illuminance
  float currentLux = readLux();
  
  critical_section_enter_blocking(&commStateLock);
  
  // Store in circular buffer
  commState.previousReadings[commState.readingIndex] = currentLux;
  commState.readingIndex = (commState.readingIndex + 1) % 5;
  
  // After we have enough readings, check for stability
  bool haveEnoughReadings = (currentTime - commState.stabilizationStartTime >= 500);
  
  if (haveEnoughReadings) {
    // Calculate mean
    float sum = 0.0f;
    for (int i = 0; i < 5; i++) {
      sum += commState.previousReadings[i];
    }
    float mean = sum / 5.0f;
    
    // Calculate variance
    float variance = 0.0f;
    for (int i = 0; i < 5; i++) {
      float diff = commState.previousReadings[i] - mean;
      variance += diff * diff;
    }
    variance /= 5.0f;
    
    // Check if variance is below threshold
    bool stable = (variance / (mean * mean) < CAL_STABILITY_THRESHOLD * CAL_STABILITY_THRESHOLD);
    
    if (stable) {
      commState.measurementsStable = true;
      critical_section_exit(&commStateLock);
      return true;
    }
  }
  
  critical_section_exit(&commStateLock);
  return false;
}

/**
 * Update calibration state machine
 * Called periodically from the main loop to advance the calibration process
 */
void updateCalibrationState() {
  unsigned long currentTime = millis();
  
  critical_section_enter_blocking(&commStateLock);
  
  if (!commState.calibrationInProgress) {
    critical_section_exit(&commStateLock);
    return;
  }

  // Master node state machine
  if (commState.isCalibrationMaster) {
    switch (commState.calibrationStep) {
      case 0: // Stage 1: Wait for acknowledgments
        if (commState.waitingForAcks) {
          // Check if we've received all expected acknowledgments
          if (commState.acksReceived >= commState.calibMatrix.numNodes - 1) {
            // All nodes acknowledged, move to measuring external light
            commState.waitingForAcks = false;
            commState.calibrationStep = 1; // Move to Stage 2
            commState.calLastStepTime = currentTime;
            commState.measurementsStable = false;
            commState.stabilizationStartTime = currentTime;
            
            Serial.println("All nodes acknowledged. Starting external light measurement...");
            critical_section_exit(&commStateLock);            
            // Turn off all LEDs for external light measurement
            sendControlCommand(CAN_ADDR_BROADCAST, 7, STATE_OFF);
            return;
          }
          
          // Check for timeout while waiting for acknowledgments
          if (currentTime - commState.calLastStepTime > CAL_TIMEOUT_MS) {
            Serial.print("Timeout waiting for acknowledgments. Received ");
            Serial.print(commState.acksReceived);
            Serial.print(" of ");
            Serial.print(commState.calibMatrix.numNodes - 1);
            Serial.println(" expected acknowledgments.");
            
            // Continue with the nodes that did acknowledge
            if (commState.acksReceived > 0) {  // CHANGE: Only proceed if we have at least one node
              Serial.println("Continuing with available nodes...");
              
              // Update nodes count to match actual acknowledged nodes
              commState.calibMatrix.numNodes = commState.acksReceived + 1;  // +1 for master
              
              commState.waitingForAcks = false;
              commState.calibrationStep = 1; // Move to Stage 2
              commState.currentCalNode = -1; // -1 means measuring external light
              commState.calLastStepTime = currentTime;
              commState.stabilizationStartTime = currentTime;
              
              // Make sure all LEDs are still off for external light measurement
              critical_section_exit(&commStateLock);
              sendControlCommand(CAN_ADDR_BROADCAST, 7, STATE_OFF);
              return;
            } else {
              // If no nodes acknowledged, abort calibration
              Serial.println("No nodes acknowledged. Aborting calibration.");
              commState.calibrationInProgress = false;
              commState.isCalibrationMaster = false;
              critical_section_exit(&commStateLock);
              return;
            }
          }
        }
        break;
        
      case 1: // Stage 2: Measure external light at all nodes
        {
          
          // Initial measurement phase - wait for lights to stabilize
          if (!commState.measurementsStable && currentTime - commState.stabilizationStartTime > CAL_STABILIZE_TIME_MS) {
            // Record our own external light measurement
            critical_section_exit(&commStateLock);
            float ourLux = readLux();
            critical_section_enter_blocking(&commStateLock);
            
            commState.luxReadings[0] = ourLux;
            
            Serial.print("Recorded our external light: ");
            Serial.println(ourLux);
            
            // Request light readings from all other nodes
            for (int i = 1; i < commState.calibMatrix.numNodes; i++) {
              uint8_t nodeId = commState.calibMatrix.nodeIds[i];
              Serial.print("Requesting external light reading from node ");
              Serial.println(nodeId);
              // Release lock before sending command
              critical_section_exit(&commStateLock);
              sendControlCommand(nodeId, CAL_CMD_SEND_READING, 0);
              critical_section_enter_blocking(&commStateLock);
            }
            
            commState.measurementsStable = true;
            commState.stabilizationStartTime = currentTime;
          }
          
          // After allowing time for responses, start sequential calibration
          // FIX: Changed from "if (commState.measurementsStable && ...)" to "if (commState.measurementsStable && ...)" 
          // to remove circular dependency
          if (commState.measurementsStable && currentTime - commState.stabilizationStartTime > 2000) {
            // Store external light readings in calibration matrix
            for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
              commState.calibMatrix.externalLight[i] = commState.luxReadings[i];
              Serial.print("External light at node ");
              Serial.print(commState.calibMatrix.nodeIds[i]);
              Serial.print(": ");
              Serial.println(commState.luxReadings[i]);
            }
            
            // Start sequential node calibration
            commState.currentCalNode = 0; // Start with first node (master)
            commState.calibrationStep = 2;
            commState.calLastStepTime = currentTime;
            commState.measurementsStable = false;
            
            Serial.print("Starting sequential node calibration. First node: ");
            Serial.println(commState.calibMatrix.nodeIds[0]);
            
            // If first node is us (master), start our calibration
            if (commState.calibMatrix.nodeIds[0] == deviceConfig.nodeId) {
              Serial.println("First node is us - starting our calibration");
              critical_section_exit(&commStateLock);
              
              // Turn off LED to prepare for calibration
              setLEDDutyCycle(0.0);
              return;
            } else {
              // Request first node to start its calibration
              uint8_t firstNodeId = commState.calibMatrix.nodeIds[commState.currentCalNode];
              Serial.print("Sending start command to node ");
              Serial.println(firstNodeId);
              
              critical_section_exit(&commStateLock);
              sendControlCommand(firstNodeId, CAL_CMD_START_NODE, commState.currentCalNode);
              return;
            }
          }
        }
      break;
      case 2: // Stage 3: Sequential node calibration
        {
          
          uint8_t currentNodeId = commState.calibMatrix.nodeIds[commState.currentCalNode];
          
          // If this is our node, run calibration
          if (currentNodeId == deviceConfig.nodeId) {
            if (!commState.measurementsStable) {
              Serial.println("Starting calibration of our own node");
              
              // Release lock before calling calibrateSystem
              critical_section_exit(&commStateLock);
              
              // Use existing calibration function
              float gain = calibrateSystem(0); // Reference value not needed
              
              critical_section_enter_blocking(&commStateLock);
              // Store self-gain in matrix
              int selfIdx = commState.currentCalNode;
              commState.calibMatrix.gains[selfIdx][selfIdx] = gain;
              
              // Store gain in device config
              deviceConfig.ledGain = gain;
              
              Serial.print("Self-calibration complete. Gain: ");
              Serial.println(gain);
              
              // Notify all nodes that our calibration is complete
              commState.measurementsStable = true;
              commState.calLastStepTime = currentTime;
              critical_section_exit(&commStateLock);
              sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, (float)commState.currentCalNode);
              return;
            }
          }
          
          // Check if we should move to next node
          if (commState.measurementsStable && currentTime - commState.calLastStepTime > 1000) {
            commState.currentCalNode++;
            
            if (commState.currentCalNode >= commState.calibMatrix.numNodes) {
              commState.calibrationStep = 3; // Move to finalization
              commState.calLastStepTime = currentTime;
              Serial.println("All nodes calibrated. Moving to finalization stage.");
              critical_section_exit(&commStateLock);
              return;
            } else {
              // Reset for next node
              commState.measurementsStable = false;
              commState.calLastStepTime = currentTime;
              
              Serial.print("Moving to next node: ");
              Serial.println(commState.calibMatrix.nodeIds[commState.currentCalNode]);
              
              // If next node is us, start our calibration
              if (commState.calibMatrix.nodeIds[commState.currentCalNode] == deviceConfig.nodeId) {
                Serial.println("Next node is us - starting our calibration");
                critical_section_exit(&commStateLock);
                
                // Turn off LED to prepare for calibration
                setLEDDutyCycle(0.0);
                return;
              } else {
                // Request other node to start calibration
                uint8_t nextNodeId = commState.calibMatrix.nodeIds[commState.currentCalNode];
                critical_section_exit(&commStateLock);
                sendControlCommand(nextNodeId, CAL_CMD_START_NODE, commState.currentCalNode);
                return;
              }
            }
          }
          critical_section_exit(&commStateLock);
          return;
        }
      break;
      case 3: // Stage 4: Finalize calibration
      {
        // Display calibration matrix after completion
        Serial.println("\nCalibration Matrix:");
        Serial.print("Nodes: ");
        for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
          Serial.print(commState.calibMatrix.nodeIds[i]);
          Serial.print(" ");
        }
        Serial.println();
        
        Serial.println("Gains Matrix:");
        for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
          for (int j = 0; j < commState.calibMatrix.numNodes; j++) {
            Serial.print(commState.calibMatrix.gains[i][j], 4);
            Serial.print(" ");
          }
          Serial.println();
        }
        
        Serial.println("External Light:");
        for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
          Serial.print(commState.calibMatrix.externalLight[i], 4);
          Serial.print(" ");
        }
        Serial.println();
        
        // Re-enable normal control
        controlState.luminaireState = STATE_UNOCCUPIED;
        controlState.setpointLux = SETPOINT_UNOCCUPIED;
        controlState.feedbackControl = true;
        commState.calibrationInProgress = false;
        commState.isCalibrationMaster = false;
        
        // Save calibration values to device configuration
        // We should already have our self-gain stored
        
        // Send notification to all nodes that calibration is complete
        sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_COMPLETE, 0);
        critical_section_exit(&commStateLock);
        
        // Send state change to broadcast - calibration is complete
        sendControlCommand(CAN_ADDR_BROADCAST, 7, STATE_UNOCCUPIED);
        return;
      }
    }
  }
  // Participant node state machine
  else {
  // Participant nodes mostly just follow commands from master
  // We need to handle our own calibration when instructed
  
  switch (commState.calibrationStep) {
    case 2: // Participating in sequential calibration
      {
        uint8_t currentNodeId = commState.calibMatrix.nodeIds[commState.currentCalNode];
        
        // If this is our node, run calibration
        if (currentNodeId == deviceConfig.nodeId && !commState.measurementsStable) {
          Serial.println("Calibrating own gain...");
          
          // Use existing calibration function
          float gain = calibrateSystem(0);
          
          // Store self-gain in matrix
          int selfIdx = commState.currentCalNode;
          commState.calibMatrix.gains[selfIdx][selfIdx] = gain;
          
          // Update gain in device config
          deviceConfig.ledGain = gain;
          
          Serial.print("Self-calibration complete. Gain: ");
          Serial.println(gain, 4);
          
          // Notify master that our calibration is complete
          sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, (float)commState.currentCalNode);
          
          commState.measurementsStable = true;
        }
      }
      break;
  }
  // Check for timeout on participant nodes
  if (currentTime - commState.calLastStepTime > 30000) {
    Serial.println("Calibration timeout. Returning to normal operation.");
    commState.calibrationInProgress = false;
    controlState.setpointLux = SETPOINT_UNOCCUPIED;
    controlState.luminaireState = STATE_UNOCCUPIED;
    controlState.feedbackControl = true;
  }
}

critical_section_exit(&commStateLock);
}

/**
 * Handle an incoming calibration control message
 * 
 * @param sourceNodeId ID of the node that sent the message
 * @param controlType Type of calibration command
 * @param value Command parameter value
 */
void handleCalibrationMessage(uint8_t sourceNodeId, uint8_t controlType, float value) {
  switch (controlType) {
    case CAL_CMD_INIT:
      // Initialize as participant in calibration
      acknowledgeCalibration(sourceNodeId);
      break;
      
    case CAL_CMD_ACK:
      // Process acknowledgment (master only)
      processCalibrationAck((uint8_t)value);
      break;
      
    case CAL_CMD_SEND_READING:
      {
        // Master is requesting our lux reading during calibration
        // This is called when master requests our external light reading
        // Read current illuminance
        float currentLux = readLux();
        // Send response to master with our current lux reading
        sendSensorReading(sourceNodeId, 0, currentLux);
        
        return;
      }
      
    case CAL_CMD_START_NODE:
      {
        // Master is instructing a specific node to start calibration
        int nodeIndex = (int)value;
        
        critical_section_enter_blocking(&commStateLock);
        commState.currentCalNode = nodeIndex;
        uint8_t currentNodeId = commState.calibMatrix.nodeIds[nodeIndex];
        
        // If this is us, start our calibration
        if (currentNodeId == deviceConfig.nodeId) {
          commState.measurementsStable = false;
          commState.calLastStepTime = millis();
          
          // Move to the calibration step if we're not already there
          commState.calibrationStep = 2;
          
          Serial.println("Starting own calibration sequence");
          
          // Start calibration by setting LED off first
          critical_section_exit(&commStateLock);
          setLEDDutyCycle(0.0);
        } else {
          critical_section_exit(&commStateLock);
        }
      }
    break;
      
    case CAL_CMD_NEXT_NODE:
      {
        // Node has completed its calibration, record cross-gains
        int nodeIndex = (int)value;
                
        critical_section_enter_blocking(&commStateLock);

        // Make sure we're in the correct calibration step
        if (commState.calibrationStep != 2) {
          critical_section_exit(&commStateLock);
          return;
        }

        // If we're the master, move to the next node
        if (commState.isCalibrationMaster) {
          // Update current node index and reset stability flag
          commState.currentCalNode = nodeIndex + 1;
          commState.measurementsStable = false;
          commState.calLastStepTime = millis();
          
          // If this completes the sequence, move to finalization
          if (commState.currentCalNode >= commState.calibMatrix.numNodes) {
            commState.calibrationStep = 3; // Move to finalization
            Serial.println("All nodes calibrated. Moving to finalization stage.");
          } else {
            // Print message about moving to next node
            Serial.print("Master moving to next node: ");
            Serial.println(commState.calibMatrix.nodeIds[commState.currentCalNode]);
            
            // If next node is us (master), start our calibration
            if (commState.calibMatrix.nodeIds[commState.currentCalNode] == deviceConfig.nodeId) {
              Serial.println("Next node is us (master) - starting our calibration");
              critical_section_exit(&commStateLock);
              
              // Turn off LED to prepare for calibration
              setLEDDutyCycle(0.0);
              return;
            } else {
              // Request next node to start its calibration
              uint8_t nextNodeId = commState.calibMatrix.nodeIds[commState.currentCalNode];
              critical_section_exit(&commStateLock);
              sendControlCommand(nextNodeId, CAL_CMD_START_NODE, commState.currentCalNode);
              return;
            }
          }
        }
      critical_section_exit(&commStateLock);
      }
      break;
      
    case CAL_CMD_COMPLETE:
      {
        // Calibration is complete, update state and return to normal operation
        critical_section_enter_blocking(&commStateLock);
        commState.calibrationInProgress = false;
        controlState.setpointLux = SETPOINT_UNOCCUPIED;
        controlState.luminaireState = STATE_UNOCCUPIED;
        controlState.feedbackControl = true;
        
        // Find and use self-gain from matrix
        int selfIdx = 0;
        for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
          if (commState.calibMatrix.nodeIds[i] == deviceConfig.nodeId) {
            selfIdx = i;
            break;
          }
        }
        deviceConfig.ledGain = commState.calibMatrix.gains[selfIdx][selfIdx];
        critical_section_exit(&commStateLock);
        
        Serial.println("Calibration completed by master. Returning to normal operation.");
        Serial.print("LED gain set to: ");
        Serial.println(deviceConfig.ledGain);
      }
      break;
  }
}

void displayCalibrationMatrix() {
  critical_section_enter_blocking(&commStateLock);
  
  Serial.println("\nCalibration Matrix:");
  Serial.print("Nodes: ");
  for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
    Serial.print(commState.calibMatrix.nodeIds[i]);
    Serial.print(" ");
  }
  Serial.println();
  
  Serial.println("Gains Matrix:");
  for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
    for (int j = 0; j < commState.calibMatrix.numNodes; j++) {
      Serial.print(commState.calibMatrix.gains[i][j], 4);
      Serial.print(" ");
    }
    Serial.println();
  }
  
  Serial.println("External Light:");
  for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
    Serial.print(commState.calibMatrix.externalLight[i], 4);
    Serial.print(" ");
  }
  Serial.println();
  
  // Display local gain value
  Serial.print("Local LED gain: ");
  Serial.println(deviceConfig.ledGain, 4);
  
  critical_section_exit(&commStateLock);
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
  if (sourceNodeID == 0 || sourceNodeID > 63) {
    if (canMonitorEnabled) {
      Serial.print("WARNING: Invalid source node ID: ");
      Serial.println(sourceNodeID);
    }
    return;
  }

  // Process messages from other nodes
  critical_section_enter_blocking(&commStateLock);
  bool isOurNode = (sourceNodeID == deviceConfig.nodeId);
  critical_section_exit(&commStateLock);

  if (msgType == CAN_TYPE_HEARTBEAT) {
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
    
    if (canMonitorEnabled)
    {
      Serial.print("CAN: Heartbeat from node ");
      Serial.println(sourceNodeID);
    }
  }
  
  if (!isOurNode && msgType == CAN_TYPE_SENSOR)
  {
    uint8_t sensorType = msg.data[1];
    float sensorValue = bytesToFloat(&msg.data[2]);

    // Check if this is part of a streaming request
    bool isPartOfStream = false;
    const char* varName = nullptr;
    
    critical_section_enter_blocking(&commStateLock);
    if (commState.streamingEnabled && commState.streamingVar != nullptr && 
        commState.streamingIndex == sourceNodeID) {
      
      // Map sensor type to variable name for output formatting
      if ((strcmp(commState.streamingVar, "y") == 0 && sensorType == 0) ||
          (strcmp(commState.streamingVar, "u") == 0 && sensorType == 1) ||
          (strcmp(commState.streamingVar, "p") == 0 && sensorType == 2) ||
          (strcmp(commState.streamingVar, "d") == 0 && sensorType == 3)) {
        isPartOfStream = true;
        varName = commState.streamingVar;
      }
    }
    critical_section_exit(&commStateLock);
    
    // Also check REMOTE stream requests
    for (int i = 0; i < MAX_STREAM_REQUESTS; i++) {
      critical_section_enter_blocking(&commStateLock);
      bool isActiveRequest = commState.remoteStreamRequests[i].active && 
                             commState.remoteStreamRequests[i].requesterNode == deviceConfig.nodeId &&
                             commState.remoteStreamRequests[i].variableType == sensorType;
      critical_section_exit(&commStateLock);
      
      if (isActiveRequest) {
        isPartOfStream = true;  // SET THE FLAG!
        
        // Map sensor type to variable name
        switch(sensorType) {
          case 0: varName = "y"; break;
          case 1: varName = "u"; break;
          case 2: varName = "p"; break;
          case 3: varName = "d"; break;
          default: varName = "?";
        }
        break;
      }
    }
    
    // If this is part of a stream, format and print in streaming format
    if (isPartOfStream && varName != nullptr) {
      Serial.print(varName);
      Serial.print(" ");
      Serial.print(sourceNodeID);
      Serial.print(" ");
      Serial.println(sensorValue, 2);
    }
    // Otherwise print standard format if monitoring enabled or not part of stream
    else if (!isPartOfStream || canMonitorEnabled) {
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
    Serial.println(msg.data[0]);
  }

  // Check if message is addressed to this node or is broadcast
  critical_section_enter_blocking(&commStateLock);
  if (destAddr != deviceConfig.nodeId && destAddr != CAN_ADDR_BROADCAST)
  {
    critical_section_exit(&commStateLock);
    return; // Message not for us, ignore
  } else
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
    else if (controlType == 15) { // DISABLE command
      // Disable the node
      critical_section_enter_blocking(&commStateLock);
      controlState.setpointLux = 0.0f;
      controlState.luminaireState = STATE_OFF;
      commState.periodicCANEnabled = false;
      critical_section_exit(&commStateLock);
      
      // Turn off LED
      setLEDDutyCycle(0.0f);
      
      if (canMonitorEnabled) {
        Serial.println("DEBUG: Node disabled by remote command");
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
  if (controlType >= 100 && controlType < 110) {
    // This is a calibration control message
    handleCalibrationMessage(sourceNodeID, controlType, value);
  }
  else if (controlType == 4) { // Set duty cycle
    // Handle duty cycle commands in calibration mode
    critical_section_enter_blocking(&commStateLock);
    bool isInCalibration = commState.calibrationInProgress;
    critical_section_exit(&commStateLock);
    
    if (isInCalibration) {
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
          //Serial.print("CAN: Received query response from node ");
          //Serial.print(sourceNode);
          //Serial.print(", value: ");
          //Serial.println(value);
          
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
    if(canMonitorEnabled)
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

/**
 * Register a callback function for CAN message handling
 *
 * @param callback Function to call when messages are received
 */
void setCANMessageCallback(CANMessageCallback callback)
{
  messageCallback = callback;
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
void displayCANStatistics() {
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
  if (sent > 0) {
      float errorRate = (float)errors / (float)sent * 100.0f;
      Serial.print("  Error rate: ");
      Serial.print(errorRate, 2);
      Serial.println("%");
  }
  
  Serial.println("ack");
}

/**
* Scan the CAN network for active nodes
* Sends discovery messages and waits for responses
*/
void scanCANNetwork() {
  Serial.println("Scanning CAN network for active nodes...");
  
  uint8_t foundNodes[64] = {0}; // Track which nodes responded
  int foundCount = 0;
  
  // Send ping messages to all possible node addresses (1-63)
  for (int node = 1; node < 64; node++) {
      Serial.print("Pinging node ");
      Serial.print(node);
      Serial.print("... ");
      
      // Send ping (discovery) message
      if (sendControlCommand(node, 3, 0.0f)) {
          // Wait briefly for response
          delay(10);
          
          // Check for response
          can_frame frame;
          bool received = false;
          
          // Try to read several times in case of multiple messages
          for (int attempt = 0; attempt < 5; attempt++) {
              if (readCANMessage(&frame) == MCP2515::ERROR_OK) {
                  uint8_t msgType, destAddr;
                  parseCANId(frame.can_id, msgType, destAddr);
                  
                  // Check if this is a response to our discovery
                  if (msgType == CAN_TYPE_RESPONSE && frame.data[0] == node && frame.data[1] == 1) {
                      foundNodes[node] = 1;
                      foundCount++;
                      received = true;
                      Serial.println("FOUND");
                      break;
                  }
              }
              delay(1);
          }
          
          if (!received) {
              Serial.println("no response");
          }
      } else {
          Serial.println("send failed");
      }
  }
  
  // Summary of found nodes
  Serial.print("Found ");
  Serial.print(foundCount);
  Serial.println(" active nodes:");
  
  for (int i = 1; i < 64; i++) {
      if (foundNodes[i]) {
          Serial.print("  Node ");
          Serial.println(i);
      }
  }
  
  Serial.println("Scan complete");
  Serial.println("ack");
}

/**
* Measure round-trip latency to a specific node
* 
* @param numTokens Number of tokens in command
* @param tokens Command tokens (tokens[2] should contain node ID if present)
*/
void measureCANLatency(int numTokens, char tokens[][TOKEN_MAX_LENGTH]) {
  // Default is to test current node
  uint8_t targetNode = 1;
  int samples = 10;
  
  // Parse target node if provided
  if (numTokens >= 3) {
      int node;
      if (parseIntParam(tokens[2], &node) && node > 0 && node < 64) {
          targetNode = (uint8_t)node;
      }
  }
  
  // Parse sample count if provided
  if (numTokens >= 4) {
      int count;
      if (parseIntParam(tokens[3], &count) && count > 0 && count <= 100) {
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
  
  for (int i = 0; i < samples; i++) {
      Serial.print("  Sample ");
      Serial.print(i+1);
      Serial.print(": ");
      
      // Record time and send echo request
      startTime = micros();
      if (sendControlCommand(targetNode, 2, testValue)) {
          bool validResponse = false;
          
          // Wait for response with timeout
          for (int wait = 0; wait < 50; wait++) { // 50ms timeout
              can_frame frame;
              if (readCANMessage(&frame) == MCP2515::ERROR_OK) {
                  uint8_t msgType, destAddr;
                  parseCANId(frame.can_id, msgType, destAddr);
                  
                  if (msgType == CAN_TYPE_RESPONSE && frame.data[0] == targetNode && frame.data[1] == 0) {
                      float value = bytesToFloat(&frame.data[2]);
                      
                      // Make sure it's a response to our test
                      if (fabs(value - testValue) < 0.01f) {
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
          
          if (!validResponse) {
              Serial.println("timeout");
          }
      } else {
          Serial.println("send failed");
      }
      
      // Small delay between samples
      delay(10);
  }
  
  // Calculate and display results
  if (successCount > 0) {
      float avgLatency = totalLatency / successCount;
      Serial.print("Average round-trip latency: ");
      Serial.print(avgLatency);
      Serial.println(" µs");
      Serial.print("Success rate: ");
      Serial.print((float)successCount / samples * 100.0f);
      Serial.println("%");
  } else {
      Serial.println("No successful measurements");
  }
  
  Serial.println("ack");
}