#include <Arduino.h>
#include <math.h>
#include "pico/multicore.h"

#include "Globals.h"
#include "CANComm.h"
#include "LEDDriver.h"
#include "Calibration.h"
#include "SensorManager.h"

extern float readLux();
extern void setLEDDutyCycle(float dutyCycle);
extern bool sendControlCommand(uint8_t destAddr, uint8_t controlType, float value);
extern bool sendSensorReading(uint8_t destAddr, uint8_t sensorType, float value);

/**
 * Perform comprehensive system calibration:
 * 1. Calibrate LDR sensor accuracy
 * 2. Measure LED contribution for external illuminance calculation
 *
 * @param referenceValue The reference illuminance value (typically very low like 1.0)
 * @return Calibrated LED gain value (G)
 */
float calibrateSystem(float referenceValue)
{
    const int SAMPLES = 5;               // Number of measurements to average
    const int STABILIZE_TIME = 500;      // Wait time between measurements in ms
    const int LED_RESPONSE_TIME = 10000; // Wait time for LDR to respond to LED changes

    // Arrays to store LED off and LED on readings from all nodes
    float offReadings[MAX_CALIB_NODES] = {0};
    float onReadings[MAX_CALIB_NODES] = {0};

    // ---- MEASURE WITH ALL LEDs OFF ----
    setLEDDutyCycle(0.0f);

    // Broadcast command to all nodes to turn off their LEDs
    sendControlCommand(CAN_ADDR_BROADCAST, 7, STATE_OFF);

    // Give time for all LEDs to turn off
    delay(STABILIZE_TIME);

    // Broadcast command to all nodes to record their EXTERNAL light reading (y1)
    sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_SEND_READING, 0);

    // Take multiple measurements and average for our node
    float y1 = 0.0;
    for (int i = 0; i < SAMPLES; i++)
    {
        y1 += readLux(); // Using calibrated readings now
        delay(STABILIZE_TIME);
    }
    y1 /= SAMPLES;

    // Store baseline illuminance for external light calculation
    critical_section_enter_blocking(&commStateLock);
    sensorState.baselineIlluminance = y1;
    int ourNodeIndex = commState.currentCalNode;
    uint8_t ourNodeId = deviceConfig.nodeId;

    // Read received values from other nodes (from CAN messages)
    for (int i = 0; i < commState.calibMatrix.numNodes; i++)
    {
        offReadings[i] = commState.luxReadings[i];

    }
    critical_section_exit(&commStateLock);

    deviceConfig.calibrationOffset = y1; // Calculate offset

    // ---- MEASURE WITH OUR LED ON ----
    setLEDDutyCycle(1.0);

    // Allow time for LED to reach full brightness and LDR to respond
    delay(LED_RESPONSE_TIME);

    // Broadcast command to all nodes to record readings WITH our LED ON (y2)
    sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_SEND_READING, 1);

    // Take multiple measurements and average for our node
    float y2 = 0.0;
    for (int i = 0; i < SAMPLES; i++)
    {
        y2 += readLux();
        delay(STABILIZE_TIME);
    }
    y2 /= SAMPLES;

    // ---- CALCULATE GAINS ----
    float gain = y2 - y1; // Self-gain (Kii)

    // Store gain in our device config
    deviceConfig.ledGain = gain;

    // Get the readings from other nodes with our LED on
    critical_section_enter_blocking(&commStateLock);
    for (int i = 0; i < commState.calibMatrix.numNodes; i++)
    {
        onReadings[i] = commState.luxReadings[i];

        // Print received readings for debugging
        if (commState.calibMatrix.nodeIds[i] != ourNodeId)
        {
            /*Serial.print("Node ");
            Serial.print(commState.calibMatrix.nodeIds[i]);
            Serial.print(" LED ON reading: ");
            Serial.println(onReadings[i]);*/
        }
    }

    // Update calibration matrix with our self-gain (diagonal element)
    commState.calibMatrix.gains[ourNodeIndex][ourNodeIndex] = gain;

    // Calculate and update gains for other nodes (how our LED affects them)
    for (int i = 0; i < commState.calibMatrix.numNodes; i++)
    {
        if (i != ourNodeIndex)
        {
            // Calculate effect of our LED on this node
            float nodeGain = onReadings[i] - offReadings[i];

            // Store in calibration matrix [row=receiver node][col=sender node]
            commState.calibMatrix.gains[i][ourNodeIndex] = nodeGain;

        }
    }
    critical_section_exit(&commStateLock);

    // Broadcast our gain to all nodes (this will be stored as the Kii value)
    sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, gain);

    // Also broadcast the effect on other nodes so they can update their matrices
    for (int i = 0; i < commState.calibMatrix.numNodes; i++)
    {
        if (i != ourNodeIndex)
        {
            // Get the node ID and the effect our LED had on them
            critical_section_enter_blocking(&commStateLock);
            uint8_t targetNodeId = commState.calibMatrix.nodeIds[i];
            float effectValue = commState.calibMatrix.gains[i][ourNodeIndex];
            critical_section_exit(&commStateLock);

            // Send this as a specific calibration update
            // We'll use a custom message format:
            // targetNode is the receiver, value is packed: ourNodeIndex * 1000 + effectValue
            // This requires modifying handleCalibrationMessage to unpack this
            float packedValue = ourNodeIndex * 1000.0f + effectValue;
            sendControlCommand(targetNodeId, CAL_CMD_NEXT_NODE, packedValue);

            delay(50); // Small delay to prevent flooding the CAN bus
        }
    }

    // Reset LED to off state after calibration
    setLEDDutyCycle(0.0);

    return gain;
}

/**
 * Start a calibration sequence as the calibration master
 * This node takes control of the calibration process across the network
 *
 * @return true if calibration started successfully
 */
bool startCalibration()
{
    // Serial.println("Starting calibration as master...");

    // Update node status to make sure our list is current
    updateNodeStatus();

    // Get list of active nodes
    uint8_t activeNodes[MAX_TRACKED_NODES];
    int numActiveNodes = getActiveNodes(activeNodes, MAX_TRACKED_NODES);

    critical_section_enter_blocking(&commStateLock);
    uint8_t masterNodeId = deviceConfig.nodeId;

    // Initialize calibration state
    commState.isCalibrationMaster = true;
    commState.calibrationInProgress = true;
    commState.calibrationStep = 0;
    commState.calLastStepTime = millis();
    commState.waitingForAcks = true;
    commState.acksReceived = 0;
    commState.currentCalNode = 0; // Current master is index 0
    commState.readingIndex = 0;

    // Setup calibration matrix
    commState.calibMatrix.numNodes = numActiveNodes + 1;
    commState.calibMatrix.nodeIds[0] = masterNodeId; // Master is first node

    // Add other active nodes to the matrix
    for (int i = 0; i < numActiveNodes; i++)
    {
        commState.calibMatrix.nodeIds[i + 1] = activeNodes[i];
    }

    // Initialize gain matrix to zeros
    for (int i = 0; i < commState.calibMatrix.numNodes; i++)
    {
        commState.calibMatrix.externalLight[i] = 0.0f;
        for (int j = 0; j < commState.calibMatrix.numNodes; j++)
        {
            commState.calibMatrix.gains[i][j] = 0.0f;
        }
    }

    // Explicitly disable feedback control during calibration
    controlState.feedbackControl = false;
    critical_section_exit(&commStateLock);

    // Broadcast calibration start to all nodes
    if (!sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_INIT, 0))
    {
        // Serial.println("Error sending calibration init command");
        return false;
    }

    // Set all LEDs to off (including our own)
    if (!sendControlCommand(CAN_ADDR_BROADCAST, 7, STATE_OFF))
    {
        // Serial.println("Error sending LED off command");
        return false;
    }

    // Apply locally as well
    critical_section_enter_blocking(&commStateLock);
    controlState.luminaireState = STATE_OFF;
    controlState.setpointLux = 0.0f;
    critical_section_exit(&commStateLock);

    setLEDDutyCycle(0.0f);

    // Also explicitly broadcast the feedback control off command
    sendControlCommand(CAN_ADDR_BROADCAST, 9, 0.0f);

    // Serial.println("Calibration initialization sent");
    return true;
}

/**
 * Acknowledge a calibration initialization request
 * Sent by nodes when they receive a calibration initialization command
 *
 * @param masterNodeId ID of the calibration master
 */
void acknowledgeCalibration(uint8_t masterNodeId)
{
    // Serial.print("Acknowledging calibration from master node ");
    Serial.println(masterNodeId);

    // Initialize as participant in calibration
    critical_section_enter_blocking(&commStateLock);
    commState.isCalibrationMaster = false;
    commState.calibrationInProgress = true;
    commState.waitingForAcks = false;
    commState.calLastStepTime = millis();

    // Explicitly disable feedback control during calibration
    controlState.feedbackControl = false;

    // Reset calibration matrix
    commState.calibMatrix.numNodes = 0;
    for (int i = 0; i < MAX_CALIB_NODES; i++)
    {
        commState.calibMatrix.externalLight[i] = 0.0f;
        for (int j = 0; j < MAX_CALIB_NODES; j++)
        {
            commState.calibMatrix.gains[i][j] = 0.0f;
        }
    }
    critical_section_exit(&commStateLock);

    // Turn off LED as required by the protocol
    setLEDDutyCycle(0.0f);

    // Send acknowledgment back to master
    sendControlCommand(masterNodeId, CAL_CMD_ACK, deviceConfig.nodeId);
}

/**
 * Process a calibration acknowledgment message
 * Updates the master's tracking of which nodes have acknowledged
 *
 * @param nodeId Node ID that sent the acknowledgment
 * @return true if all expected nodes have acknowledged
 */
bool processCalibrationAck(uint8_t nodeId)
{
    // Serial.print("Received calibration ACK from node ");
    // Serial.println(nodeId);

    critical_section_enter_blocking(&commStateLock);
    commState.acksReceived++;

    // Check if we have received all expected acknowledgments
    bool allAcksReceived = (commState.acksReceived >= commState.calibMatrix.numNodes - 1);
    critical_section_exit(&commStateLock);

    return allAcksReceived;
}

/**
 * Process and store a light reading from another node during calibration
 * Validates readings and requests retransmission if invalid
 *
 * @param nodeId Source node ID
 * @param reading Light reading value
 */
void processCalibrationReading(uint8_t nodeId, float reading)
{
    // Check for invalid readings
    if (reading <= 0.0f)
    {
        /*Serial.print("Invalid reading (");
        Serial.print(reading);
        Serial.print(") from node ");
        Serial.println(nodeId);*/

        // Request another reading from this node
        sendControlCommand(nodeId, CAL_CMD_SEND_READING, 1);
        return;
    }

    critical_section_enter_blocking(&commStateLock);

    // Find the index for this node ID
    int nodeIndex = -1;
    for (int i = 0; i < commState.calibMatrix.numNodes; i++)
    {
        if (commState.calibMatrix.nodeIds[i] == nodeId)
        {
            nodeIndex = i;
            break;
        }
    }

    // Store the reading if we found the node
    if (nodeIndex >= 0)
    {
        // Store the reading in the appropriate location
        commState.luxReadings[nodeIndex] = reading;
    }

    critical_section_exit(&commStateLock);
}

/**
 * Update calibration state machine
 * Called periodically from the main loop to advance the calibration process
 *
 * The state machine handles these states:
 * - Master node:
 *   0: Initialize and wait for acknowledgments
 *   1: Measure external light and LED gains
 *   2: Process measurements and decide next steps
 *   3: Transfer master role to next node
 *   4: Finalize calibration
 * - Participant node:
 *   Handles measurement requests and role transfers
 */
void updateCalibrationState()
{
    unsigned long currentTime = millis();

    // Check if calibration is active
    critical_section_enter_blocking(&commStateLock);
    bool calibrationActive = commState.calibrationInProgress;
    bool isMaster = commState.isCalibrationMaster;
    uint8_t currentStep = commState.calibrationStep;
    unsigned long lastStepTime = commState.calLastStepTime;
    critical_section_exit(&commStateLock);

    if (!calibrationActive)
    {
        return;
    }

    // Handle timeout for all states - abort if step takes too long
    if (currentTime - lastStepTime > CAL_TIMEOUT_MS * 2)
    {
        // Serial.println("Calibration step timeout - aborting");

        critical_section_enter_blocking(&commStateLock);
        commState.calibrationInProgress = false;
        commState.isCalibrationMaster = false;
        controlState.luminaireState = STATE_UNOCCUPIED;
        controlState.setpointLux = SETPOINT_UNOCCUPIED;
        controlState.feedbackControl = true;
        critical_section_exit(&commStateLock);

        // Return to normal operation
        setLEDDutyCycle(0.0f);
        return;
    }

    // Master node state machine
    if (isMaster)
    {
        switch (currentStep)
        {
        case 0: // Wait for acknowledgments from participating nodes
        {
            critical_section_enter_blocking(&commStateLock);
            bool waitingForAcks = commState.waitingForAcks;
            int acksReceived = commState.acksReceived;
            int expectedAcks = commState.calibMatrix.numNodes - 1;
            bool timeoutOccurred = (currentTime - lastStepTime > CAL_TIMEOUT_MS);
            critical_section_exit(&commStateLock);

            if (waitingForAcks && (acksReceived >= expectedAcks || timeoutOccurred))
            {


                if (timeoutOccurred && acksReceived < expectedAcks)
                {
                    // Serial.println("Warning: Not all nodes responded, continuing anyway");
                }

                // -- NEW CODE --
                // First, get baseline measurements with all LEDs OFF before starting column calibrations
                Serial.println("Getting baseline illuminance readings from all nodes...");

                // Ensure all LEDs are off
                setLEDDutyCycle(0.0f);
                sendControlCommand(CAN_ADDR_BROADCAST, 7, STATE_OFF);
                delay(1000); // Wait for LEDs to turn off

                // Request baseline readings from all nodes
                sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_SEND_READING, 0);

                // Take our own baseline measurement
                float baselineLux = 0.0;
                const int SAMPLES = 5;
                const int STABILIZE_TIME = 500;

                for (int i = 0; i < SAMPLES; i++)
                {
                    baselineLux += readLux();
                    delay(STABILIZE_TIME / SAMPLES);
                }
                baselineLux /= SAMPLES;

                // Wait for other nodes to respond
                delay(1500);

                // Store all baseline readings
                critical_section_enter_blocking(&commStateLock);
                commState.luxReadings[0] = baselineLux;
                sensorState.baselineIlluminance = baselineLux;

                // Store all baseline readings in the calibration matrix external light array
                for (int i = 0; i < commState.calibMatrix.numNodes; i++)
                {
                    commState.calibMatrix.externalLight[i] = commState.luxReadings[i];

                }

                // Move to measuring LED gain (first column)
                commState.waitingForAcks = false;
                commState.calibrationStep = 1;
                commState.calLastStepTime = currentTime;
                commState.currentCalNode = 0; // Start with our own node (master)
                critical_section_exit(&commStateLock);

                // Perform system calibration for the master node (column 0)

                calibrateColumn(0);

                // Update state to move to next column
                critical_section_enter_blocking(&commStateLock);
                commState.calibrationStep = 2;
                commState.calLastStepTime = millis();
                critical_section_exit(&commStateLock);
            }
            break;
        }

        case 1: // Measuring LED gain and external light for current column
            // This step is handled by calibrateColumn(), just move to step 2
            critical_section_enter_blocking(&commStateLock);
            commState.calibrationStep = 2;
            commState.calLastStepTime = currentTime;
            critical_section_exit(&commStateLock);
            break;

        case 2: // Process measurements and prepare for next column
        {
            critical_section_enter_blocking(&commStateLock);
            int currentNodeIdx = commState.currentCalNode;
            int totalNodes = commState.calibMatrix.numNodes;
            critical_section_exit(&commStateLock);

            // Move to next node (column)
            currentNodeIdx++;

            if (currentNodeIdx >= totalNodes)
            {
                // All columns have been calibrated, move to finalization
                Serial.println("All columns have been calibrated. Moving to finalization...");
                critical_section_enter_blocking(&commStateLock);
                commState.calibrationStep = 5; // Skip to final state
                commState.calLastStepTime = currentTime;
                critical_section_exit(&commStateLock);
            }
            else
            {
                // Prepare to calibrate next column
                critical_section_enter_blocking(&commStateLock);
                commState.currentCalNode = currentNodeIdx;
                commState.calibrationStep = 3; // Move to the column calibration state
                commState.calLastStepTime = currentTime;
                critical_section_exit(&commStateLock);


            }
            break;
        }

        case 3: // Calibrate next column in the gain matrix
        {
            critical_section_enter_blocking(&commStateLock);
            int currentNodeIdx = commState.currentCalNode;
            uint8_t targetNodeId = commState.calibMatrix.nodeIds[currentNodeIdx];
            critical_section_exit(&commStateLock);



            // Calibrate the specified column
            calibrateColumn(currentNodeIdx);

            // Move back to step 2 to prepare for the next column or finalization
            critical_section_enter_blocking(&commStateLock);
            commState.calibrationStep = 2;
            commState.calLastStepTime = currentTime;
            critical_section_exit(&commStateLock);
            break;
        }

        case 5: // Finalize calibration and broadcast results
        {
            // Display calibration matrix
            Serial.println("\nFinal Calibration Matrix:");

            critical_section_enter_blocking(&commStateLock);
            int numNodes = commState.calibMatrix.numNodes;

            // Print matrix header with node IDs
            for (int j = 0; j < numNodes; j++)
            {
                Serial.print("Node ");
                Serial.print(commState.calibMatrix.nodeIds[j]);
                Serial.print(" | ");
            }
            Serial.println();

            // Print separator line
            for (int j = 0; j < numNodes + 1; j++)
            {
                Serial.print("--------");
            }
            Serial.println();

            // Print matrix rows with labels
            for (int i = 0; i < numNodes; i++)
            {
                Serial.print("Node ");
                Serial.print(commState.calibMatrix.nodeIds[i]);
                Serial.print(" | ");

                for (int j = 0; j < numNodes; j++)
                {
                    Serial.print(commState.calibMatrix.gains[i][j], 3);
                    Serial.print(" | ");
                }
                Serial.println();
            }

            // Print separator line
            for (int j = 0; j < numNodes + 1; j++)
            {
                Serial.print("--------");
            }
            Serial.println();

            // Copy our node ID before releasing lock
            uint8_t ourNodeId = deviceConfig.nodeId;
            critical_section_exit(&commStateLock);

            // Broadcast completion to all nodes
            // Serial.println("Broadcasting final calibration matrix...");

            // First broadcast calibration complete command
            sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_COMPLETE, 0);

            Serial.println("Broadcasting node ID mappings...");
            critical_section_enter_blocking(&commStateLock);
            int totalNodes = commState.calibMatrix.numNodes;
            critical_section_exit(&commStateLock);

            // First, resend the number of nodes
            float sizeMessage = 990000.0f + totalNodes;
            sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, sizeMessage);
            delay(100);

            // Then send each node ID mapping again
            for (int i = 0; i < totalNodes; i++) {
                critical_section_enter_blocking(&commStateLock);
                uint8_t nodeId = commState.calibMatrix.nodeIds[i];
                critical_section_exit(&commStateLock);
                
                // Send node ID mapping - format: 980000 + index*1000 + nodeId
                float nodeIdMessage = 980000.0f + i * 1000.0f + nodeId;
                
                // Send mapping 3 times with small delay to ensure reception
                for (int retry = 0; retry < 3; retry++) {
                    sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, nodeIdMessage);
                    delay(50);
                }
                // Allow some time between nodes to prevent message collision
                delay(100);
            }

            // Add additional delay to ensure processing
            delay(300);

            // Now broadcast the entire gain matrix to all nodes
            Serial.println("Broadcasting gain values to all nodes...");

            // Send each gain value in the matrix to all nodes
            for (int col = 0; col < totalNodes; col++)
            {
                for (int row = 0; row < totalNodes; row++)
                {
                    critical_section_enter_blocking(&commStateLock);
                    float gain = commState.calibMatrix.gains[row][col];
                    critical_section_exit(&commStateLock);

                    // Format: col*10000 + row*1000 + gain
                    float packedValue = col * 10000.0f + row * 1000.0f + gain;

                    // Send gain value to all nodes
                    sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, packedValue);
                    delay(30); // Small delay to prevent flooding
                }
            }

           // Broadcast matrix size and structure information
            Serial.println("Broadcasting matrix structure information...");

            // First, send the number of nodes with special code - increase delay for this critical message
            sizeMessage = 990000.0f + totalNodes;
            sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, sizeMessage);
            delay(200); // Longer delay for this important message

            // Then send each node ID with a different special code - send multiple times for reliability
            for (int i = 0; i < totalNodes; i++) {
                critical_section_enter_blocking(&commStateLock);
                uint8_t nodeId = commState.calibMatrix.nodeIds[i];
                critical_section_exit(&commStateLock);
                
                // Format: 980000 + index*1000 + nodeId
                float nodeIdMessage = 980000.0f + i * 1000.0f + nodeId;
                
                // Send each node ID message multiple times to ensure reception
                for (int retry = 0; retry < 3; retry++) {
                    sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, nodeIdMessage);
                    delay(50);
                }
                
                // Add extra delay between nodes to prevent message congestion
                delay(100);
            }

            // Wait longer for nodes to process the structure
            delay(500);

            // Now broadcast all gain values to ensure all nodes have a complete matrix
            Serial.println("Broadcasting all gain values...");
            for (int col = 0; col < totalNodes; col++) {
                for (int row = 0; row < totalNodes; row++) {
                    critical_section_enter_blocking(&commStateLock);
                    float gain = commState.calibMatrix.gains[row][col];
                    critical_section_exit(&commStateLock);
                    
                    // Format: col*10000 + row*1000 + gain
                    float packedValue = col * 10000.0f + row * 1000.0f + gain;
                    
                    // Send gain value to all nodes
                    sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, packedValue);
                    delay(30); // Small delay to prevent flooding
                }
            }

            // Send final confirmation message
            sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_COMPLETE, 1.0f);

            // Wait for nodes to process the complete matrix
            delay(500);

            // Also send a specific command to re-enable feedback control
            sendControlCommand(CAN_ADDR_BROADCAST, 9, 1.0f);

            // Return to normal operation
            critical_section_enter_blocking(&commStateLock);
            controlState.luminaireState = STATE_UNOCCUPIED;
            controlState.setpointLux = SETPOINT_UNOCCUPIED;
            controlState.feedbackControl = true;
            commState.calibrationInProgress = false;
            critical_section_exit(&commStateLock);

            Serial.println("Calibration process completed successfully");
            break;
        }
        }
    }
    else // Participant node state machine
    {
        // Participant nodes primarily respond to commands from the master
        // Here we handle any periodic tasks or state transitions

        // Check if we should have received commands by now
        if (currentTime - lastStepTime > CAL_TIMEOUT_MS)
        {
            // If we haven't received any commands for a while, assume something went wrong
            // Serial.println("Warning: No calibration commands received for too long");

            // Send a heartbeat to make sure we're still visible on the network
            sendHeartbeat();

            // Update last step time to prevent multiple warnings
            critical_section_enter_blocking(&commStateLock);
            commState.calLastStepTime = currentTime;
            critical_section_exit(&commStateLock);
        }
    }
}

/**
 * Calibrate a specific column of the gain matrix
 * Each column corresponds to one node's LED effect on all nodes
 *
 * @param columnIndex The column index to calibrate (node index in the matrix)
 * @return Average gain value
 */
float calibrateColumn(int columnIndex)
{
    const int SAMPLES = 5;               // Number of measurements to average
    const int STABILIZE_TIME = 500;      // Wait time between measurements in ms
    const int LED_RESPONSE_TIME = 10000; // Wait time for LDR to respond to LED changes
    const int MAX_RETRIES = 3;           // Maximum retries for readings


    // Arrays to store LED on readings from all nodes
    float onReadings[MAX_CALIB_NODES] = {0};

    // Get relevant node information
    critical_section_enter_blocking(&commStateLock);
    uint8_t targetNodeId = commState.calibMatrix.nodeIds[columnIndex];
    uint8_t ourNodeId = deviceConfig.nodeId;
    bool isSelfCalibration = (targetNodeId == ourNodeId);
    int numNodes = commState.calibMatrix.numNodes;

    // Retrieve stored baseline illuminance values
    float baselineLux = sensorState.baselineIlluminance;
    float offReadings[MAX_CALIB_NODES];

    // Copy the stored baseline readings
    for (int i = 0; i < numNodes; i++)
    {
        offReadings[i] = commState.calibMatrix.externalLight[i];
    }

    // Reset readings tracking variables to ensure we get fresh readings
    for (int i = 0; i < numNodes; i++)
    {
        commState.luxReadings[i] = -1.0f; // Mark as invalid to ensure we get new readings
    }
    critical_section_exit(&commStateLock);

    // ---- ENSURE ALL LEDS ARE OFF FIRST ----
    // First ensure all LEDs are turned off before starting
    sendControlCommand(CAN_ADDR_BROADCAST, 7, STATE_OFF);
    delay(1000); // Wait for LEDs to turn off completely

    // ---- MEASURE WITH TARGET NODE LED ON ----
    // If we're calibrating our own node, turn on our LED
    if (isSelfCalibration)
    {
        setLEDDutyCycle(1.0f);
        // Serial.println("Setting OUR LED to full brightness");
    }
    else
    {
        // Send command to target node to turn on its LED

        // Use the exact same command format as the serial 'u' command
        // This is control type 4 for direct duty cycle setting
        for (int i = 0; i < 5; i++) // Send multiple times to ensure receipt
        {
            sendControlCommand(targetNodeId, 4, 1.0f); // Set duty cycle to 100%
            delay(200);                                // Longer delay between attempts
        }

        // Additionally, explicitly turn OFF all other LEDs
        for (int i = 0; i < numNodes; i++)
        {
            uint8_t nodeId = 0;
            critical_section_enter_blocking(&commStateLock);
            nodeId = commState.calibMatrix.nodeIds[i];
            critical_section_exit(&commStateLock);

            if (nodeId != targetNodeId && nodeId != 0)
            {
                sendControlCommand(nodeId, 4, 0.0f); // Set duty cycle to 0%
                delay(50);
            }
        }
    }

    // Allow time for LED to reach full brightness and LDRs to respond

    // During the wait period, send reinforcement commands to ensure LED stays on
    for (int i = 0; i < 3; i++)
    {
        delay(LED_RESPONSE_TIME / 4); // Divide the wait time into quarters

        if (!isSelfCalibration)
        {
            // Send the command again to ensure it stays on
            sendControlCommand(targetNodeId, 4, 1.0f);
            /*Serial.print("Resending LED ON command to node ");
            Serial.println(targetNodeId);*/
        }
    }

    delay(LED_RESPONSE_TIME / 4); // Final quarter of wait time

    // Request readings from all nodes with this LED on

    // Send request multiple times to ensure reception
    for (int attempt = 0; attempt < 3; attempt++)
    {
        sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_SEND_READING, 1);
        delay(100);
    }

    // Take multiple measurements for our own node and average
    float onLux = 0.0;
    for (int i = 0; i < SAMPLES; i++)
    {
        onLux += readLux();
        delay(STABILIZE_TIME / SAMPLES);
    }
    onLux /= SAMPLES;

    // Store our reading with the LED on
    critical_section_enter_blocking(&commStateLock);
    commState.luxReadings[0] = onLux;
    critical_section_exit(&commStateLock);

    // Wait for all nodes to respond with their readings
    // Serial.println("Waiting for responses from other nodes...");

    // Check if we have readings from all nodes with retries
    bool allReadingsReceived = false;
    int retryCount = 0;

    while (!allReadingsReceived && retryCount < MAX_RETRIES)
    {
        // Wait for responses
        delay(1000);

        // For remote node calibration, send the LED ON command again
        // to ensure it stays on during the entire measurement period
        if (!isSelfCalibration)
        {
            sendControlCommand(targetNodeId, 4, 1.0f);

        }

        // Check if all readings are received
        critical_section_enter_blocking(&commStateLock);
        bool missingReadings = false;
        for (int i = 0; i < numNodes; i++)
        {
            if (commState.luxReadings[i] <= 0.0f)
            {
                missingReadings = true;

            }
        }
        allReadingsReceived = !missingReadings;
        critical_section_exit(&commStateLock);

        // If not all readings received, retry
        if (!allReadingsReceived)
        {
            retryCount++;
            // Ensure target LED is still ON
            if (!isSelfCalibration)
            {
                sendControlCommand(targetNodeId, 4, 1.0f);
            }

            sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_SEND_READING, 1);
            delay(1000);
        }
    }

    // Copy all readings with LED on
    critical_section_enter_blocking(&commStateLock);
    for (int i = 0; i < numNodes; i++)
    {
        onReadings[i] = commState.luxReadings[i];
        // If we still have invalid readings, use baseline to avoid errors
        if (onReadings[i] <= 0.0f)
        {

            onReadings[i] = offReadings[i]; // Use baseline as fallback
        }
    }
    critical_section_exit(&commStateLock);

    // ---- CALCULATE GAINS FOR THIS COLUMN ----
    // Calculate the gains for this column (effect of this node's LED on all nodes)
    float averageGain = 0.0f;
    int validGainCount = 0;

    critical_section_enter_blocking(&commStateLock);
    for (int i = 0; i < numNodes; i++)
    {
        // Calculate gain: difference between LED on and LED off readings
        float gain = onReadings[i] - offReadings[i];

        // Ensure gain is not negative (could happen with measurement noise)
        if (gain < 0.0f)
        {
            gain = 0.0f;
        }

        // Store in calibration matrix [row=receiver node][col=LED node]
        commState.calibMatrix.gains[i][columnIndex] = gain;


        // Add to average if it's a valid reading
        if (gain >= 0.0f)
        {
            averageGain += gain;
            validGainCount++;
        }
    }
    critical_section_exit(&commStateLock);

    // Calculate average gain
    if (validGainCount > 0)
    {
        averageGain /= validGainCount;
    }

    // If this is our own LED being calibrated, store the self-gain
    if (isSelfCalibration)
    {
        deviceConfig.ledGain = onLux - baselineLux;
    }

    // ---- BROADCAST GAIN VALUES TO ALL NODES ----

    // First broadcast to all nodes
    critical_section_enter_blocking(&commStateLock);
    for (int i = 0; i < numNodes; i++)
    {
        float gain = commState.calibMatrix.gains[i][columnIndex];
        critical_section_exit(&commStateLock);

        // Format: columnIndex * 1000 + rowIndex * 100 + gain
        float packedValue = columnIndex * 1000.0f + i * 100.0f + gain;
        sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, packedValue);
        delay(50); // Small delay to prevent flooding

        critical_section_enter_blocking(&commStateLock);
    }
    critical_section_exit(&commStateLock);

    // Turn off all LEDs
    if (isSelfCalibration)
    {
        setLEDDutyCycle(0.0f);
    }
    else
    {
        sendControlCommand(targetNodeId, 4, 0.0f);
    }
    sendControlCommand(CAN_ADDR_BROADCAST, 7, STATE_OFF);

    critical_section_enter_blocking(&commStateLock);
    commState.calLastStepTime = millis();
    critical_section_exit(&commStateLock);

    return averageGain;
}

/**
 * Handle incoming calibration control message
 * Processes various calibration commands and coordinates the distributed calibration process
 *
 * @param sourceNodeId ID of the node that sent the message
 * @param controlType Type of calibration command
 * @param value Command parameter value
 */
void handleCalibrationMessage(uint8_t sourceNodeId, uint8_t controlType, float value){
    switch (controlType)
    {
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
        // Request for illuminance reading
        // value: 0=external light (baseline), 1=with LED on
        float currentLux = readLux();

        // Validate the reading - retry if invalid
        int retries = 0;
        while (currentLux <= 0.0f && retries < 5)
        {

            delay(100);
            currentLux = readLux();
            retries++;
        }

        if (currentLux <= 0.0f)
        {
            //  Use a small positive value instead of zero/negative to avoid division issues
            currentLux = 0.1f;
        }


        // Store reading in our calibration matrix
        critical_section_enter_blocking(&commStateLock);
        if (value == 0)
        {
            // Store external light reading as baseline
            sensorState.baselineIlluminance = currentLux;

            // Store this in our local copy of the matrix too for future use
            for (int i = 0; i < commState.calibMatrix.numNodes; i++)
            {
                if (commState.calibMatrix.nodeIds[i] == deviceConfig.nodeId)
                {
                    commState.calibMatrix.externalLight[i] = currentLux;
                    break;
                }
            }
        }
        critical_section_exit(&commStateLock);

        // Send reading back to master - retry multiple times to ensure delivery
        bool sent = false;
        for (int i = 0; i < 3 && !sent; i++)
        {
            sent = sendSensorReading(sourceNodeId, 0, currentLux);
            if (!sent)
            {
                delay(50);
            }
        }
    }
    break;

    case CAL_CMD_NEXT_NODE:
    {
        // Update calibration matrix with received gain values
        critical_section_enter_blocking(&commStateLock);
        if (commState.calibrationInProgress)
        {
            // Check if this is the matrix size message (990xxx)
            if ((int)(value / 1000.0f) == 990) {
                // Extract matrix size from message
                int numNodes = (int)(value - 990000.0f);
                commState.calibMatrix.numNodes = numNodes;
                Serial.print("Setting matrix size to ");
                Serial.println(numNodes);
            }
            // Check if this is a node ID message (980xxx)
            else if ((int)(value / 1000.0f) == 980) {
                // Extract node index and ID from message
                float remainder = value - 980000.0f;
                int nodeIndex = (int)(remainder / 1000.0f);
                int nodeId = (int)(remainder) % 1000;
                
                // Store the node ID if indices are valid and nodeId is not zero
                if (nodeIndex >= 0 && nodeIndex < MAX_CALIB_NODES && nodeId > 0) {
                    // Add additional debug output
                    Serial.print("Received node ID mapping: index ");
                    Serial.print(nodeIndex);
                    Serial.print(" -> node ");
                    Serial.println(nodeId);
                    
                    // Store the node ID
                    commState.calibMatrix.nodeIds[nodeIndex] = nodeId;
                    
                    // If this is our node, track our position in the matrix
                    uint8_t ourNodeId = deviceConfig.nodeId;
                    if (nodeId == ourNodeId) {
                        Serial.print("Found our node ID at index ");
                        Serial.println(nodeIndex);
                        
                        // Save our position for self-gain calculation when all data is received
                        commState.ourNodeIndex = nodeIndex;
                    }
                }
                else {
                    // Log invalid node ID information
                    Serial.print("Warning: Invalid node ID mapping (index=");
                    Serial.print(nodeIndex);
                    Serial.print(", id=");
                    Serial.print(nodeId);
                    Serial.println(")");
                }
            }
            // Process gain values (10000+)
            else if (value >= 10000.0f) {
                // Format: col*10000 + row*1000 + gain
                int col = (int)(value / 10000.0f);
                int row = (int)((value - col * 10000.0f) / 1000.0f);
                float gain = value - col * 10000.0f - row * 1000.0f;
                
                // Store the gain if indices are valid
                if (col >= 0 && col < MAX_CALIB_NODES && row >= 0 && row < MAX_CALIB_NODES) {
                    commState.calibMatrix.gains[row][col] = gain;
                    
                    // Check if this is our self-gain (diagonal element for our node)
                    if (row == col) {
                        uint8_t nodeId = commState.calibMatrix.nodeIds[row];
                        uint8_t ourNodeId = deviceConfig.nodeId;
                        
                        if (nodeId == ourNodeId) {
                            deviceConfig.ledGain = gain;
                            Serial.print("Set self-gain to ");
                            Serial.println(gain);
                        }
                    }
                }
                else {
                    Serial.print("Warning: Invalid gain matrix indices (row=");
                    Serial.print(row);
                    Serial.print(", col=");
                    Serial.print(col);
                    Serial.println(")");
                }
            }
            // Handle legacy format
            else if (value > 100.0f) {
                // Old format: columnIndex * 1000 + rowIndex * 100 + gain
                int columnIndex = (int)(value / 1000.0f);
                int rowIndex = (int)((value - columnIndex * 1000.0f) / 100.0f);
                float gain = value - columnIndex * 1000.0f - rowIndex * 100.0f;
                
                if (columnIndex >= 0 && columnIndex < MAX_CALIB_NODES &&
                    rowIndex >= 0 && rowIndex < MAX_CALIB_NODES)
                {
                    commState.calibMatrix.gains[rowIndex][columnIndex] = gain;
                }
            }
            else {
                // Simple gain value (original format)
                // Find the index for the source node
                int sourceIndex = -1;
                for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
                    if (commState.calibMatrix.nodeIds[i] == sourceNodeId) {
                        sourceIndex = i;
                        break;
                    }
                }
    
                if (sourceIndex >= 0) {
                    // Update the master's self-gain
                    commState.calibMatrix.gains[sourceIndex][sourceIndex] = value;
                }
            }
        }
        critical_section_exit(&commStateLock);
    }
    break;

    case CAL_CMD_COMPLETE:
    {
        critical_section_enter_blocking(&commStateLock);
        
        // Only respond to the final confirmation message (value == 1.0)
        if (value >= 1.0f)
        {
            commState.calibrationInProgress = false;
            controlState.setpointLux = SETPOINT_UNOCCUPIED;
            controlState.luminaireState = STATE_UNOCCUPIED;
            controlState.feedbackControl = true;
            controlState.systemReady = true;

            // Find our node in the matrix and ensure our self-gain is set
            uint8_t ourNodeId = deviceConfig.nodeId;
            bool foundOurNode = false;

            // If any node ID is 0, try to fix it
            for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
                if (commState.calibMatrix.nodeIds[i] == 0) {
                    // Node 0 is invalid - try to infer the node ID from its position
                    if (i == 1 && commState.calibMatrix.numNodes == 3) {
                        // In a 3-node system, if we have node IDs [33, 0, 0], this is likely node 40
                        commState.calibMatrix.nodeIds[i] = 40;
                    } else if (i == 2 && commState.calibMatrix.numNodes == 3) {
                        // In a 3-node system, if we have node IDs [33, 40, 0], this is likely node 52
                        commState.calibMatrix.nodeIds[i] = 52;
                    }
                }
            }

            // Search through the node IDs to find our position
            for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
                if (commState.calibMatrix.nodeIds[i] == ourNodeId) {
                    // Set our self-gain from the matrix
                    deviceConfig.ledGain = commState.calibMatrix.gains[i][i];
                    foundOurNode = true;
                    
                    // Ensure it's not zero (would cause control issues)
                    if (deviceConfig.ledGain <= 0.0f) {
                        // If diagonal is zero, use maximum value from this row
                        float maxGain = 0.0f;
                        for (int j = 0; j < commState.calibMatrix.numNodes; j++) {
                            if (commState.calibMatrix.gains[i][j] > maxGain) {
                                maxGain = commState.calibMatrix.gains[i][j];
                            }
                        }
                        
                        // Use maximum as fallback or a reasonable default
                        deviceConfig.ledGain = maxGain > 0.0f ? maxGain : 10.0f;
                    }
                    
                    Serial.print("Final self-gain set to: ");
                    Serial.println(deviceConfig.ledGain);
                    break;
                }
            }
            
            if (!foundOurNode) {
                Serial.println("Warning: Could not find our node ID in the matrix!");
            }

            // Print the final calibration matrix
            Serial.println("\nFinal Calibration Matrix:");
            
            // Print matrix header with node IDs
            for (int j = 0; j < commState.calibMatrix.numNodes; j++)
            {
                Serial.print("Node ");
                Serial.print(commState.calibMatrix.nodeIds[j]);
                Serial.print(" | ");
            }
            Serial.println();

            // Print separator line
            for (int j = 0; j < commState.calibMatrix.numNodes + 1; j++)
            {
                Serial.print("--------");
            }
            Serial.println();

            // Print matrix rows with labels
            for (int i = 0; i < commState.calibMatrix.numNodes; i++)
            {
                Serial.print("Node ");
                Serial.print(commState.calibMatrix.nodeIds[i]);
                Serial.print(" | ");

                for (int j = 0; j < commState.calibMatrix.numNodes; j++)
                {
                    Serial.print(commState.calibMatrix.gains[i][j], 3);
                    Serial.print(" | ");
                }
                Serial.println();
            }
            
            // Print separator line
            for (int j = 0; j < commState.calibMatrix.numNodes + 1; j++)
            {
                Serial.print("--------");
            }
            
            
            // If this was part of the wake-up sequence, update system state
            if (!controlState.systemReady && !controlState.standbyMode)
            {
                controlState.systemReady = true;
                controlState.feedbackControl = true;

                // Move to occupied state if not already set
                if (controlState.luminaireState != STATE_OCCUPIED)
                {
                    critical_section_exit(&commStateLock);
                    changeState(STATE_OCCUPIED);
                    critical_section_enter_blocking(&commStateLock);
                }
            }
        }

        critical_section_exit(&commStateLock);

        // Reset LED to off state
        setLEDDutyCycle(0.0f);
    }
    break;

    default:
        Serial.print("Unknown calibration command type: ");
        Serial.println(controlType);
        break;
    }
}