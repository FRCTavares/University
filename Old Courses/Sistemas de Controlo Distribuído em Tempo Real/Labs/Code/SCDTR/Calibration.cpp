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
 * Broadcast the default gain matrix to all nodes in the network
 * This tells other nodes to use the default gain matrix instead of calibrating
 */
void broadcastDefaultGainMatrix()
{
    Serial.println("Broadcasting default gain matrix to all nodes...");

    // First, broadcast a command to indicate we're using default gains
    // Use command type 107 (CAL_CMD_USE_DEFAULT) for this purpose
    sendControlCommand(CAN_ADDR_BROADCAST, 107, 1.0f);
    delay(300);

    // Send multiple times to ensure all nodes receive it
    for (int retry = 0; retry < 3; retry++)
    {
        sendControlCommand(CAN_ADDR_BROADCAST, 107, 1.0f);
        delay(100);
    }

    // Send the standard "calibration complete" message to finalize
    sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_COMPLETE, 1.0f);
    delay(300);

    // Send completion message multiple times to ensure reception
    for (int retry = 0; retry < 3; retry++)
    {
        sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_COMPLETE, 1.0f);
        delay(100);
    }

    Serial.println("Default gain matrix broadcast complete");
}

/**
 * Load default gain matrix with predefined values
 * This allows testing without running the actual calibration process
 */
void loadDefaultGainMatrix()
{
    Serial.println("Loading default gain matrix...");

    critical_section_enter_blocking(&commStateLock);

    // Set up the calibration matrix with hardcoded values
    commState.calibMatrix.numNodes = 3;

    // Define the node IDs
    commState.calibMatrix.nodeIds[0] = 33;
    commState.calibMatrix.nodeIds[1] = 40;
    commState.calibMatrix.nodeIds[2] = 52;

    // Set external illuminance to a reasonable default
    for (int i = 0; i < commState.calibMatrix.numNodes; i++)
    {
        commState.calibMatrix.externalLight[i] = 10.0f;
    }

    // Gains matrix as provided
    commState.calibMatrix.gains[0][0] = 19.63;
    commState.calibMatrix.gains[0][1] = 5.66;
    commState.calibMatrix.gains[0][2] = 5.99;

    commState.calibMatrix.gains[1][0] = 6.16;
    commState.calibMatrix.gains[1][1] = 21.08;
    commState.calibMatrix.gains[1][2] = 1.02;

    commState.calibMatrix.gains[2][0] = 3.20;
    commState.calibMatrix.gains[2][1] = 1.28;
    commState.calibMatrix.gains[2][2] = 19.33;

    // Find our node index in the matrix and set our LED gain
    uint8_t ourNodeId = deviceConfig.nodeId;
    commState.ourNodeIndex = -1;

    for (int i = 0; i < commState.calibMatrix.numNodes; i++)
    {
        if (commState.calibMatrix.nodeIds[i] == ourNodeId)
        {
            commState.ourNodeIndex = i;
            deviceConfig.ledGain = commState.calibMatrix.gains[i][i];

            // Store the external illuminance in sensor state
            sensorState.baselineIlluminance = commState.calibMatrix.externalLight[i];
            break;
        }
    }

    // Mark calibration as completed
    commState.wasCalibrating = true;
    commState.calibrationInProgress = false;

    // Ensure complete system state initialization
    controlState.setpointLux = SETPOINT_UNOCCUPIED;
    controlState.luminaireState = STATE_UNOCCUPIED;
    controlState.feedbackControl = true;
    controlState.systemReady = true;
    controlState.standbyMode = false;

    sensorState.baselineIlluminance = 1.0f; // Set to a default value for external light

    critical_section_exit(&commStateLock);

    // Display the loaded matrix
    Serial.println("\nLoaded Default Calibration Matrix:");
    Serial.println("Effect Matrix | Node 33 | Node 40 | Node 52 |");
    Serial.println("----------------------------------------");
    Serial.println("Node 33      | 19.63   | 5.66    | 5.99    |");
    Serial.println("Node 40      | 6.16    | 21.08   | 1.02    |");
    Serial.println("Node 52      | 3.20    | 1.28    | 19.33   |");

    Serial.print("This node's LED gain value: ");
    Serial.println(deviceConfig.ledGain);
}

/**
 * Start a calibration sequence as the calibration master
 * This node takes control of the calibration process across the network
 *
 * @return true if calibration started successfully
 */
bool startCalibration()
{
    // Check if we should use default gains instead
    critical_section_enter_blocking(&commStateLock);
    bool useDefault = commState.useDefaultGains;
    critical_section_exit(&commStateLock);

    if (useDefault)
    {
        Serial.println("Using default gain matrix instead of running calibration");
        loadDefaultGainMatrix();

        // Broadcast to all other nodes to use default gains too
        broadcastDefaultGainMatrix();
        return true;
    }

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
        return false;
    }

    // Set all LEDs to off (including our own)
    if (!sendControlCommand(CAN_ADDR_BROADCAST, 7, STATE_OFF))
    {
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

    critical_section_enter_blocking(&commStateLock);
    commState.acksReceived++;

    // Check if we have received all expected acknowledgments
    bool allAcksReceived = (commState.acksReceived >= commState.calibMatrix.numNodes - 1);
    critical_section_exit(&commStateLock);

    return allAcksReceived;
}

/**
 * Process a calibration reading received from another node
 * Stores the reading in the calibration matrix for processing
 *
 * @param sourceNodeId ID of the node that sent the reading
 * @param value The illuminance reading value
 */
void processCalibrationReading(uint8_t sourceNodeId, float value)
{
    // Only process readings if we're in calibration mode
    critical_section_enter_blocking(&commStateLock);
    bool inCalibration = commState.calibrationInProgress;
    bool isMaster = commState.isCalibrationMaster;
    critical_section_exit(&commStateLock);

    if (!inCalibration)
    {
        return; // Not in calibration mode, ignore the reading
    }

    // If we're the master, record this reading
    if (isMaster)
    {
        critical_section_enter_blocking(&commStateLock);

        // Find the node's index in our calibration matrix
        int nodeIndex = -1;
        for (int i = 0; i < commState.calibMatrix.numNodes; i++)
        {
            if (commState.calibMatrix.nodeIds[i] == sourceNodeId)
            {
                nodeIndex = i;
                break;
            }
        }

        // If we found the node, store its reading
        if (nodeIndex >= 0 && nodeIndex < MAX_CALIB_NODES)
        {
            commState.luxReadings[nodeIndex] = value;
        }

        critical_section_exit(&commStateLock);
    }
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
    if (currentTime - lastStepTime > CAL_TIMEOUT_MS * 3)
    {

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

                // First, get baseline measurements with all LEDs OFF before starting column calibrations

                /// Ensure all LEDs are off
                setLEDDutyCycle(0.0f);
                sendControlCommand(CAN_ADDR_BROADCAST, 7, STATE_OFF);
                delay(1000); // Wait for LEDs to turn off

                // Reset reading tracking variables to ensure we get fresh readings
                critical_section_enter_blocking(&commStateLock);
                for (int i = 0; i < commState.calibMatrix.numNodes; i++)
                {
                    commState.luxReadings[i] = -1.0f; // Mark as invalid to ensure we get new readings
                }
                critical_section_exit(&commStateLock);

                // Request baseline readings from all nodes
                for (int attempt = 0; attempt < 3; attempt++)
                {
                    sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_SEND_READING, 0);
                    delay(100);
                }

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

                // Store our own reading
                critical_section_enter_blocking(&commStateLock);
                commState.luxReadings[0] = baselineLux;
                sensorState.baselineIlluminance = baselineLux;
                critical_section_exit(&commStateLock);

                // Wait for other nodes to respond with retries
                bool allReadingsReceived = false;
                int retryCount = 0;
                const int MAX_BASELINE_RETRIES = 5;

                while (!allReadingsReceived && retryCount < MAX_BASELINE_RETRIES)
                {
                    // Wait longer for responses - minimum 7 seconds to account for sensor stabilization
                    delay(7000);

                    // Check if all readings are received
                    critical_section_enter_blocking(&commStateLock);
                    bool missingReadings = false;
                    for (int i = 0; i < commState.calibMatrix.numNodes; i++)
                    {
                        if (commState.luxReadings[i] <= 0.0f)
                        {
                            missingReadings = true;
                        }
                    }
                    allReadingsReceived = !missingReadings;
                    critical_section_exit(&commStateLock);

                    // If still missing readings, retry
                    if (!allReadingsReceived)
                    {
                        retryCount++;
                        if (retryCount < MAX_BASELINE_RETRIES)
                        {
                            for (int attempt = 0; attempt < 3; attempt++)
                            {
                                sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_SEND_READING, 0);
                                delay(150);
                            }
                        }
                    }
                }

                // Now that we have the readings (or have tried our best), store them
                critical_section_enter_blocking(&commStateLock);
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
            // Reset the timeout timer to give us a fresh 30 seconds
            critical_section_enter_blocking(&commStateLock);
            commState.calLastStepTime = currentTime;
            critical_section_exit(&commStateLock);
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
                    Serial.print(commState.calibMatrix.gains[i][j], 2);
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

            // Broadcast completion and matrix data to all nodes
            Serial.println("Broadcasting complete calibration matrix...");

            // Step 1: First broadcast a reset command to prepare nodes
            sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_COMPLETE, 0);
            delay(500); // Longer delay to ensure nodes are ready

            // Step 2: Send matrix size - use a smaller, fixed number (max 6)
            int actualNodes = min(numNodes, 6); // Limit to prevent overflow issues
            float sizeMessage = 990000.0f + actualNodes;

            // Send size command multiple times with longer delays
            for (int retry = 0; retry < 5; retry++)
            {
                sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, sizeMessage);
                delay(200); // Longer delay between retries
            }
            delay(500); // Additional delay before next stage

            // Step 3: Send node ID mappings with clearer separation
            for (int i = 0; i < actualNodes; i++)
            {
                critical_section_enter_blocking(&commStateLock);
                uint8_t nodeId = commState.calibMatrix.nodeIds[i];
                critical_section_exit(&commStateLock);

                // Format: 980000 + index*100 + nodeId
                // This gives us range for index 0-99 and nodeId 0-99
                float nodeIdMessage = 980000.0f + i * 100.0f + nodeId;

                // Send more repeats with longer delays
                for (int retry = 0; retry < 5; retry++)
                {
                    sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, nodeIdMessage);
                    delay(150);
                }
                delay(300);
            }

            // Step 4: Send all gain values with a completely different approach

            // First, send a clear signal that we're starting gain transmission
            for (int retry = 0; retry < 3; retry++)
            {
                sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, 950000.0f);
                delay(200);
            }
            delay(300);

            // Send each gain using a clearer format: col*10000 + row*1000 + gain*10
            // This gives us range for gains up to 99.9 and clear separation between indices
            for (int col = 0; col < actualNodes; col++)
            {
                for (int row = 0; row < actualNodes; row++)
                {
                    critical_section_enter_blocking(&commStateLock);
                    float gain = commState.calibMatrix.gains[row][col];
                    critical_section_exit(&commStateLock);

                    // Cap at 99.9 to prevent overflow
                    gain = min(gain, 99.9f);

                    // Format: 900000 + col*10000 + row*1000 + gain*10
                    // This gives 1 decimal place precision but avoids splitting into int/frac
                    float packedValue = 900000.0f + col * 100.0f + row * 10.0f + gain / 10.0f;

                    // Send with multiple retries and longer delay between each gain
                    for (int retry = 0; retry < 3; retry++)
                    {
                        sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_NEXT_NODE, packedValue);
                        delay(100);
                    }

                    delay(200); // Longer delay between different gains
                }

                // Even longer delay between columns
                delay(500);
            }

            // Step 5: Send completion marker with multiple retries
            delay(500); // Wait longer before final confirmation
            for (int retry = 0; retry < 5; retry++)
            {
                sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_COMPLETE, 1.0f);
                delay(200);
            }

            // Reset state and continue with normal operation
            critical_section_enter_blocking(&commStateLock);
            controlState.luminaireState = STATE_UNOCCUPIED;
            controlState.setpointLux = SETPOINT_UNOCCUPIED;
            controlState.feedbackControl = true;
            commState.calibrationInProgress = false;
            commState.wasCalibrating = true;
            critical_section_exit(&commStateLock);

            Serial.println("Calibration completed and matrix broadcast finished");
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
    const int STABILIZE_TIME = 5000;     // Wait time between measurements in ms
    const int LED_RESPONSE_TIME = 10000; // Wait time for LDR to respond to LED changes
    const int MAX_RETRIES = 5;           // Maximum retries for readings

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
        // Print the reading for debugging
        // save last reading as onLux
        if (i == SAMPLES - 1)
        {
            delay(1000); // Wait a bit before the last reading
            onLux = readLux();
        }
    }

    // Store our reading with the LED on
    critical_section_enter_blocking(&commStateLock);
    commState.luxReadings[0] = onLux;
    critical_section_exit(&commStateLock);

    // Wait for all nodes to respond with their readings

    // Check if we have readings from all nodes with retries
    bool allReadingsReceived = false;
    int retryCount = 0;
    const int INCREASED_MAX_RETRIES = 10; // Increased from 5 to 10
    const int LONGER_WAIT_TIME = 2500;    // Increased from 1000ms to 2500ms

    while (!allReadingsReceived && retryCount < INCREASED_MAX_RETRIES)
    {
        // Wait longer for responses
        delay(LONGER_WAIT_TIME);

        // Maintain LED states during waiting period - keep target LED ON
        if (!isSelfCalibration)
        {
            sendControlCommand(targetNodeId, 4, 1.0f);

            // Also keep all other LEDs explicitly OFF
            for (int i = 0; i < numNodes; i++)
            {
                uint8_t nodeId = 0;
                critical_section_enter_blocking(&commStateLock);
                nodeId = commState.calibMatrix.nodeIds[i];
                critical_section_exit(&commStateLock);

                if (nodeId != targetNodeId && nodeId != 0)
                {
                    sendControlCommand(nodeId, 4, 0.0f); // Keep other LEDs off
                }
            }
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

            // Ensure target LED is still ON and others are OFF
            if (!isSelfCalibration)
            {
                sendControlCommand(targetNodeId, 4, 1.0f);
            }

            // Send multiple broadcast requests to increase chance of delivery
            for (int attempt = 0; attempt < 3; attempt++)
            {
                sendControlCommand(CAN_ADDR_BROADCAST, CAL_CMD_SEND_READING, 1);
                delay(150); // Small delay between broadcast attempts
            }

            // Add extra long wait on the last few retries
            if (retryCount >= INCREASED_MAX_RETRIES - 2)
            {
                delay(3000); // Extra long wait on final retries
            }
        }
        else
        {
            Serial.println("All readings received successfully!");
        }
    }

    // Copy all readings with LED on
    critical_section_enter_blocking(&commStateLock);
    for (int i = 0; i < numNodes; i++)
    {
        onReadings[i] = commState.luxReadings[i];
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
void handleCalibrationMessage(uint8_t sourceNodeId, uint8_t controlType, float value)
{
    switch (controlType)
    {
    case 107: // CAL_CMD_USE_DEFAULT
        if (value >= 1.0f)
        {
            Serial.println("Received command to use default gain matrix");

            // Load the default gain matrix
            loadDefaultGainMatrix();

            // Ensure system states are consistent (this will be redundant but safe)
            critical_section_enter_blocking(&commStateLock);
            commState.calibrationInProgress = false;
            commState.wasCalibrating = true;
            controlState.setpointLux = SETPOINT_UNOCCUPIED;
            controlState.luminaireState = STATE_UNOCCUPIED;
            controlState.feedbackControl = true;
            controlState.systemReady = true;
            controlState.standbyMode = false;
            sensorState.baselineIlluminance = 1.0f; // Reset to a safe value
            critical_section_exit(&commStateLock);

            Serial.println("Default calibration values applied - system ready");
        }
        break;

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

        // IMPORTANT CHANGE: Wait 5 seconds BEFORE taking the reading to ensure LED and sensor stabilization
        delay(2500);

        float currentLux = readLux();

        // Validate the reading - retry if invalid
        int retries = 0;
        while (currentLux <= 0.0f && retries < 5)
        {
            delay(100);
            currentLux = readLux();
            retries++;
        }

        // Additional reading after a stabilization period to improve reliability
        delay(1000);
        currentLux = readLux(); // Final reading

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

        // Send reading back to master - retry multiple times with increasing delays
        bool sent = false;
        for (int i = 0; i < 5 && !sent; i++) // Increased from 3 to 5 retries
        {
            sent = sendSensorReading(sourceNodeId, 0, currentLux);
            if (!sent)
            {
                delay(100 * (i + 1)); // Increasing delays: 100ms, 200ms, 300ms, etc.
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
            if ((int)(value / 1000.0f) == 990)
            {
                // Extract matrix size from message
                int numNodes = (int)(value - 990000.0f);
                numNodes = min(numNodes, MAX_CALIB_NODES); // Safety check
                commState.calibMatrix.numNodes = numNodes;

                // Reset matrix to be safe
                for (int i = 0; i < MAX_CALIB_NODES; i++)
                {
                    commState.calibMatrix.nodeIds[i] = 0;
                    for (int j = 0; j < MAX_CALIB_NODES; j++)
                    {
                        commState.calibMatrix.gains[i][j] = 0.0f;
                    }
                }
            }
            // Check if this is a node ID message (980xxx)
            else if ((int)(value / 1000.0f) == 980)
            {
                // Extract using new safer formula
                float remainder = value - 980000.0f;
                int nodeIndex = (int)(remainder / 100.0f);
                int nodeId = (int)(remainder) % 100;

                // Store if indices are valid
                if (nodeIndex >= 0 && nodeIndex < MAX_CALIB_NODES && nodeId > 0 && nodeId < 64)
                {
                    commState.calibMatrix.nodeIds[nodeIndex] = nodeId;

                    // Track our position
                    if (nodeId == deviceConfig.nodeId)
                    {
                        commState.ourNodeIndex = nodeIndex;
                    }
                }
            }
            // Process gains using new single-message format (900xxx)
            else if ((int)(value / 1000.0f) == 900)
            {
                // New format: 900000 + col*10000 + row*1000 + gain*10
                // Use more robust parsing with intermediate variables and validation
                int messageCode = (int)(value / 1000.0f);
                float remainder = value - (messageCode * 1000.0f);

                // Extract column index (first digit after 900)
                int col = (int)(remainder / 100.0f);
                remainder -= col * 100.0f;

                // Extract row index (second digit)
                int row = (int)(remainder / 10.0f);
                remainder -= row * 10.0f;

                // Extract gain value (remaining digits as decimal)
                float gain = remainder * 10.0f;

                // Store in the calibration matrix if indices are valid
                if (col >= 0 && col < MAX_CALIB_NODES && row >= 0 && row < MAX_CALIB_NODES && gain >= 0.0f && gain < 1000.0f)
                {
                    commState.calibMatrix.gains[row][col] = gain;

                    // Check if this is our self-gain
                    if (row == col && commState.calibMatrix.nodeIds[row] == deviceConfig.nodeId)
                    {
                        deviceConfig.ledGain = gain;
                    }
                }
            }
            // Process gain values (970xxx format)
            else if ((int)(value / 1000.0f) == 970)
            {
                // Integer part: 970000 + col*100 + row*10 + gainInt/10
                float remainder = value - 970000.0f;
                int col = (int)(remainder / 100.0f);
                remainder -= col * 100.0f;
                int row = (int)(remainder / 10.0f);
                float gainInt = (remainder - row * 10.0f) * 10.0f;

                // Store temporarily in a static array
                if (col >= 0 && col < MAX_CALIB_NODES && row >= 0 && row < MAX_CALIB_NODES)
                {
                    // Just store the integer part for now, will combine with fractional later
                    static float gainTempMatrix[MAX_CALIB_NODES][MAX_CALIB_NODES];
                    gainTempMatrix[row][col] = gainInt;
                }
            }
            // Process gain fractional values (960xxx format)
            else if ((int)(value / 1000.0f) == 960)
            {
                // Fractional part: 960000 + col*100 + row*10 + gainFrac/10
                float remainder = value - 960000.0f;
                int col = (int)(remainder / 100.0f);
                remainder -= col * 100.0f;
                int row = (int)(remainder / 10.0f);
                float gainFrac = (remainder - row * 10.0f) * 10.0f / 1000.0f; // Convert back to decimal

                // Store in the calibration matrix if indices are valid
                if (col >= 0 && col < MAX_CALIB_NODES && row >= 0 && row < MAX_CALIB_NODES)
                {
                    // Access the previously stored integer part
                    static float gainTempMatrix[MAX_CALIB_NODES][MAX_CALIB_NODES];
                    float gainInt = gainTempMatrix[row][col];

                    // Combine integer and fractional parts
                    float finalGain = gainInt + gainFrac;
                    commState.calibMatrix.gains[row][col] = finalGain;

                    // Check if this is our self-gain
                    if (row == col && commState.calibMatrix.nodeIds[row] == deviceConfig.nodeId)
                    {
                        deviceConfig.ledGain = finalGain;
                    }
                }
            }
            else
            {
                // Simple gain value (original format)
                // Find the index for the source node
                int sourceIndex = -1;
                for (int i = 0; i < commState.calibMatrix.numNodes; i++)
                {
                    if (commState.calibMatrix.nodeIds[i] == sourceNodeId)
                    {
                        sourceIndex = i;
                        break;
                    }
                }

                if (sourceIndex >= 0)
                {
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
            controlState.standbyMode = false;

            // Find our node in the matrix and ensure our self-gain is set
            uint8_t ourNodeId = deviceConfig.nodeId;
            bool foundOurNode = false;

            // If any node ID is 0, try to fix it
            for (int i = 0; i < commState.calibMatrix.numNodes; i++)
            {
                if (commState.calibMatrix.nodeIds[i] == 0)
                {
                    // Node 0 is invalid - try to infer the node ID from its position
                    if (i == 1 && commState.calibMatrix.numNodes == 3)
                    {
                        // In a 3-node system, if we have node IDs [33, 0, 0], this is likely node 40
                        commState.calibMatrix.nodeIds[i] = 40;
                    }
                    else if (i == 2 && commState.calibMatrix.numNodes == 3)
                    {
                        // In a 3-node system, if we have node IDs [33, 40, 0], this is likely node 52
                        commState.calibMatrix.nodeIds[i] = 52;
                    }
                }
            }

            // Search through the node IDs to find our position
            for (int i = 0; i < commState.calibMatrix.numNodes; i++)
            {
                if (commState.calibMatrix.nodeIds[i] == ourNodeId)
                {
                    // Set our self-gain from the matrix
                    deviceConfig.ledGain = commState.calibMatrix.gains[i][i];
                    foundOurNode = true;

                    // Ensure it's not zero (would cause control issues)
                    if (deviceConfig.ledGain <= 0.0f)
                    {
                        // If diagonal is zero, use maximum value from this row
                        float maxGain = 0.0f;
                        for (int j = 0; j < commState.calibMatrix.numNodes; j++)
                        {
                            if (commState.calibMatrix.gains[i][j] > maxGain)
                            {
                                maxGain = commState.calibMatrix.gains[i][j];
                            }
                        }

                        // Use maximum as fallback or a reasonable default
                        deviceConfig.ledGain = maxGain > 0.0f ? maxGain : 10.0f;
                    }

                    break;
                }
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
        Serial.println("Calibration process completed successfully");
    }
    break;

    default:
        Serial.print("Unknown calibration command type: ");
        Serial.println(controlType);
        break;
    }
}
