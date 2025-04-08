#include <Arduino.h>
#include <math.h>
#include "pico/multicore.h"

#include "Globals.h"
#include "CANComm.h"
#include "CommandInterface.h"
#include "DataLogger.h"
#include "LEDDriver.h"
#include "Metrics.h"
#include "PIController.h"
#include "SensorManager.h"

//=============================================================================
// CONSTANTS AND DEFINITIONS
//=============================================================================

// Maximum command line length
#define CMD_MAX_LENGTH 64
// Maximum number of tokens in a command
#define MAX_TOKENS 6
// Maximum length of a single token
#define TOKEN_MAX_LENGTH 16
// Maximum number of pending queries that can be tracked
#define MAX_PENDING_QUERIES 10

// Forward declarations of functions used but defined elsewhere
extern float readLux();
extern float getVoltageAtLDR();
extern float getExternalIlluminance();
extern float getPowerConsumption();
extern unsigned long getElapsedTime();
extern bool responseReceived;
extern bool canMonitorEnabled;

static void handleDataBufferQuery(const char *subCommand, int numTokens, char tokens[][TOKEN_MAX_LENGTH]);
static void handleBasicVariableQuery(const char *subCommand, const char *idx);
extern float computeVisibilityErrorFromBuffer();
extern float computeEnergyFromBuffer();

float flickerWithFilter = 0.0f;

//==========================================================================================================================================================
// DATA STRUCTURES
//==========================================================================================================================================================

// Structure to track pending query requests
struct PendingQuery
{
    bool active;               // Is this query slot active?
    uint8_t targetNode;        // Node we're querying
    uint8_t queryType;         // Type of query we sent
    char originalCommand[16];  // Original command (like "y", "u", "V", etc.)
    char displayIndex[8];      // Index to display in response
    unsigned long timeoutTime; // When this query expires
};

//==========================================================================================================================================================
// QUERY MANAGEMENT
//==========================================================================================================================================================

// Array of pending queries
static PendingQuery pendingQueries[MAX_PENDING_QUERIES];

/**
 * Initialize the pending queries array
 * Clears all query slots by marking them as inactive
 */
void initPendingQueries()
{
    for (int i = 0; i < MAX_PENDING_QUERIES; i++)
    {
        pendingQueries[i].active = false;
    }
}

/**
 * Add a new query to the pending queries list
 *
 * @param targetNode The node ID being queried
 * @param queryType The type of query being sent
 * @param cmd The original command string (e.g., "u", "y", "V")
 * @param index The display index to use in the response
 * @return true if query was added successfully, false if no slots available
 */
static bool addPendingQuery(uint8_t targetNode, uint8_t queryType, const char *cmd, const char *index)
{
    // Find an inactive slot
    int slot = -1;
    for (int i = 0; i < MAX_PENDING_QUERIES; i++)
    {
        if (!pendingQueries[i].active)
        {
            slot = i; // Save the slot index we found
            break;
        }
    }

    if (slot == -1)
    {
        return false; // No free slots
    }

    // Set up the pending query
    pendingQueries[slot].active = true;
    pendingQueries[slot].targetNode = targetNode;
    pendingQueries[slot].queryType = queryType;
    strncpy(pendingQueries[slot].originalCommand, cmd, sizeof(pendingQueries[slot].originalCommand) - 1);
    pendingQueries[slot].originalCommand[sizeof(pendingQueries[slot].originalCommand) - 1] = '\0';
    strncpy(pendingQueries[slot].displayIndex, index, sizeof(pendingQueries[slot].displayIndex) - 1);
    pendingQueries[slot].displayIndex[sizeof(pendingQueries[slot].displayIndex) - 1] = '\0';

    // Increase timeout from default 1000ms to 2000ms for more reliable response capture
    pendingQueries[slot].timeoutTime = millis() + 2000;

    return true;
}

/**
 * Process all pending queries
 * Checks for responses or timeouts for each active query
 * Should be called regularly from the main loop
 */
static void processPendingQueries()
{
    // Check if we received any responses
    extern bool responseReceived;
    extern uint8_t responseSourceNode;
    extern float responseValue;
    can_frame frame;
    uint8_t msgType, destAddr;
    bool messageProcessed = false;

    // First check if we have a pending response in the global variables
    if (responseReceived)
    {

        // Try to match this response to one of our pending queries
        for (int i = 0; i < MAX_PENDING_QUERIES; i++)
        {
            if (pendingQueries[i].active && pendingQueries[i].targetNode == responseSourceNode)
            {
                // We have a matching query!
                Serial.print("CAN: Matched response to pending query ");
                Serial.println(i);

                // Format the command and send response
                if (isupper(pendingQueries[i].originalCommand[0]))
                {
                    // Handle upper case commands (V, F, E)
                    Serial.print(pendingQueries[i].originalCommand);
                }
                else
                {
                    // Handle lower case commands
                    Serial.print(pendingQueries[i].originalCommand);
                }

                Serial.print(" ");
                Serial.print(pendingQueries[i].displayIndex);
                Serial.print(" ");

                // Format the value based on command type (same formatting as before)
                const char *cmd = pendingQueries[i].originalCommand;
                if (strcmp(cmd, "u") == 0 || strcmp(cmd, "F") == 0 || strcmp(cmd, "E") == 0)
                {
                    Serial.println(responseValue, 4); // 4 decimal places
                }
                else if (strcmp(cmd, "V") == 0 || strcmp(cmd, "y") == 0 ||
                         strcmp(cmd, "p") == 0 || strcmp(cmd, "d") == 0)
                {
                    Serial.println(responseValue, 2); // 2 decimal places
                }
                else if (strcmp(cmd, "v") == 0)
                {
                    Serial.println(responseValue, 3); // 3 decimal places
                }
                else if (strcmp(cmd, "o") == 0 || strcmp(cmd, "a") == 0 ||
                         strcmp(cmd, "f") == 0 || strcmp(cmd, "t") == 0)
                {
                    Serial.println((int)responseValue); // Integer values
                }
                else
                {
                    Serial.println(responseValue); // Default format
                }

                // Mark this query as handled
                pendingQueries[i].active = false;
                messageProcessed = true;

                // Add success acknowledgment
                Serial.println("ack");
                break;
            }
        }

        // If we didn't find a matching query, still reset the flag
        responseReceived = false;
    }

    // Check for direct CAN messages - Try multiple reads to clear buffer
    int msgCount = 0;
    while (msgCount < 5 && readCANMessage(&frame) == MCP2515::ERROR_OK)
    {
        msgCount++;

        parseCANId(frame.can_id, msgType, destAddr);
        critical_section_enter_blocking(&commStateLock);
        uint8_t myNodeId = deviceConfig.nodeId;
        critical_section_exit(&commStateLock);

        // Only process response messages addressed to this node or broadcast
        if (msgType == CAN_TYPE_RESPONSE &&
            (destAddr == myNodeId || destAddr == CAN_ADDR_BROADCAST))
        {

            uint8_t senderNodeID = frame.data[0];
            uint8_t responseType = frame.data[1];

            // Debug info for diagnostics
            Serial.print("CAN: Received response from node ");
            Serial.print(senderNodeID);
            Serial.print(", type: ");
            Serial.print(responseType);

            // Check if it's a query response (type 2)
            if (responseType == 2)
            {
                float value = bytesToFloat(&frame.data[2]);
                Serial.print(", value: ");
                Serial.println(value);

                // Find matching query
                for (int i = 0; i < MAX_PENDING_QUERIES; i++)
                {
                    if (pendingQueries[i].active && pendingQueries[i].targetNode == senderNodeID)
                    {
                        // Format the command and send response
                        if (isupper(pendingQueries[i].originalCommand[0]))
                        {
                            Serial.print(pendingQueries[i].originalCommand);
                        }
                        else
                        {
                            Serial.print(pendingQueries[i].originalCommand);
                        }

                        Serial.print(" ");
                        Serial.print(pendingQueries[i].displayIndex);
                        Serial.print(" ");

                        // Format the value based on command type (same formatting as before)
                        const char *cmd = pendingQueries[i].originalCommand;
                        if (strcmp(cmd, "u") == 0 || strcmp(cmd, "F") == 0 || strcmp(cmd, "E") == 0)
                        {
                            Serial.println(value, 4); // 4 decimal places
                        }
                        else if (strcmp(cmd, "V") == 0 || strcmp(cmd, "y") == 0 ||
                                 strcmp(cmd, "p") == 0 || strcmp(cmd, "d") == 0)
                        {
                            Serial.println(value, 2); // 2 decimal places
                        }
                        else if (strcmp(cmd, "v") == 0)
                        {
                            Serial.println(value, 3); // 3 decimal places
                        }
                        else if (strcmp(cmd, "o") == 0 || strcmp(cmd, "a") == 0 ||
                                 strcmp(cmd, "f") == 0 || strcmp(cmd, "t") == 0)
                        {
                            Serial.println((int)value); // Integer values
                        }
                        else
                        {
                            Serial.println(value); // Default format
                        }

                        // Mark this query as handled
                        pendingQueries[i].active = false;
                        messageProcessed = true;

                        break;
                    }
                }
            }
            else
            {
                Serial.println(" (not a query response)");
            }
        }
    }

    // Check for timeouts on all active queries
    unsigned long currentTime = millis();
    for (int i = 0; i < MAX_PENDING_QUERIES; i++)
    {
        if (pendingQueries[i].active)
        {
            if (currentTime > pendingQueries[i].timeoutTime)
            {
                Serial.print("err: No response from node ");
                Serial.println(pendingQueries[i].targetNode);

                // Consider retrying the query once before giving up
                // This is helpful for intermittent communication issues
                if (pendingQueries[i].timeoutTime > 0)
                { // Check if not already retried
                    Serial.print("Retrying query to node ");
                    Serial.println(pendingQueries[i].targetNode);

                    // Send the query again with the same parameters
                    if (sendControlCommand(pendingQueries[i].targetNode,
                                           pendingQueries[i].queryType, 0.0f))
                    {
                        // Set a shorter timeout for the retry
                        pendingQueries[i].timeoutTime = currentTime + 1500;
                    }
                    else
                    {
                        // If retry failed, mark as inactive
                        pendingQueries[i].active = false;
                    }
                }
                else
                {
                    // Already retried or couldn't retry, mark as inactive
                    pendingQueries[i].active = false;
                }
            }
        }
    }
}

//==========================================================================================================================================================
// COMMAND PARSING HELPERS
//==========================================================================================================================================================

/**
 * Parse a string to an integer with error checking
 *
 * @param str String to parse
 * @param result Output parameter for the parsed integer value
 * @return true if parsing was successful, false otherwise
 */
bool parseIntParam(const char *str, int *result)
{
    if (str == NULL || result == NULL)
    {
        return false;
    }

    // Check if string is empty
    if (str[0] == '\0')
    {
        return false;
    }

    char *endPtr;
    long value = strtol(str, &endPtr, 10);

    // Check if any conversion happened and if we reached the end of the string
    if (endPtr == str || *endPtr != '\0')
    {
        return false;
    }

    // Check for overflow/underflow
    if (value > INT_MAX || value < INT_MIN)
    {
        return false;
    }

    *result = (int)value;
    return true;
}

/**
 * Parse a string to a float with error checking
 *
 * @param str String to parse
 * @param result Output parameter for the parsed float value
 * @return true if parsing was successful, false otherwise
 */
static bool parseFloatParam(const char *str, float *result)
{
    if (str == NULL || result == NULL)
    {
        return false;
    }

    // Check if string is empty
    if (str[0] == '\0')
    {
        return false;
    }

    char *endPtr;
    float value = strtof(str, &endPtr);

    // Check if any conversion happened and if we reached the end of the string
    if (endPtr == str || *endPtr != '\0')
    {
        return false;
    }

    // Check for NaN or infinity
    if (isnan(value) || isinf(value))
    {
        return false;
    }

    *result = value;
    return true;
}

/**
 * Parse a command line into tokens
 *
 * @param cmdLine The command line string to parse
 * @param tokens Array to store the parsed tokens
 * @param maxTokens Maximum number of tokens to extract
 * @return Number of tokens found
 */
int parseTokensChar(const char *cmdLine, char tokens[][TOKEN_MAX_LENGTH], int maxTokens)
{
    int numTokens = 0;
    int cmdLen = strlen(cmdLine);
    int i = 0;

    while (i < cmdLen && numTokens < maxTokens)
    {
        // Skip leading whitespace
        while (i < cmdLen && isspace(cmdLine[i]))
            i++;

        if (i >= cmdLen)
            break;

        // Copy token
        int tokenLen = 0;
        while (i < cmdLen && !isspace(cmdLine[i]) && tokenLen < TOKEN_MAX_LENGTH - 1)
        {
            tokens[numTokens][tokenLen] = cmdLine[i];
            tokenLen++;
            i++;
        }

        tokens[numTokens][tokenLen] = '\0';
        numTokens++;
    }

    return numTokens;
}

/**
 * Check if float value is within specified range
 *
 * @param value Value to check
 * @param min Minimum allowed value (inclusive)
 * @param max Maximum allowed value (inclusive)
 * @return true if value is within range, false otherwise
 */
static bool isInRange(float value, float min, float max)
{
    return value >= min && value <= max;
}

/**
 * Prepare command line by trimming and tokenizing
 *
 * @param cmdLine The raw command line to process
 * @param tokens Array to store the parsed tokens
 * @return Number of tokens found
 */
static int prepareCommand(const char *cmdLine, char tokens[][TOKEN_MAX_LENGTH])
{
    char trimmedCmd[CMD_MAX_LENGTH];
    int cmdLen = strlen(cmdLine);

    // Trim leading spaces
    int startPos = 0;
    while (startPos < cmdLen && cmdLine[startPos] == ' ')
    {
        startPos++;
    }

    // Trim trailing spaces
    int endPos = cmdLen - 1;
    while (endPos >= 0 && cmdLine[endPos] == ' ')
    {
        endPos--;
    }

    // Copy trimmed command
    if (endPos >= startPos)
    {
        int trimmedLen = endPos - startPos + 1;
        if (trimmedLen >= CMD_MAX_LENGTH)
            trimmedLen = CMD_MAX_LENGTH - 1;
        strncpy(trimmedCmd, cmdLine + startPos, trimmedLen);
        trimmedCmd[trimmedLen] = '\0';
    }
    else
    {
        // Empty command after trimming
        trimmedCmd[0] = '\0';
        return 0;
    }

    // Tokenize the command line
    return parseTokensChar(trimmedCmd, tokens, MAX_TOKENS);
}

//==========================================================================================================================================================
// COMMAND EXECUTION UTILITIES
//==========================================================================================================================================================

/**
 * Map variable name to numeric code for CAN transmission
 *
 * @param var Variable name string (e.g., "y", "u", "p")
 * @param varCode Output parameter to store the resulting code
 * @return true if mapping successful, false if variable not recognized
 */
static bool mapVariableToCode(const char *var, float *varCode)
{
    if (strcmp(var, "y") == 0)
    {
        *varCode = 0.0f; // Illuminance
    }
    else if (strcmp(var, "u") == 0)
    {
        *varCode = 1.0f; // Duty cycle
    }
    else if (strcmp(var, "p") == 0)
    {
        *varCode = 2.0f; // Power
    }
    else if (strcmp(var, "d") == 0)
    {
        *varCode = 3.0f; // External illuminance
    }
    else if (strcmp(var, "r") == 0)
    {
        *varCode = 4.0f; // Reference illuminance
    }
    else if (strcmp(var, "o") == 0)
    {
        *varCode = 5.0f; // Occupancy state
    }
    else if (strcmp(var, "a") == 0)
    {
        *varCode = 6.0f; // Anti-windup state
    }
    else if (strcmp(var, "f") == 0)
    {
        *varCode = 7.0f; // Feedback control state
    }
    else if (strcmp(var, "v") == 0)
    {
        *varCode = 8.0f; // LDR voltage
    }
    else if (strcmp(var, "t") == 0)
    {
        *varCode = 9.0f; // Elapsed time
    }
    else if (strcmp(var, "V") == 0)
    {
        *varCode = 10.0f; // Visibility error
    }
    else if (strcmp(var, "F") == 0)
    {
        *varCode = 11.0f; // Flicker
    }
    else if (strcmp(var, "E") == 0)
    {
        *varCode = 12.0f; // Energy
    }
    else
    {
        // Default to illuminance if not recognized
        *varCode = 0.0f;
        return false;
    }

    return true;
}

/**
 * Apply a command locally based on control type
 *
 * @param controlType CAN control type code
 * @param value Command value
 */
void applyLocalCommand(uint8_t controlType, float value)
{
    switch (controlType)
    {
    case 4: // Set duty cycle directly
        setLEDDutyCycle(value);
        break;
    case 5: // Set LED percentage
        setLEDPercentage(value);
        break;
    case 6: // Set LED power
        setLEDPower(value);
        break;
    case 7: // Set luminaire state
        changeState((LuminaireState)((int)value));
        break;
    case 8: // Set anti-windup
        critical_section_enter_blocking(&commStateLock);
        controlState.antiWindup = (value > 0.5f);
        critical_section_exit(&commStateLock);
        break;
    case 9: // Set feedback control
        critical_section_enter_blocking(&commStateLock);
        controlState.feedbackControl = (value > 0.5f);
        critical_section_exit(&commStateLock);
        break;
    case 10: // Set reference illuminance (lux)
        critical_section_enter_blocking(&commStateLock);
        controlState.setpointLux = value;
        critical_section_exit(&commStateLock);
        break;
    case 14: // Set filter enable/disable
        critical_section_enter_blocking(&commStateLock);
        sensorState.filterEnabled = (value > 0.5f);
        critical_section_exit(&commStateLock);
        break;
    }
}

/**
 * Determine if a command should be executed locally or forwarded to another node
 *
 * @param targetNode The node ID specified in the command
 * @return true if command should be forwarded, false for local execution
 */
static bool shouldForwardCommand(uint8_t targetNode)
{
    // If target is broadcast (0) or matches this node, process locally
    critical_section_enter_blocking(&commStateLock);
    if (targetNode == 0 || targetNode == deviceConfig.nodeId)
    {
        critical_section_exit(&commStateLock);
        return false;
    }
    // Otherwise, forward to target node
    critical_section_exit(&commStateLock);
    return true;
}

/**
 * Generic handler for targeted commands that follow the same pattern
 *
 * @param targetNode Node to target (0 for broadcast)
 * @param controlType CAN control type code
 * @param value Command value
 * @return true if handled successfully
 */
static bool handleTargetedCommand(uint8_t targetNode, uint8_t controlType, float value)
{
    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
        // Forward to specific node
        if (sendControlCommand(targetNode, controlType, value))
        {
            Serial.println("ack");
        }
        else
        {
            Serial.println("err: CAN forwarding failed");
        }
        return true;
    }

    // Handle broadcast case (targetNode = 0)
    if (targetNode == 0)
    {
        if (sendControlCommand(CAN_ADDR_BROADCAST, controlType, value))
        {
            // Also apply locally since broadcast includes this node
            applyLocalCommand(controlType, value);
            Serial.println("ack");
        }
        else
        {
            Serial.println("err: CAN broadcast failed");
        }
        return true;
    }

    // Apply locally
    applyLocalCommand(controlType, value);
    Serial.println("ack");
    return true;
}

/**
 * Handle a local query (non-forwarded)
 *
 * @param subCommand The query command
 * @param originalCase Original case-preserved command
 * @param idx Node index string
 * @param numTokens Total number of tokens
 * @param tokens Full token array
 */
static void handleLocalQuery(const char *subCommand, const char *originalCase, const char *idx, int numTokens, char tokens[][TOKEN_MAX_LENGTH])
{
    // Quality metrics
    if (strcmp(originalCase, "V") == 0)
    {
        float V = computeVisibilityErrorFromBuffer();
        Serial.print("V ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(V, 2);
    }
    else if (strcmp(originalCase, "F") == 0)
    {
        float F = computeFlickerFromBuffer();
        Serial.print("F ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(F, 4);
    }
    else if (strcmp(originalCase, "E") == 0)
    {
        float E = computeEnergyFromBuffer();
        Serial.print("E ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(E, 4);
    }
    // Control system variables
    else if (strcmp(subCommand, "u") == 0)
    {
        Serial.print("u ");
        Serial.print(idx);
        Serial.print(" ");
        critical_section_enter_blocking(&commStateLock);
        Serial.println(controlState.dutyCycle, 4);
        critical_section_exit(&commStateLock);
    }
    else if (strcmp(subCommand, "o") == 0)
    {
        critical_section_enter_blocking(&commStateLock);
        int occVal = static_cast<int>(controlState.luminaireState);
        critical_section_exit(&commStateLock);
        Serial.print("o ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(occVal);
    }
    // Additional query handlers for other variables
    else if (strcmp(subCommand, "b") == 0 || strcmp(subCommand, "bigdump") == 0 ||
             strcmp(subCommand, "mdump") == 0)
    {
        handleDataBufferQuery(subCommand, numTokens, tokens);
    }
    else
    {
        // Handle remaining variable queries
        handleBasicVariableQuery(subCommand, idx);
    }
}

//=============================================================================
// DATA STREAMING SUBSYSTEM
//=============================================================================

void startStream(const char *var, int index)
{

    commState.streamingEnabled = true;
    // Allocate memory for streaming variable if needed
    if (commState.streamingVar == nullptr)
    {
        commState.streamingVar = (char *)malloc(16); // Allocate space for the variable name
    }
    strncpy(commState.streamingVar, var, 15);
    commState.streamingVar[15] = '\0'; // Ensure null termination
    commState.streamingIndex = index;
    commState.lastStreamTime = millis();
}

void stopStream(const char *var, int index)
{
    critical_section_enter_blocking(&commStateLock);
    // Only stop streaming if we're tracking the specified node and variable
    if (commState.streamingEnabled &&
        commState.streamingIndex == index &&
        commState.streamingVar != nullptr &&
        strcmp(commState.streamingVar, var) == 0)
    {
        commState.streamingEnabled = false;
        if (commState.streamingVar != nullptr)
        {
            commState.streamingVar[0] = '\0';
        }
    }
    critical_section_exit(&commStateLock);
}

/**
 * Process streaming in main loop
 * Sends requested variable at regular intervals
 */
void handleStreaming()
{
    critical_section_enter_blocking(&commStateLock);
    if (!commState.streamingEnabled || (millis() - commState.lastStreamTime < 500))
    {
        critical_section_exit(&commStateLock);
        return;
    }

    char *var = commState.streamingVar;
    int index = commState.streamingIndex;
    uint8_t myNodeId = deviceConfig.nodeId;
    commState.lastStreamTime = millis();
    critical_section_exit(&commStateLock);

    if (var == nullptr || var[0] == '\0')
        return;

    // Only generate and print values if we're streaming our own data
    // Otherwise, remote values will come through CAN messages
    if (index == myNodeId)
    {
        // This is local streaming - get data from local sensors
        if (strcmp(var, "y") == 0)
        {
            float lux = readLux();
            Serial.print("y ");
            Serial.print(index);
            Serial.print(" ");
            Serial.println(lux, 2);
        }
        else if (strcmp(var, "u") == 0)
        {
            Serial.print("u ");
            Serial.print(index);
            Serial.print(" ");
            critical_section_enter_blocking(&commStateLock);
            Serial.println(controlState.dutyCycle, 4);
            critical_section_exit(&commStateLock);
        }
        else if (strcmp(var, "p") == 0 || strcmp(var, "V") == 0 ||
                 strcmp(var, "F") == 0 || strcmp(var, "E") == 0)
        {
            float power = getPowerConsumption();
            Serial.print(var);
            Serial.print(" ");
            Serial.print(index);
            Serial.print(" ");
            if (strcmp(var, "p") == 0)
            {
                Serial.println(power, 2);
            }
            else if (strcmp(var, "F") == 0)
            { // flicker
                float currentFlicker = computeFlickerFromBuffer();
                flickerWithFilter = currentFlicker; // Update the global variable
                Serial.println(flickerWithFilter, 2);
            }
            else if (strcmp(var, "V") == 0)
            { // Visibility
                Serial.println(computeVisibilityErrorFromBuffer(), 2);
            }
            else if (strcmp(var, "E") == 0)
            { // Energy
                Serial.println(computeEnergyFromBuffer(), 2);
            }
        }
    }
    else
    {
        // For remote nodes, we don't generate data locally.
        // Remote values should come through CAN messages.
        // We can refresh the request occasionally to ensure data keeps flowing.
        float varCode;
        if (mapVariableToCode(var, &varCode))
        {
            // Every 2 seconds (4 cycles of 500ms), send a refresh request
            static unsigned long lastRefreshTime = 0;
            unsigned long now = millis();
            if (now - lastRefreshTime > 2000)
            {
                lastRefreshTime = now;
                sendControlCommand(index, 11, varCode); // 11 = start streaming
            }
        }
    }
}

void handleRemoteStreamRequests()
{
    unsigned long now = millis();

    // Check if it's time to send data (every 500ms)
    critical_section_enter_blocking(&commStateLock);
    StreamRequest *requests = commState.remoteStreamRequests;
    critical_section_exit(&commStateLock);

    for (int i = 0; i < MAX_STREAM_REQUESTS; i++)
    {
        critical_section_enter_blocking(&commStateLock);
        if (!requests[i].active)
        {
            critical_section_exit(&commStateLock);
            continue;
        }

        if (now - requests[i].lastSent >= 500)
        {
            critical_section_exit(&commStateLock);
            float value = 0.0;

            switch (requests[i].variableType)
            {
            case 0: // y = illuminance
                value = readLux();
                break;
            case 1: // u = duty cycle
                critical_section_enter_blocking(&commStateLock);
                value = controlState.dutyCycle;
                critical_section_exit(&commStateLock);
                break;
            case 2: // p = power
                value = getPowerConsumption();
                break;
            default:
                value = 0.0;
            }

            // Send the value to the requesting node
            sendSensorReading(requests[i].requesterNode,
                              requests[i].variableType,
                              value);

            critical_section_enter_blocking(&commStateLock);
            requests[i].lastSent = now;
            critical_section_exit(&commStateLock);
        }
        else
        {
            critical_section_exit(&commStateLock);
        }
    }
}

//==========================================================================================================================================================
// COMMAND CATEGORY HANDLERS
//==========================================================================================================================================================

/**
 * Handle queries related to data buffer operations
 * Processes buffer commands like dump, export, etc.
 *
 * @param subCommand The specific buffer operation requested
 * @param numTokens Number of tokens in the command
 * @param tokens Full token array
 */
static void handleDataBufferQuery(const char *subCommand, int numTokens, char tokens[][TOKEN_MAX_LENGTH])
{
    if (strcmp(subCommand, "b") == 0 || strcmp(subCommand, "bigdump") == 0)
    {
        // Full data dump to serial
        dumpBufferToSerial();
        Serial.println("ack");
    }
    else if (strcmp(subCommand, "mdump") == 0)
    {
        // Clear the buffer
        clearBuffer();
        Serial.println("Buffer cleared");
        Serial.println("ack");
    }
    else
    {
        Serial.println("err: Unknown buffer command");
    }
}

/**
 * Handle basic variable query operations
 * Responds with the current value of the requested variable
 *
 * @param subCommand The variable to query
 * @param idx Node index to display in the response
 */
static void handleBasicVariableQuery(const char *subCommand, const char *idx)
{
    if (strcmp(subCommand, "y") == 0)
    {
        // Current illuminance
        float lux = readLux();
        Serial.print("y ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(lux, 2);
    }
    else if (strcmp(subCommand, "p") == 0)
    {
        // Power consumption
        float power = getPowerConsumption();
        Serial.print("p ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(power, 2);
    }
    else if (strcmp(subCommand, "t") == 0)
    {
        // Elapsed time
        unsigned long time = getElapsedTime();
        Serial.print("t ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(time);
    }
    else if (strcmp(subCommand, "v") == 0)
    {
        // LDR voltage
        float voltage = getVoltageAtLDR();
        Serial.print("v ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(voltage, 3);
    }
    else if (strcmp(subCommand, "d") == 0)
    {
        // External illuminance
        float extLux = getExternalIlluminance();
        Serial.print("d ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(extLux, 2);
    }
    else if (strcmp(subCommand, "a") == 0)
    {
        // Anti-windup state
        critical_section_enter_blocking(&commStateLock);
        int antiWindup = controlState.antiWindup ? 1 : 0;
        critical_section_exit(&commStateLock);
        Serial.print("a ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(antiWindup);
    }
    else if (strcmp(subCommand, "f") == 0)
    {
        // Feedback control state
        critical_section_enter_blocking(&commStateLock);
        int feedback = controlState.feedbackControl ? 1 : 0;
        critical_section_exit(&commStateLock);
        Serial.print("f ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(feedback);
    }
    else if (strcmp(subCommand, "r") == 0)
    {
        // Reference illuminance (setpoint)
        critical_section_enter_blocking(&commStateLock);
        float setpoint = controlState.setpointLux;
        critical_section_exit(&commStateLock);
        Serial.print("r ");
        Serial.print(idx);
        Serial.print(" ");
        Serial.println(setpoint, 1);
    }
    else
    {
        Serial.print("err: Unknown variable '");
        Serial.print(subCommand);
        Serial.println("'");
    }
}

/**
 * Handle LED control commands (u, p, w)
 *
 * @param tokens Command tokens
 * @param numTokens Number of tokens
 * @return true if command was handled, false otherwise
 */
static bool handleLEDCommands(char tokens[][TOKEN_MAX_LENGTH], int numTokens)
{
    if (strcmp(tokens[0], "u") == 0)
    {
        // Duty cycle command
        if (numTokens < 3)
        {
            Serial.println("err: Missing parameters");
            return true;
        }

        int targetNode;
        float val;

        // Parse node ID
        if (!parseIntParam(tokens[1], &targetNode) || targetNode < 0 || targetNode > 63)
        {
            Serial.println("err: Invalid node ID");
            return true;
        }

        // Parse duty cycle value
        if (!parseFloatParam(tokens[2], &val) || !isInRange(val, 0.0f, 1.0f))
        {
            Serial.println("err: Invalid duty cycle (must be between 0.0 and 1.0)");
            return true;
        }

        // Process command based on target
        if (handleTargetedCommand(targetNode, 4, val))
        {
            return true;
        }
    }
    else if (strcmp(tokens[0], "p") == 0)
    {
        // Percentage command
        if (numTokens < 3)
        {
            Serial.println("err: Missing parameters");
            return true;
        }

        int targetNode;
        float val;

        // Parse node ID
        if (!parseIntParam(tokens[1], &targetNode) || targetNode < 0 || targetNode > 63)
        {
            Serial.println("err: Invalid node ID");
            return true;
        }

        // Parse percentage value
        if (!parseFloatParam(tokens[2], &val) || !isInRange(val, 0.0f, 100.0f))
        {
            Serial.println("err: Invalid percentage (must be between 0.0 and 100.0)");
            return true;
        }

        // Process command based on target
        if (handleTargetedCommand(targetNode, 5, val))
        {
            return true;
        }
    }
    else if (strcmp(tokens[0], "w") == 0)
    {
        // Power in watts command
        if (numTokens < 3)
        {
            Serial.println("err: Missing parameters");
            return true;
        }

        int targetNode;
        float val;

        // Parse node ID
        if (!parseIntParam(tokens[1], &targetNode) || targetNode < 0 || targetNode > 63)
        {
            Serial.println("err: Invalid node ID");
            return true;
        }

        // Parse power value
        if (!parseFloatParam(tokens[2], &val) || !isInRange(val, 0.0f, MAX_POWER_WATTS))
        {
            Serial.println("err: Invalid power (must be between 0.0 and MAX_POWER_WATTS)");
            return true;
        }

        // Process command based on target
        if (handleTargetedCommand(targetNode, 6, val))
        {
            return true;
        }
    }
    else
    {
        return false; // Not an LED command
    }

    return true;
}

/**
 * Handle system state control commands (o, a, fi, f, r, st, cal)
 *
 * @param tokens Command tokens
 * @param numTokens Number of tokens
 * @return true if command was handled, false otherwise
 */
static bool handleSystemStateCommands(char tokens[][TOKEN_MAX_LENGTH], int numTokens)
{
    if (strcmp(tokens[0], "o") == 0)
    {
        // Occupancy control command
        int occupancyState = -1;
        uint8_t targetNode = 0; // Default to broadcast

        // Parse optional parameters
        if (numTokens > 1)
        {
            // First parameter is always occupancy state
            if (!parseIntParam(tokens[1], &occupancyState) || occupancyState < 0 || occupancyState > 2)
            {
                Serial.println("error: invalid occupancy state (0=off, 1=unoccupied, 2=occupied)");
                return true;
            }

            // Second parameter, if present, is target node
            if (numTokens > 2)
            {
                int node;
                if (!parseIntParam(tokens[2], &node) || node < 0 || node > 63)
                {
                    Serial.println("error: invalid node ID");
                    return true;
                }
                targetNode = (uint8_t)node;
            }
        }
        else
        {
            // No parameters, print current state
            critical_section_enter_blocking(&commStateLock);
            int state = (int)controlState.luminaireState;
            critical_section_exit(&commStateLock);

            Serial.print("Occupancy state: ");
            switch (state)
            {
            case 0:
                Serial.println("Off");
                break;
            case 1:
                Serial.println("Unoccupied");
                break;
            case 2:
                Serial.println("Occupied");
                break;
            default:
                Serial.println("Unknown");
            }
            return true;
        }

        // Apply the command locally or forward it
        if (shouldForwardCommand(targetNode))
        {
            sendControlCommand(targetNode, 13, (float)occupancyState);
            Serial.println("ack: forwarded occupancy command");
        }
        else
        {
            // Local execution
            switch (occupancyState)
            {
            case 0:
                changeState(STATE_OFF);
                break;
            case 1:
                changeState(STATE_UNOCCUPIED);
                break;
            case 2:
                changeState(STATE_OCCUPIED);
                break;
            }
            Serial.println("ack: occupancy state changed");
        }

        return true;
    }
    else if (strcmp(tokens[0], "a") == 0)
    {
        // Anti-windup control command
        int antiWindupState = -1;
        uint8_t targetNode = 0; // Default to broadcast

        // Parse optional parameters
        if (numTokens > 1)
        {
            // First parameter is always anti-windup state
            if (!parseIntParam(tokens[1], &antiWindupState) || (antiWindupState != 0 && antiWindupState != 1))
            {
                Serial.println("error: invalid anti-windup state (0=off, 1=on)");
                return true;
            }

            // Second parameter, if present, is target node
            if (numTokens > 2)
            {
                int node;
                if (!parseIntParam(tokens[2], &node) || node < 0 || node > 63)
                {
                    Serial.println("error: invalid node ID");
                    return true;
                }
                targetNode = (uint8_t)node;
            }
        }
        else
        {
            // No parameters, print current state
            critical_section_enter_blocking(&commStateLock);
            bool awState = controlState.antiWindup;
            critical_section_exit(&commStateLock);

            Serial.print("Anti-windup: ");
            Serial.println(awState ? "On" : "Off");
            return true;
        }

        // Apply the command locally or forward it
        if (shouldForwardCommand(targetNode))
        {
            sendControlCommand(targetNode, 8, (float)antiWindupState);
            Serial.println("ack: forwarded anti-windup command");
        }
        else
        {
            // Local execution
            critical_section_enter_blocking(&commStateLock);
            controlState.antiWindup = (antiWindupState == 1);
            critical_section_exit(&commStateLock);
            Serial.println("ack: anti-windup state changed");
        }

        return true;
    }
    else if (strcmp(tokens[0], "fi") == 0)
    {
        // Filter control command
        int filterState = -1;
        uint8_t targetNode = 0; // Default to broadcast

        // Parse optional parameters
        if (numTokens > 1)
        {
            // First parameter is always filter state
            if (!parseIntParam(tokens[1], &filterState) || (filterState != 0 && filterState != 1))
            {
                Serial.println("error: invalid filter state (0=off, 1=on)");
                return true;
            }

            // Second parameter, if present, is target node
            if (numTokens > 2)
            {
                int node;
                if (!parseIntParam(tokens[2], &node) || node < 0 || node > 63)
                {
                    Serial.println("error: invalid node ID");
                    return true;
                }
                targetNode = (uint8_t)node;
            }
        }
        else
        {
            // No parameters, print current state
            critical_section_enter_blocking(&commStateLock);
            bool fState = sensorState.filterEnabled;
            critical_section_exit(&commStateLock);

            Serial.print("Sensor filter: ");
            Serial.println(fState ? "On" : "Off");
            return true;
        }

        // Apply the command locally or forward it
        if (shouldForwardCommand(targetNode))
        {
            sendControlCommand(targetNode, 14, (float)filterState);
            Serial.println("ack: forwarded filter command");
        }
        else
        {
            // Local execution
            critical_section_enter_blocking(&commStateLock);
            sensorState.filterEnabled = (filterState == 1);
            critical_section_exit(&commStateLock);
            Serial.println("ack: filter state changed");
        }

        return true;
    }
    else if (strcmp(tokens[0], "f") == 0)
    {
        // Feedback control command
        int feedbackState = -1;
        uint8_t targetNode = 0; // Default to broadcast

        // Parse optional parameters
        if (numTokens > 1)
        {
            // First parameter is always feedback state
            if (!parseIntParam(tokens[1], &feedbackState) || (feedbackState != 0 && feedbackState != 1))
            {
                Serial.println("error: invalid feedback control state (0=off, 1=on)");
                return true;
            }

            // Second parameter, if present, is target node
            if (numTokens > 2)
            {
                int node;
                if (!parseIntParam(tokens[2], &node) || node < 0 || node > 63)
                {
                    Serial.println("error: invalid node ID");
                    return true;
                }
                targetNode = (uint8_t)node;
            }
        }
        else
        {
            // No parameters, print current state
            critical_section_enter_blocking(&commStateLock);
            bool fbState = controlState.feedbackControl;
            critical_section_exit(&commStateLock);

            Serial.print("Feedback control: ");
            Serial.println(fbState ? "On" : "Off");
            return true;
        }

        // Apply the command locally or forward it
        if (shouldForwardCommand(targetNode))
        {
            sendControlCommand(targetNode, 9, (float)feedbackState);
            Serial.println("ack: forwarded feedback control command");
        }
        else
        {
            // Local execution
            critical_section_enter_blocking(&commStateLock);
            controlState.feedbackControl = (feedbackState == 1);
            critical_section_exit(&commStateLock);

            // Reset controller when feedback is enabled to prevent windup
            if (controlState.feedbackControl)
            {
                pid.reset();
            }

            Serial.println("ack: feedback control state changed");
        }

        return true;
    }
    else if (strcmp(tokens[0], "r") == 0)
    {
        // Reference illuminance command
        float reference = -1.0f;
        uint8_t targetNode = 0; // Default to broadcast

        // Parse optional parameters
        if (numTokens > 1)
        {
            // First parameter is reference value
            if (!parseFloatParam(tokens[1], &reference) || reference < 0 || reference > MAX_ILLUMINANCE)
            {
                Serial.println("error: invalid reference value (0-2000 lux)");
                return true;
            }

            // Second parameter, if present, is target node
            if (numTokens > 2)
            {
                int node;
                if (!parseIntParam(tokens[2], &node) || node < 0 || node > 63)
                {
                    Serial.println("error: invalid node ID");
                    return true;
                }
                targetNode = (uint8_t)node;
            }
        }
        else
        {
            // No parameters, print current reference
            critical_section_enter_blocking(&commStateLock);
            float ref = controlState.setpointLux;
            critical_section_exit(&commStateLock);

            Serial.print("Reference illuminance: ");
            Serial.print(ref);
            Serial.println(" lux");
            return true;
        }

        // Apply the command locally or forward it
        if (shouldForwardCommand(targetNode))
        {
            sendControlCommand(targetNode, 10, reference);
            Serial.println("ack: forwarded reference command");
        }
        else
        {
            // Local execution
            critical_section_enter_blocking(&commStateLock);
            controlState.setpointLux = reference;
            critical_section_exit(&commStateLock);

            // Reset controller to prevent windup on large setpoint changes
            pid.reset();

            Serial.print("ack: reference set to ");
            Serial.print(reference);
            Serial.println(" lux");
        }

        return true;
    }
    else if (strcmp(tokens[0], "st") == 0)
    {
        // Display system status
        critical_section_enter_blocking(&commStateLock);
        float lux = sensorState.filteredLux;
        bool filterEnabled = sensorState.filterEnabled;
        float rawLux = sensorState.rawLux;
        float duty = controlState.dutyCycle;
        bool feedback = controlState.feedbackControl;
        bool antiWindup = controlState.antiWindup;
        float reference = controlState.setpointLux;
        int state = (int)controlState.luminaireState;
        float ledGain = deviceConfig.ledGain;
        float offset = deviceConfig.calibrationOffset;
        critical_section_exit(&commStateLock);

        // Get controller terms
        float pTerm, iTerm;
        pid.getTerms(pTerm, iTerm);

        // Calculate power
        float powerW = duty * MAX_POWER_WATTS;

        // Print system status
        Serial.println("\n===== System Status =====");
        Serial.print("Node ID: ");
        Serial.println(deviceConfig.nodeId);
        Serial.print("Uptime: ");
        Serial.print(getElapsedTime());
        Serial.println(" sec");

        Serial.println("\n-- Control State --");
        Serial.print("Luminaire state: ");
        switch (state)
        {
        case 0:
            Serial.println("Off");
            break;
        case 1:
            Serial.println("Unoccupied");
            break;
        case 2:
            Serial.println("Occupied");
            break;
        default:
            Serial.println("Unknown");
        }
        Serial.print("Reference: ");
        Serial.print(reference);
        Serial.println(" lux");
        Serial.print("Feedback control: ");
        Serial.println(feedback ? "On" : "Off");
        Serial.print("Anti-windup: ");
        Serial.println(antiWindup ? "On" : "Off");

        Serial.println("\n-- Sensor Readings --");
        Serial.print("Illuminance: ");
        Serial.print(lux);
        Serial.println(" lux");
        Serial.print("Raw reading: ");
        Serial.print(rawLux);
        Serial.println(" lux");
        Serial.print("Filter: ");
        Serial.println(filterEnabled ? "On" : "Off");
        Serial.print("External illuminance: ");
        Serial.print(getExternalIlluminance());
        Serial.println(" lux");

        Serial.println("\n-- Output and Control --");
        Serial.print("Duty cycle: ");
        Serial.print(duty * 100.0f);
        Serial.println("%");
        Serial.print("Power: ");
        Serial.print(powerW * 1000.0f);
        Serial.println(" mW");
        Serial.print("P-term: ");
        Serial.println(pTerm);
        Serial.print("I-term: ");
        Serial.println(iTerm);
        Serial.print("LED gain (Kii): ");
        Serial.println(ledGain);
        Serial.print("Calibration offset: ");
        Serial.println(offset);

        Serial.println("========================\n");

        return true;
    }
    else if (strcmp(tokens[0], "cal") == 0)
    {
        // Start calibration
        // The actual calibration is implemented elsewhere
        startCalibration();
        Serial.println("ack: calibration started");
        return true;
    }
    else if (strcmp(tokens[0], "wakeup") == 0 || strcmp(tokens[0], "w") == 0)
    {
        // Get target node ID (if specified)
        uint8_t targetNode = 0; // Default to broadcast
        if (numTokens > 1)
        {
            int node;
            if (!parseIntParam(tokens[1], &node) || node < 0 || node > 63)
            {
                Serial.println("error: invalid node ID");
                return true;
            }
            targetNode = (uint8_t)node;
        }

        // Handle locally or forward command
        if (shouldForwardCommand(targetNode))
        {
            sendControlCommand(targetNode, 15, 1.0f); // 15 is our command type for wakeup
            Serial.println("ack: forwarded wakeup");
        }
        else
        {
            // Exit standby mode and start discovery phase
            critical_section_enter_blocking(&commStateLock);
            controlState.standbyMode = false;
            controlState.systemAwake = true;
            controlState.discoveryStartTime = millis();
            commState.isCalibrationMaster = true; // This node becomes calibration master
            critical_section_exit(&commStateLock);

            // Inform user
            Serial.println("Wake-up sequence initiated");
            Serial.println("Starting 10-second node discovery phase...");
        }
        return true;
    }
    else if (strcmp(tokens[0], "m") == 0)
    {
        critical_section_enter_blocking(&commStateLock);
        
        // Print self gain first (most important value)
        Serial.println("\n----- Calibration Gain Values -----");
        Serial.print("Self-gain (Kii): ");
        Serial.println(deviceConfig.ledGain, 4);
        
        // Print how many nodes we have in the matrix
        int numNodes = commState.calibMatrix.numNodes;
        Serial.print("Matrix Size: ");
        Serial.print(numNodes);
        Serial.println(" nodes");
        
        // Print header with node IDs
        Serial.print("Effect Matrix | ");
        for (int j = 0; j < numNodes; j++) {
            Serial.print("Node ");
            Serial.print(commState.calibMatrix.nodeIds[j]);
            Serial.print(" | ");
        }
        Serial.println();
        
        // Print separator
        for (int j = 0; j <= numNodes; j++) {
            Serial.print("----------");
        }
        Serial.println();
        
        // Print each row with proper labels
        for (int i = 0; i < numNodes; i++) {
            Serial.print("Node ");
            Serial.print(commState.calibMatrix.nodeIds[i]);
            Serial.print(" | ");
            
            for (int j = 0; j < numNodes; j++) {
                Serial.print(commState.calibMatrix.gains[i][j], 4);
                Serial.print(" | ");
            }
            Serial.println();
        }
        
        // Print external light contributions
        Serial.println("\nExternal Illuminance Values:");
        for (int i = 0; i < numNodes; i++) {
            Serial.print("Node ");
            Serial.print(commState.calibMatrix.nodeIds[i]);
            Serial.print(": ");
            Serial.print(commState.calibMatrix.externalLight[i], 2);
            Serial.println(" lux");
        }
        Serial.println("--------------------------------\n");
        
        critical_section_exit(&commStateLock);
        return true;
    }


    // If we got here, the command wasn't recognized
    return false;
}

/**
 * Handle data query commands (g)
 *
 * @param tokens Command tokens
 * @param numTokens Number of tokens
 * @return true if command was handled, false otherwise
 */
static bool handleDataQueryCommands(char tokens[][TOKEN_MAX_LENGTH], int numTokens)
{
    if (strcmp(tokens[0], "g") == 0)
    {
        if (numTokens < 3)
        {
            Serial.println("err: Missing parameters");
            return true;
        }

        char subCommand[TOKEN_MAX_LENGTH];
        char originalCase[TOKEN_MAX_LENGTH];
        char idx[TOKEN_MAX_LENGTH];
        strcpy(subCommand, tokens[1]);
        strcpy(originalCase, tokens[1]);
        strcpy(idx, tokens[2]);

        int targetNode;
        // Parse node ID with error checking
        if (!parseIntParam(tokens[2], &targetNode) || targetNode < 0 || targetNode > 63)
        {
            Serial.println("err: Invalid node ID");
            return true;
        }

        // Check if we need to forward this command
        if (shouldForwardCommand(targetNode))
        {
            // Map the get command to a CAN query message type (using code 20-32)
            uint8_t queryType = 20; // Default query type

            // Map variable types to query codes
            if (strcmp(originalCase, "V") == 0)
                queryType = 20; // Visibility error
            else if (strcmp(originalCase, "F") == 0)
                queryType = 21; // Flicker
            else if (strcmp(originalCase, "E") == 0)
                queryType = 22; // Energy
            else if (strcmp(subCommand, "u") == 0)
                queryType = 23; // Duty cycle
            else if (strcmp(subCommand, "o") == 0)
                queryType = 24; // Occupancy
            else if (strcmp(subCommand, "a") == 0)
                queryType = 25; // Anti-windup
            else if (strcmp(subCommand, "f") == 0)
                queryType = 26; // Feedback control
            else if (strcmp(subCommand, "r") == 0)
                queryType = 27; // Reference illuminance
            else if (strcmp(subCommand, "y") == 0)
                queryType = 28; // Current illuminance
            else if (strcmp(subCommand, "p") == 0)
                queryType = 29; // Power consumption
            else if (strcmp(subCommand, "t") == 0)
                queryType = 30; // Elapsed time
            else if (strcmp(subCommand, "v") == 0)
                queryType = 31; // LDR voltage
            else if (strcmp(subCommand, "d") == 0)
                queryType = 32; // External illuminance
            else
            {
                Serial.println("err: Unsupported remote variable query");
                return true;
            }

            if (sendControlCommand(targetNode, queryType, 0.0f))
            {
                Serial.print("Query sent to node ");
                Serial.println(targetNode);

                // Instead of waiting in a blocking loop, store the query details
                if (!addPendingQuery(targetNode, queryType, originalCase[0] == 'V' || originalCase[0] == 'F' || originalCase[0] == 'E' ? originalCase : subCommand, idx))
                {
                    Serial.println("err: Too many pending queries");
                }
                return true;
            }
            else
            {
                Serial.print("err: Failed to send query to node ");
                Serial.println(targetNode);
                return true;
            }
        }

        // Handle local queries
        handleLocalQuery(subCommand, originalCase, idx, numTokens, tokens);
        return true;
    }
    return false;
}

/**
 * Handle streaming commands (s, S)
 *
 * @param tokens Command tokens
 * @param numTokens Number of tokens
 * @return true if command was handled, false otherwise
 */
static bool handleStreamingCommands(char tokens[][TOKEN_MAX_LENGTH], int numTokens)
{
    if (strcmp(tokens[0], "s") == 0 || strcmp(tokens[0], "S") == 0)
    {
        if (numTokens < 3)
        {
            Serial.println("Error: Invalid streaming command format");
            Serial.println("Usage: s variable_name node_index");
            return true;
        }

        int targetNode;
        if (!parseIntParam(tokens[2], &targetNode) || targetNode < 0 || targetNode > 63)
        {
            Serial.println("Error: Invalid node index");
            return true;
        }

        const char *varName = tokens[1];
        float varCode;
        if (!mapVariableToCode(varName, &varCode))
        {
            Serial.println("Error: Unknown variable name");
            return true;
        }

        Serial.print("Streaming command for var: ");
        Serial.print(varName);
        Serial.print(", node: ");
        Serial.println(targetNode);

        uint8_t controlType = (tokens[0][0] == 's') ? 11 : 12; // 11=start, 12=stop
        bool isLocalNode = (targetNode == deviceConfig.nodeId);
        bool isStopCommand = (tokens[0][0] == 'S');

        // First update local tracking state
        if (isStopCommand)
        {
            stopStream(varName, targetNode);
            Serial.println("Local tracking stopped");
        }
        else
        {
            startStream(varName, targetNode);
            Serial.println("Local tracking started");
        }

        // Then handle remote node if needed
        if (!isLocalNode)
        {
            Serial.println(isStopCommand ? "Stopping remote stream" : "Starting remote stream");
            if (sendControlCommand(targetNode, controlType, varCode))
            {
                Serial.println("Remote command sent successfully");
            }
            else
            {
                Serial.println("Error sending remote command");
            }
        }

        return true;
    }
    return false;
}

/**
 * Handle CAN network commands (c)
 *
 * @param tokens Command tokens
 * @param numTokens Number of tokens
 * @return true if command was handled, false otherwise
 */
static bool handleCANNetworkCommands(char tokens[][TOKEN_MAX_LENGTH], int numTokens)
{
    if (strcmp(tokens[0], "c") == 0)
    {
        if (numTokens < 2)
        {
            Serial.println("err: Missing subcommand");
            return true;
        }

        if (strcmp(tokens[1], "m") == 0)
        {
            // CAN monitoring
            if (numTokens < 3)
            {
                Serial.println("err: Missing parameters");
                return true;
            }

            int val;
            if (!parseIntParam(tokens[2], &val) || !isInRange(val, 0, 1))
            {
                Serial.println("err: Invalid monitoring value (must be 0=off or 1=on)");
                return true;
            }

            canMonitorEnabled = (val == 1);
            Serial.print("CAN monitoring ");
            Serial.println(canMonitorEnabled ? "enabled" : "disabled");
            Serial.println("ack");
        }
        else if (strcmp(tokens[1], "st") == 0)
        {
            // CAN statistics
            displayCANStatistics();
        }
        else if (strcmp(tokens[1], "r") == 0)
        {
            // Reset CAN statistics
            resetCANStats();
            Serial.println("CAN statistics reset");
            Serial.println("ack");
        }
        else if (strcmp(tokens[1], "sc") == 0)
        {
            // Scan for CAN nodes
            displayDiscoveredNodes();
        }
        else if (strcmp(tokens[1], "l") == 0)
        {
            // Latency test
            measureCANLatency(numTokens, tokens);
        }
        else if (strcmp(tokens[1], "d") == 0)
        {
            displayDiscoveredNodes();
            Serial.println("ack");
        }
        else
        {
            Serial.println("err: Unknown CAN subcommand");
        }
        return true;
    }
    return false;
}

/**
 * Handle controller parameter commands (k)
 *
 * @param tokens Command tokens
 * @param numTokens Number of tokens
 * @return true if command was handled, false otherwise
 */
static bool handleControllerParameterCommands(char tokens[][TOKEN_MAX_LENGTH], int numTokens)
{
    if (strcmp(tokens[0], "k") == 0)
    {
        if (numTokens < 4)
        {
            Serial.println("err: Missing parameters");
            return true;
        }

        int targetNode;
        float value;

        if (!parseIntParam(tokens[1], &targetNode) || targetNode < 0 || targetNode > 63)
        {
            Serial.println("err: Invalid node ID");
            return true;
        }

        char param[TOKEN_MAX_LENGTH];
        strcpy(param, tokens[2]);

        if (!parseFloatParam(tokens[3], &value))
        {
            Serial.println("err: Invalid parameter value");
            return true;
        }

        // Check if we need to forward this command
        if (shouldForwardCommand(targetNode))
        {
            // Map parameter to control code for CAN
            uint8_t paramCode;
            if (strcmp(param, "b") == 0)
                paramCode = 15;
            else if (strcmp(param, "k") == 0)
                paramCode = 16;
            else
            {
                Serial.println("err: Unknown parameter (must be 'b' or 'k')");
                return true;
            }

            if (sendControlCommand(targetNode, paramCode, value))
            {
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: CAN forwarding failed");
            }
            return true;
        }

        // Handle locally
        if (strcmp(param, "b") == 0)
        {
            if (!isInRange(value, 0.0f, 1.0f))
            {
                Serial.println("err: Beta must be between 0.0 and 1.0");
                return true;
            }
            pid.setWeighting(value);
        }
        else if (strcmp(param, "k") == 0)
        {
            pid.setGains(value, value);
        }
        else
        {
            Serial.println("err: Unknown parameter (must be 'b' or 'k')");
            return true;
        }
        Serial.println("ack");
        return true;
    }
    return false;
}

/**
 * Handle node disable command
 *
 * @param tokens Command tokens
 * @param numTokens Number of tokens
 * @return true if command was handled, false otherwise
 */
static bool handleDisableCommand(char tokens[][TOKEN_MAX_LENGTH], int numTokens)
{
    if (strcmp(tokens[0], "disable") == 0)
    {
        if (numTokens < 2)
        {
            Serial.println("err: Expected node ID");
            return true;
        }

        int targetNode;
        if (!parseIntParam(tokens[1], &targetNode) || targetNode < 0 || targetNode > 63)
        {
            Serial.println("err: Invalid node ID");
            return true;
        }

        // Define a special control code for disable command (15)
        const uint8_t CONTROL_TYPE_DISABLE = 15;

        if (shouldForwardCommand(targetNode))
        {
            // Forward to specific node
            if (sendControlCommand(targetNode, CONTROL_TYPE_DISABLE, 0.0f))
            {
                Serial.println("ack: Node disable command sent");
            }
            else
            {
                Serial.println("err: CAN forwarding failed");
            }
            return true;
        }

        // Handle locally
        critical_section_enter_blocking(&commStateLock);
        // Set setpoint to zero
        controlState.setpointLux = 0.0f;
        // Set luminaire state to OFF
        controlState.luminaireState = STATE_OFF;
        // Disable CAN communication
        commState.periodicCANEnabled = false;
        critical_section_exit(&commStateLock);

        // Turn off LED
        setLEDDutyCycle(0.0f);

        Serial.println("ack: Node disabled");
        return true;
    }
    return false;
}

//==========================================================================================================================================================
// COMMAND PROCESSING PIPELINE
//==========================================================================================================================================================

/**
 * Process a single command line from Serial
 * Parses and executes commands for control, metrics, and CAN operations
 *
 * @param cmdLine The command string to process
 */
static void processCommandLine(const char *cmdLine)
{
    char tokens[MAX_TOKENS][TOKEN_MAX_LENGTH];
    int numTokens = prepareCommand(cmdLine, tokens);

    if (numTokens == 0)
    {
        return;
    }

    // Try each command category handler in sequence
    if (handleLEDCommands(tokens, numTokens))
    {
        return;
    }

    else if (handleSystemStateCommands(tokens, numTokens))
    {
        return;
    }

    else if (handleDataQueryCommands(tokens, numTokens))
    {
        return;
    }

    else if (handleStreamingCommands(tokens, numTokens))
    {
        return;
    }

    else if (handleCANNetworkCommands(tokens, numTokens))
    {
        return;
    }

    else if (handleControllerParameterCommands(tokens, numTokens))
    {
        return;
    }
    else if (handleDisableCommand(tokens, numTokens))
    {
        return;
    }
    // Help command
    else if (strcmp(tokens[0], "h") == 0)
    {
        printHelp();
        return;
    }
    else
    {
        Serial.print("err: Unknown command '");
        Serial.print(tokens[0]);
        Serial.println("'");
    }

    // Default response for unrecognized commands
    Serial.println("ack");
}

/**
 * Process any pending serial commands
 * This function should be called regularly in the main loop
 */
void processSerialCommands()
{
    static char inputBuffer[CMD_MAX_LENGTH];
    static int bufferPos = 0;

    // Process any pending queries first
    processPendingQueries();

    while (Serial.available() > 0)
    {
        char c = Serial.read();

        // Process complete line when newline is received
        if (c == '\n' || c == '\r')
        {
            if (bufferPos > 0)
            {
                inputBuffer[bufferPos] = '\0';
                processCommandLine(inputBuffer);
                bufferPos = 0;
            }
        }
        // Add character to buffer if space available
        else if (bufferPos < CMD_MAX_LENGTH - 1)
        {
            inputBuffer[bufferPos++] = c;
        }
    }
}

//==========================================================================================================================================================
// USER INTERFACE
//==========================================================================================================================================================

/**
 * Print a comprehensive list of all available commands
 * Organizes commands by category and provides descriptions
 */
void printHelp()
{
    Serial.println("\n===== Distributed Lighting Control System Commands =====\n");

    Serial.println("------- LED CONTROL -------");
    Serial.println("u <i> <val>  : Set duty cycle (0.0-1.0) for node i");
    Serial.println("p <i> <val>  : Set brightness percentage (0-100%) for node i");
    Serial.println("w <i> <val>  : Set power in watts for node i");

    Serial.println("\n------- CONTROLLER PARAMETERS -------");
    Serial.println("k <i> beta <val>: Set setpoint weighting factor (0.0-1.0) for node i");
    Serial.println("k <i> k <val>   : Set both proportional and integral gain for node i");

    Serial.println("\n------- SYSTEM STATE -------");
    Serial.println("o <i> <val>  : Set occupancy (0=unoccupied, 1=occupied) for node i");
    Serial.println("a <i> <val>  : Set anti-windup (0=off, 1=on) for node i");
    Serial.println("f <i> <val>  : Set feedback control (0=off, 1=on) for node i");
    Serial.println("fi <i> <val> : Set sensor filtering (0=off, 1=on) for node i");
    Serial.println("r <i> <val>  : Set illuminance reference (lux) for node i");
    Serial.println("st <i> <state>: Set luminaire state (off/unoccupied/occupied) for node i");

    Serial.println("\n------- DATA STREAMING -------");
    Serial.println("s <x> <i>    : Start streaming variable x from node i");
    Serial.println("S <x> <i>    : Stop streaming variable x from node i");
    Serial.println("  Variables: y=illuminance, u=duty, p=power, o=occupancy,");
    Serial.println("             a=anti-windup, f=feedback, r=reference,");
    Serial.println("             v=LDR voltage, d=external illuminance,");
    Serial.println("             t=elapsed time, V=visibility error,");
    Serial.println("             F=flicker, E=energy");

    Serial.println("\n------- DATA QUERIES -------");
    Serial.println("g <var> <i>  : Get value of variable <var> from node i");
    Serial.println("  Variables: Same as streaming, plus:");
    Serial.println("  g b <x> <i>: Get buffer history of variable x from node i");
    Serial.println("g bigdump <i>: Get complete time series data (timestamp, lux, duty, setpoint)");
    Serial.println("g mdump <i>  : Get comprehensive metrics (timestamp, lux, duty, setpoint, flickers, energy, visibility)");

    Serial.println("\n------- CAN NETWORK -------");
    Serial.println("c m <0|1>    : Disable/enable CAN message monitoring");
    Serial.println("c st         : Display CAN communication statistics");
    Serial.println("c r          : Reset CAN statistics counters");
    Serial.println("c sc         : Scan for active nodes on the CAN network");
    Serial.println("c l <i> <n>  : Measure round-trip latency to node i (n samples)");
    Serial.println("c d          : Display discovered nodes and their status");

    Serial.println("\n------- SYSTEM -------");
    Serial.println("h            : Display this help information");

    Serial.println("\n------- DATA FORMAT NOTES -------");
    Serial.println("Energy      : Cumulative energy consumption in joules");
    Serial.println("Visibility  : Cumulative visibility error when lux < setpoint");
    Serial.println("FlickerInstant: Instantaneous flicker value");
    Serial.println("FlickerWithFilter: Cumulative flicker with filtering");
    Serial.println("FlickerNoFilter : Cumulative flicker without filtering");

    Serial.println("\nNode addressing:");
    Serial.println("  i = 0      : Broadcast to all nodes");
    Serial.println("  i = n      : Address specific node n");
    Serial.println("  Use this node's ID for local commands\n");

    Serial.println("======================================================\n");
}