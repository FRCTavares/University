#include "CommandInterface.h"
#include "CANComm.h"
#include <Arduino.h>
#include <math.h>
#include "Globals.h"
#include "PIController.h"

static void printHelp();

// Maximum command line length
#define CMD_MAX_LENGTH 64
// Maximum number of tokens in a command
#define MAX_TOKENS 6
// Maximum length of a single token
#define TOKEN_MAX_LENGTH 16
// Maximum number of pending queries that can be tracked
#define MAX_PENDING_QUERIES 10

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

// Array of pending queries
static PendingQuery pendingQueries[MAX_PENDING_QUERIES];

// Initialize the pending queries array
static void initPendingQueries()
{
  for (int i = 0; i < MAX_PENDING_QUERIES; i++)
  {
    pendingQueries[i].active = false;
  }
}

static bool addPendingQuery(uint8_t targetNode, uint8_t queryType, const char *cmd, const char *index)
{
  // Find an empty slot
  for (int i = 0; i < MAX_PENDING_QUERIES; i++)
  {
    if (!pendingQueries[i].active)
    {
      pendingQueries[i].active = true;
      pendingQueries[i].targetNode = targetNode;
      pendingQueries[i].queryType = queryType;
      strncpy(pendingQueries[i].originalCommand, cmd, sizeof(pendingQueries[i].originalCommand) - 1);
      pendingQueries[i].originalCommand[sizeof(pendingQueries[i].originalCommand) - 1] = '\0';
      strncpy(pendingQueries[i].displayIndex, index, sizeof(pendingQueries[i].displayIndex) - 1);
      pendingQueries[i].displayIndex[sizeof(pendingQueries[i].displayIndex) - 1] = '\0';
      pendingQueries[i].timeoutTime = millis() + 500; // 500ms timeout
      return true;
    }
  }
  return false; // No slots available
}

static void processPendingQueries()
{
  // Check each active query
  for (int i = 0; i < MAX_PENDING_QUERIES; i++)
  {
    if (pendingQueries[i].active)
    {
      // Check for timeout
      if (millis() > pendingQueries[i].timeoutTime)
      {
        Serial.print("err: No response from node ");
        Serial.println(pendingQueries[i].targetNode);
        pendingQueries[i].active = false;
        continue;
      }

      // Check for a response
      can_frame frame;
      if (readCANMessage(&frame) == MCP2515::ERROR_OK)
      {
        // Parse message details
        uint8_t msgType, destAddr;
        parseCANId(frame.can_id, msgType, destAddr);
        uint8_t senderNodeID = frame.data[0];

        // Check if this is a response for our query
        if (msgType == CAN_TYPE_RESPONSE &&
            senderNodeID == pendingQueries[i].targetNode &&
            (destAddr == nodeID || destAddr == CAN_ADDR_BROADCAST))
        {

          // Extract the float value
          float value = bytesToFloat(&frame.data[2]);

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

          // Format the value based on command type
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
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
// COMMAND PARSING HELPERS
//-----------------------------------------------------------------------------

/**
 * Split a command line into space-separated tokens using char arrays
 *
 * @param cmd The command string to parse
 * @param tokens Array of char arrays to store tokens
 * @param maxTokens Maximum number of tokens to extract
 * @return Number of tokens found
 */
static int parseTokensChar(const char *cmd, char tokens[][TOKEN_MAX_LENGTH], int maxTokens)
{
  int numTokens = 0;
  int cmdLen = strlen(cmd);
  int tokenStart = 0;
  int tokenPos = 0;

  // Skip leading spaces
  while (tokenStart < cmdLen && cmd[tokenStart] == ' ')
  {
    tokenStart++;
  }

  for (int i = tokenStart; i <= cmdLen && numTokens < maxTokens; i++)
  {
    // Check for token delimiter (space or end of string)
    if (i == cmdLen || cmd[i] == ' ')
    {
      if (tokenPos > 0)
      { // We have a complete token
        // Copy token and add null terminator
        tokens[numTokens][tokenPos] = '\0';
        numTokens++;
        tokenPos = 0;

        // Skip consecutive spaces
        while (i < cmdLen && cmd[i + 1] == ' ')
        {
          i++;
        }
      }
    }
    else if (tokenPos < TOKEN_MAX_LENGTH - 1)
    {
      // Add character to current token
      tokens[numTokens][tokenPos++] = cmd[i];
    }
  }

  return numTokens;
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
  if (targetNode == 0 || targetNode == nodeID)
  {
    return false;
  }
  // Otherwise, forward to target node
  return true;
}

//-----------------------------------------------------------------------------
// COMMAND PROCESSING
//-----------------------------------------------------------------------------
/**
 * Case-insensitive string comparison
 *
 * @param str1 First string to compare
 * @param str2 Second string to compare
 * @return 0 if equal, non-zero otherwise (like strcmp)
 */
static int strcmpIgnoreCase(const char *str1, const char *str2)
{
  while (*str1 && *str2)
  {
    char c1 = *str1 >= 'A' && *str1 <= 'Z' ? *str1 + 32 : *str1;
    char c2 = *str2 >= 'A' && *str2 <= 'Z' ? *str2 + 32 : *str2;

    if (c1 != c2)
    {
      return c1 - c2;
    }

    str1++;
    str2++;
  }

  return *str1 - *str2;
}

/**
 * Process a single command line from Serial
 * Parses and executes commands for control, metrics, and CAN operations
 *
 * @param cmdLine The command string to process
 */
static void processCommandLine(const char *cmdLine)
{
  // Buffer to store trimmed command
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
    return;
  }

  // Tokenize the command line
  char tokens[MAX_TOKENS][TOKEN_MAX_LENGTH];
  int numTokens = parseTokensChar(trimmedCmd, tokens, MAX_TOKENS);

  if (numTokens == 0)
  {
    return;
  }

  //-----------------------------------------------------------------------------
  // LED CONTROL COMMANDS
  //-----------------------------------------------------------------------------

  // "u <i> <val>" => set duty cycle (0.0-1.0)
  if (strcmp(tokens[0], "u") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = atoi(tokens[1]);
    float val = strtof(tokens[2], NULL);

    if (val < 0.0f || val > 1.0f)
    {
      Serial.println("err");
      return;
    }

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Forward to specific node - control type 4 = duty cycle
      if (sendControlCommand(targetNode, 4, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle broadcast case (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 4, val))
      {
        // Also apply locally since broadcast includes this node
        setLEDDutyCycle(val);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    // Apply locally
    setLEDDutyCycle(val);
    Serial.println("ack");
    return;
  }

  // "p <i> <percentage>" => set LED by percentage (0-100%)
  else if (strcmp(tokens[0], "p") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = atoi(tokens[1]);
    float val = strtof(tokens[2], NULL);

    if (val < 0.0f || val > 100.0f)
    {
      Serial.println("err");
      return;
    }

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Forward to specific node - control type 5 = percentage
      if (sendControlCommand(targetNode, 5, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle broadcast case (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 5, val))
      {
        // Also apply locally since broadcast includes this node
        setLEDPercentage(val); // FIXED: Use percentage function instead of duty cycle
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    // Apply locally
    setLEDPercentage(val);
    Serial.println("ack");
    return;
  }

  // "w <i> <watts>" => set LED by power in watts
  else if (strcmp(tokens[0], "w") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = atoi(tokens[1]);
    float val = strtof(tokens[2], NULL);

    if (val < 0.0f || val > MAX_POWER_WATTS)
    {
      Serial.println("err");
      return;
    }

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Forward to specific node - control type 6 = power in watts
      if (sendControlCommand(targetNode, 6, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle broadcast case (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 6, val))
      {
        // Also apply locally since broadcast includes this node
        setLEDPower(val);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    setLEDPower(val);
    Serial.println("ack");
    return;
  }

  //-----------------------------------------------------------------------------
  // SYSTEM STATE CONTROL COMMANDS
  //-----------------------------------------------------------------------------

  // "o <i> <val>" => set occupancy state (0=off, 1=unoccupied, 2=occupied)

  else if (strcmp(tokens[0], "o") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = atoi(tokens[1]);
    float val = strtof(tokens[2], NULL);

    if (val < 0 || val > 2) // Allow values 0, 1, and 2
    {
      Serial.println("err");
      return;
    }

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Forward to specific node - control type 7 = occupancy
      if (sendControlCommand(targetNode, 7, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle broadcast case (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 7, val))
      {
        // Also apply locally since broadcast includes this node
        critical_section_enter(&commStateLock);
        controlState.luminaireState = static_cast<LuminaireState>(val); // Direct casting
        critical_section_exit(&commStateLock);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    // Apply locally
    critical_section_enter(&commStateLock);
    controlState.luminaireState = static_cast<LuminaireState>(val); // Direct casting
    critical_section_exit(&commStateLock);
    Serial.println("ack");
    return;
  }

  // "a <i> <val>" => set anti-windup on/off (0=off, 1=on)
  else if (strcmp(tokens[0], "a") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = atoi(tokens[1]);
    int val = atoi(tokens[2]); // Changed from strtof to atoi

    if (val != 0 && val != 1)
    {
      Serial.println("err");
      return;
    }

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Forward to specific node - control type 8 = anti-windup
      if (sendControlCommand(targetNode, 8, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle broadcast case (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 8, val))
      {
        // Also apply locally since broadcast includes this node
        critical_section_enter(&commStateLock);
        controlState.antiWindup = (val == 1);
        critical_section_exit(&commStateLock);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }
    critical_section_enter(&commStateLock);
    controlState.antiWindup = (val == 1);
    critical_section_exit(&commStateLock);
    Serial.println("ack");
    return;
  }
  // "fi <i> <val>" => set filter enable/disable (0=off, 1=on)
  else if (strcmp(tokens[0], "fi") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = atoi(tokens[1]);
    int val = atoi(tokens[2]); // Changed from strtof to atoi

    if (val != 0 && val != 1)
    {
      Serial.println("err");
      return;
    }

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Forward to specific node - control type 14 = filter enable/disable
      if (sendControlCommand(targetNode, 14, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle broadcast case (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 14, val))
      {
        // Also apply locally since broadcast includes this node
        critical_section_enter(&commStateLock);
        sensorState.filterEnabled = (val == 1);
        critical_section_exit(&commStateLock);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    critical_section_enter(&commStateLock);
    sensorState.filterEnabled = (val == 1);
    critical_section_exit(&commStateLock);
    Serial.println("ack");
    return;
  }

  // "f <i> <val>" => set feedback control on/off (0=off, 1=on)
  else if (strcmp(tokens[0], "f") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = atoi(tokens[1]);
    int val = atoi(tokens[2]); // Changed from strtof to atoi

    if (val != 0 && val != 1)
    {
      Serial.println("err");
      return;
    }

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Forward to specific node - control type 9 = feedback control
      if (sendControlCommand(targetNode, 9, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle broadcast case (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 9, val))
      {
        // Also apply locally since broadcast includes this node
        critical_section_enter(&commStateLock);
        controlState.feedbackControl = (val == 1);
        critical_section_exit(&commStateLock);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }
    critical_section_enter(&commStateLock);
    controlState.feedbackControl = (val == 1);
    critical_section_exit(&commStateLock);
    Serial.println("ack");
    return;
  }

  // "r <i> <val>" => set illuminance reference (lux)
  else if (strcmp(tokens[0], "r") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = atoi(tokens[1]);
    float val = strtof(tokens[2], NULL);

    if (val < 0.0f || val > MAX_ILLUMINANCE)
    {
      Serial.println("err");
      return;
    }

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Forward to specific node - control type 10 = reference illuminance
      if (sendControlCommand(targetNode, 10, val))
      {
        Serial.print("Query request sent to node ");
        Serial.print(targetNode);
        Serial.print(": ");
        Serial.println(val);

        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle broadcast case (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 10, val))
      {
        // Also apply locally since broadcast includes this node
        critical_section_enter(&commStateLock);
        controlState.setpointLux = val;
        critical_section_exit(&commStateLock);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }
    critical_section_enter(&commStateLock);
    controlState.setpointLux = val;
    critical_section_exit(&commStateLock);
    Serial.println("ack");
    return;
  }
  // "h" => Print help information
  else if (strcmp(tokens[0], "h") == 0)
  {
    printHelp();
    return;
  }

  //-----------------------------------------------------------------------------
  // DATA STREAMING COMMANDS
  //-----------------------------------------------------------------------------

  // "s <x> <i>" => start stream of real-time variable <x> for desk <i>
  else if (strcmp(tokens[0], "s") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    char var[TOKEN_MAX_LENGTH];
    strcpy(var, tokens[1]); // Variable name is in tokens[1]

    uint8_t targetNode = atoi(tokens[2]); // Target node is in tokens[2]

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Map variable names to numeric codes for CAN transmission
      float varCode = 0; // Default for 'y' (lux)

      if (strcmpIgnoreCase(var, "u") == 0)
        varCode = 1; // Duty cycle
      else if (strcmpIgnoreCase(var, "p") == 0)
        varCode = 2; // Power percentage
      else if (strcmpIgnoreCase(var, "o") == 0)
        varCode = 3; // Occupancy
      else if (strcmpIgnoreCase(var, "a") == 0)
        varCode = 4; // Anti-windup
      else if (strcmpIgnoreCase(var, "f") == 0)
        varCode = 5; // Feedback control
      else if (strcmpIgnoreCase(var, "r") == 0)
        varCode = 6; // Reference illuminance
      else if (strcmpIgnoreCase(var, "y") == 0)
        varCode = 0; // Illuminance (lux)
      else if (strcmpIgnoreCase(var, "v") == 0)
        varCode = 7; // LDR voltage
      else if (strcmpIgnoreCase(var, "d") == 0)
        varCode = 8; // External illuminance
      else if (strcmpIgnoreCase(var, "t") == 0)
        varCode = 9; // Elapsed time
      else if (strcmpIgnoreCase(var, "V") == 0)
        varCode = 10; // Visibility error
      else if (strcmpIgnoreCase(var, "F") == 0)
        varCode = 11; // Flicker
      else if (strcmpIgnoreCase(var, "E") == 0)
        varCode = 12; // Energy

      if (sendControlCommand(targetNode, 11, varCode))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }
    // Handle locally
    startStream(var, targetNode);
    Serial.println("ack");
    return;
  }

  // "S <x> <i>" => stop stream of variable <x> for desk <i>
  else if (strcmp(tokens[0], "S") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    char var[TOKEN_MAX_LENGTH];
    strcpy(var, tokens[1]); // Variable name is in tokens[1]

    uint8_t targetNode = atoi(tokens[2]); // Target node is in tokens[2]

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Map variable names to numeric codes for CAN transmission
      float varCode = 0; // Default for 'y' (lux)

      if (strcmpIgnoreCase(var, "u") == 0)
        varCode = 1; // Duty cycle
      else if (strcmpIgnoreCase(var, "p") == 0)
        varCode = 2; // Power percentage
      else if (strcmpIgnoreCase(var, "o") == 0)
        varCode = 3; // Occupancy
      else if (strcmpIgnoreCase(var, "a") == 0)
        varCode = 4; // Anti-windup
      else if (strcmpIgnoreCase(var, "f") == 0)
        varCode = 5; // Feedback control
      else if (strcmpIgnoreCase(var, "r") == 0)
        varCode = 6; // Reference illuminance
      else if (strcmpIgnoreCase(var, "y") == 0)
        varCode = 0; // Illuminance (lux)
      else if (strcmpIgnoreCase(var, "v") == 0)
        varCode = 7; // LDR voltage
      else if (strcmpIgnoreCase(var, "d") == 0)
        varCode = 8; // External illuminance
      else if (strcmpIgnoreCase(var, "t") == 0)
        varCode = 9; // Elapsed time
      else if (strcmpIgnoreCase(var, "V") == 0)
        varCode = 10; // Visibility error
      else if (strcmpIgnoreCase(var, "F") == 0)
        varCode = 11; // Flicker
      else if (strcmpIgnoreCase(var, "E") == 0)
        varCode = 12; // Energy

      if (sendControlCommand(targetNode, 12, varCode))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle locally
    stopStream(var, targetNode);
    Serial.println("ack");
    return;
  }

  //-----------------------------------------------------------------------------
  // METRICS AND DATA QUERY COMMANDS
  //-----------------------------------------------------------------------------

  // "g <var> <i>" => Get value of variable <var> from node <i>
  else if (strcmp(tokens[0], "g") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    char subCommand[TOKEN_MAX_LENGTH];
    char originalCase[TOKEN_MAX_LENGTH];
    char idx[TOKEN_MAX_LENGTH];
    strcpy(subCommand, tokens[1]);
    strcpy(originalCase, tokens[1]);
    strcpy(idx, tokens[2]);

    uint8_t targetNode = atoi(tokens[2]);

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
        return;
      }

      // Send the query to the remote node
      if (sendControlCommand(targetNode, queryType, 0.0f))
      {
        Serial.print("Query sent to node ");
        Serial.println(targetNode);

        // Wait for response with timeout
        unsigned long timeout = millis() + 500; // 500ms timeout
        bool responseReceived = false;

        if (sendControlCommand(targetNode, queryType, 0.0f))
        {
          Serial.print("Query sent to node ");
          Serial.println(targetNode);

          // Instead of waiting in a blocking loop, store the query details
          if (!addPendingQuery(targetNode, queryType, originalCase[0] == 'V' || originalCase[0] == 'F' || originalCase[0] == 'E' ? originalCase : subCommand, idx))
          {
            Serial.println("err: Too many pending queries");
          }
          return;
        }
        else
        {
          Serial.print("err: Failed to send query to node ");
          Serial.println(targetNode);
          return;
        }

        if (!responseReceived)
        {
          Serial.print("err: No response from node ");
          Serial.println(targetNode);
        }
      }
      else
      {
        Serial.print("err: Failed to send query to node ");
        Serial.println(targetNode);
      }
      return;
    }

    // Handle local metric queries

    // Quality metrics
    // "g V <i>" => "V <i> <val>" (Visibility error metric)
    if (strcmp(originalCase, "V") == 0)
    {
      float V = computeVisibilityErrorFromBuffer();
      Serial.print("V ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(V, 2);
      return;
    }
    // "g F <i>" => "F <i> <val>" (Flicker metric)
    else if (strcmp(originalCase, "F") == 0)
    {
      float F = computeFlickerFromBuffer();
      Serial.print("F ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(F, 4);
      return;
    }
    // "g E <i>" => "E <i> <val>" (Energy metric)
    else if (strcmp(originalCase, "E") == 0)
    {
      float E = computeEnergyFromBuffer();
      Serial.print("E ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(E, 4);
      return;
    }

    // Control system variables
    // "g u <i>" => "u <i> <val>" (duty cycle)
    if (strcmp(subCommand, "u") == 0)
    {
      Serial.print("u ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(dutyCycle, 4);
      return;
    }
    // "g o <i>" => "o <i> <val>" (occupancy)
    else if (strcmp(subCommand, "o") == 0)
    {
      critical_section_enter(&commStateLock);
      int occVal = static_cast<int>(controlState.luminaireState);
      critical_section_exit(&commStateLock);
      Serial.print("o ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(occVal);
      return;
    }
    // "g a <i>" => "a <i> <val>" (anti-windup)
    else if (strcmp(subCommand, "a") == 0)
    {
      critical_section_enter(&commStateLock);
      int awVal = controlState.antiWindup ? 1 : 0;
      critical_section_exit(&commStateLock);
      Serial.print("a ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(awVal);
      return;
    }
    // "g f <i>" => "f <i> <val>" (feedback control)
    else if (strcmp(subCommand, "f") == 0)
    {
      critical_section_enter(&commStateLock);
      int fbVal = controlState.feedbackControl ? 1 : 0;
      critical_section_exit(&commStateLock);
      Serial.print("f ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(fbVal);
      return;
    }
    // "g r <i>" => "r <i> <val>" (reference illuminance)
    else if (strcmp(subCommand, "r") == 0)
    {
      Serial.print("r ");
      Serial.print(idx);
      Serial.print(" ");
      critical_section_enter(&commStateLock);
      Serial.println(controlState.setpointLux, 4);
      critical_section_exit(&commStateLock);
      return;
    }
    // "g y <i>" => "y <i> <val>" (current illuminance)
    else if (strcmp(subCommand, "y") == 0)
    {
      float lux = readLux();
      Serial.print("y ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(lux, 2);
      return;
    }

    // Sensor measurements
    // "g v <i>" => measure voltage level at LDR => "v <i> <val>"
    else if (strcmp(subCommand, "v") == 0)
    {
      float vLdr = getVoltageAtLDR();
      Serial.print("v ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(vLdr, 3);
      return;
    }
    // "g d <i>" => external illuminance => "d <i> <val>"
    else if (strcmp(subCommand, "d") == 0)
    {
      float dVal = getExternalIlluminance();
      Serial.print("d ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(dVal, 2);
      return;
    }
    // "g p <i>" => instantaneous power => "p <i> <val>"
    else if (strcmp(subCommand, "p") == 0)
    {
      float pVal = getPowerConsumption();
      Serial.print("p ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(pVal, 2);
      return;
    }
    // "g t <i>" => elapsed time => "t <i> <val>"
    else if (strcmp(subCommand, "t") == 0)
    {
      unsigned long sec = getElapsedTime();
      Serial.print("t ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(sec);
      return;
    }

    // Historical data buffer
    // "g b <x> <i>" => "b <x> <i> <val1>,<val2>..."
    else if (strcmp(subCommand, "b") == 0)
    {
      if (numTokens < 4)
      {
        Serial.println("err");
        return;
      }
      char xVar[TOKEN_MAX_LENGTH];
      strcpy(xVar, tokens[2]);
      int iDesk = atoi(tokens[3]);
      char bufferData[1024]; // Adjust size as needed
      getLastMinuteBuffer(xVar, iDesk, bufferData, sizeof(bufferData));
      Serial.print("b ");
      Serial.print(xVar);
      Serial.print(" ");
      Serial.print(iDesk);
      Serial.print(" ");
      Serial.println(bufferData);
      return;
    }
    else if (strcmp(subCommand, "bigdump") == 0)
    {
      int iDesk = atoi(tokens[2]);

      // Get the log buffer and count
      LogEntry *logBuffer = getLogBuffer();
      int count = getLogCount();
      if (count == 0)
      {
        Serial.println("No data in buffer");
        return;
      }

      // Calculate starting index
      int startIndex = isBufferFull() ? getCurrentIndex() : 0;

      // Dump time values
      Serial.print("Time: ");
      for (int i = 0; i < count; i++)
      {
        int realIndex = (startIndex + i) % LOG_SIZE;
        Serial.print(logBuffer[realIndex].timestamp);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      // Dump measured lux values
      Serial.print("MeasuredLux: ");
      for (int i = 0; i < count; i++)
      {
        int realIndex = (startIndex + i) % LOG_SIZE;
        Serial.print(logBuffer[realIndex].lux, 2);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      // Dump duty cycle values
      Serial.print("DutyCycle: ");
      for (int i = 0; i < count; i++)
      {
        int realIndex = (startIndex + i) % LOG_SIZE;
        Serial.print(logBuffer[realIndex].duty, 4);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      Serial.print("SetpointLux: ");
      for (int i = 0; i < count; i++)
      {
        int realIndex = (startIndex + i) % LOG_SIZE;
        Serial.print(logBuffer[realIndex].setpoint, 2);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      // Confirmation
      Serial.println("Data dump complete");
      return;
    }
    else if (strcmp(subCommand, "mdump") == 0)
    {
      int count = getLogCount();
      if (count == 0)
      {
        Serial.println("No data available for metrics");
        return;
      }

      // Get access to the log buffer
      LogEntry *buffer = getLogBuffer();
      int startIndex = isBufferFull() ? getCurrentIndex() : 0;

      // Print Time row
      Serial.print("Time: ");
      for (int i = 0; i < count; i++)
      {
        int idx = (startIndex + i) % LOG_SIZE;
        Serial.print(buffer[idx].timestamp);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      // Measured Lux row
      Serial.print("MeasuredLux: ");
      for (int i = 0; i < count; i++)
      {
        int idx = (startIndex + i) % LOG_SIZE;
        Serial.print(buffer[idx].lux, 2);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      // DutyCycle row
      Serial.print("DutyCycle: ");
      for (int i = 0; i < count; i++)
      {
        int idx = (startIndex + i) % LOG_SIZE;
        Serial.print(buffer[idx].duty, 4);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      // Setpoint row
      Serial.print("SetpointLux: ");
      for (int i = 0; i < count; i++)
      {
        int idx = (startIndex + i) % LOG_SIZE;
        Serial.print(buffer[idx].setpoint, 2);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      // Flicker row (instantaneous flicker error)
      Serial.print("Flicker: ");
      for (int i = 0; i < count; i++)
      {
        int idx = (startIndex + i) % LOG_SIZE;
        Serial.print(buffer[idx].flicker, 6);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      // Accumulated energy row
      Serial.print("Energy: ");
      float energySum = 0.0f;
      for (int i = 0; i < count; i++)
      {
        int idx = (startIndex + i) % LOG_SIZE;
        // Calculate time step (dt) in seconds
        float dt = 0.0f;
        if (i > 0)
        {
          int prevIdx = (startIndex + i - 1) % LOG_SIZE;
          dt = (buffer[idx].timestamp - buffer[prevIdx].timestamp) / 1000.0f;
        }
        // Power = MAX_POWER_WATTS * duty
        float power = MAX_POWER_WATTS * buffer[idx].duty;
        energySum += power * dt;
        Serial.print(energySum, 4);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      // Accumulated visibility error row
      Serial.print("VisibilityError: ");
      float visSum = 0.0f;
      for (int i = 0; i < count; i++)
      {
        int idx = (startIndex + i) % LOG_SIZE;
        float dt = 0.0f;
        if (i > 0)
        {
          int prevIdx = (startIndex + i - 1) % LOG_SIZE;
          dt = (buffer[idx].timestamp - buffer[prevIdx].timestamp) / 1000.0f;
        }
        // Only add error when measured < setpoint
        float err = 0.0f;
        if (buffer[idx].lux < buffer[idx].setpoint)
        {
          err = (buffer[idx].setpoint - buffer[idx].lux) * dt;
        }
        visSum += err;
        Serial.print(visSum, 2);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      // Jitter row (in microseconds)
      // -------------------------------------------------------------------------
      Serial.print("Jitter_us: ");
      for (int i = 0; i < count; i++)
      {
        int idx = (startIndex + i) % LOG_SIZE;
        Serial.print(buffer[idx].jitter / 1000, 4); // e.g. 4 decimals
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      // External Illuminance row
      Serial.print("ExternalLux: ");
      for (int i = 0; i < count; i++)
      {
        int idx = (startIndex + i) % LOG_SIZE;
        Serial.print(buffer[idx].extLux, 2);
        if (i < count - 1)
          Serial.print(",");
      }
      Serial.println();

      Serial.println("Data dump complete (mdump, transposed).");
      return;
    }
    else
    {
      Serial.println("err: Unknown metric or variable");
      return;
    }
  }
  //-----------------------------------------------------------------------------
  // CONTROLLER PARAMETER COMMANDS
  //-----------------------------------------------------------------------------

  // "k <i> <param> <value>" => set controller parameter
  else if (strcmp(tokens[0], "k") == 0)
  {
    if (numTokens < 4)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = atoi(tokens[1]);
    char param[TOKEN_MAX_LENGTH];
    strcpy(param, tokens[2]);
    float value = strtof(tokens[3], NULL);

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Map parameter to control code for CAN
      uint8_t paramCode = 0;
      if (strcmp(tokens[2], "b") == 0) // Correct token index
        paramCode = 15;
      else if (strcmp(tokens[2], "k") == 0) // Correct token index
        paramCode = 16;
      else
      {
        Serial.println("err: unknown parameter");
        return;
      }

      if (sendControlCommand(targetNode, paramCode, value))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle locally
    if (strcmp(tokens[2], "b") == 0)
    {
      if (value < 0.0 || value > 1.0)
      {
        Serial.println("err: beta must be between 0.0 and 1.0");
        return;
      }
      pid.setWeighting(value);
      Serial.println("ack");
    }
    else if (strcmp(tokens[2], "k") == 0) // Changed from "kp" to "k"
    {
      // Set both gains to the same value
      pid.setGains(value, value);
      Serial.println("ack");
    }
    else
    {
      Serial.println("err: unknown parameter");
    }
    return;
  }

  //-----------------------------------------------------------------------------
  // CAN NETWORK COMMANDS
  //-----------------------------------------------------------------------------

  // "c <subcommand> [params]" => CAN-related commands
  else if (strcmp(tokens[0], "c") == 0)
  {
    // "c m <0|1>" => Enable/disable CAN message monitoring
    if (strcmp(tokens[1], "m") == 0)
    {
      if (numTokens < 3)
      {
        Serial.println("err");
        return;
      }

      canMonitorEnabled = (atoi(tokens[2]) == 1);

      Serial.print("CAN monitoring ");
      Serial.println(canMonitorEnabled ? "enabled" : "disabled");
      Serial.println("ack");
      return;
    }

    // "c st" => Display CAN communication statistics
    else if (strcmp(tokens[1], "st") == 0)
    {
      uint32_t sent, received, errors;
      float avgLatency;
      getCANStats(sent, received, errors, avgLatency);

      Serial.println("CAN Statistics:");
      Serial.print("  Node ID: ");
      Serial.println(nodeID);
      Serial.print("  Messages sent: ");
      Serial.println(sent);
      Serial.print("  Messages received: ");
      Serial.println(received);
      Serial.print("  Errors: ");
      Serial.println(errors);
      Serial.print("  Avg. latency: ");
      Serial.print(avgLatency);
      Serial.println(" us");
      Serial.println("ack");
      return;
    }

    // "c r" => Reset CAN statistics counters
    else if (strcmp(tokens[1], "r") == 0)
    {
      resetCANStats();
      Serial.println("CAN statistics reset");
      Serial.println("ack");
      return;
    }

    // "c sc" => Scan for active nodes on the CAN network
    else if (strcmp(tokens[1], "sc") == 0)
    {
      Serial.println("Scanning for active CAN nodes...");

      // Track which nodes respond
      bool nodeFound[64] = {false};
      int foundCount = 0;

      // Send ping messages to all possible node addresses
      for (uint8_t node = 1; node < 64; node++)
      {
        // Send a special ping message (control type 3 = discovery)
        if (sendControlCommand(node, 3, 0))
        {
          // Give some time for node to respond
          delay(50);

          // Process any responses that came in
          for (int i = 0; i < 5; i++)
          {
            can_frame frame;
            if (readCANMessage(&frame) == MCP2515::ERROR_OK)
            {
              uint8_t msgType, srcAddr, priority;
              parseCANId(frame.can_id, msgType, srcAddr, priority);

              if (!nodeFound[srcAddr])
              {
                nodeFound[srcAddr] = true;
                foundCount++;
              }
            }
            delay(10);
          }
        }
      }

      // Now send a broadcast message to catch any we missed
      sendControlCommand(CAN_ADDR_BROADCAST, 3, 0);
      delay(200);

      // Process any additional responses
      for (int i = 0; i < 20; i++)
      {
        can_frame frame;
        if (readCANMessage(&frame) == MCP2515::ERROR_OK)
        {
          uint8_t msgType, srcAddr, priority;
          parseCANId(frame.can_id, msgType, srcAddr, priority);

          if (!nodeFound[srcAddr])
          {
            nodeFound[srcAddr] = true;
            foundCount++;
          }
        }
        delay(10);
      }

      // Display results
      Serial.print("Found ");
      Serial.print(foundCount);
      Serial.println(" active nodes:");

      for (uint8_t node = 1; node < 64; node++)
      {
        if (nodeFound[node])
        {
          Serial.print("  Node ");
          Serial.println(node);
        }
      }

      Serial.println("Network scan complete");
      Serial.println("ack");
      return;
    }

    // "c l <destNode> <count>" => Measure round-trip latency
    else if (strcmp(tokens[1], "l") == 0)
    {
      if (numTokens < 4)
      {
        Serial.println("err");
        return;
      }

      uint8_t destNode = atoi(tokens[2]);
      int count = atoi(tokens[3]);

      Serial.print("Measuring round-trip latency to node ");
      Serial.print(destNode);
      Serial.print(" (");
      Serial.print(count);
      Serial.println(" samples)");

      unsigned long totalLatency = 0;
      int successCount = 0;

      for (int i = 0; i < count; i++)
      {
        unsigned long startTime = micros();

        // Send echo request (using control message type 2 = echo)
        if (sendControlCommand(destNode, 2, startTime))
        {
          // Wait for response with timeout
          unsigned long timeout = millis() + 500; // 500ms timeout
          bool responseReceived = false;

          while (millis() < timeout && !responseReceived)
          {
            can_frame frame;
            if (readCANMessage(&frame) == MCP2515::ERROR_OK)
            {
              // Check if it's an echo response
              uint8_t msgType, srcAddr, priority;
              parseCANId(frame.can_id, msgType, srcAddr, priority);

              if (msgType == CAN_TYPE_RESPONSE && srcAddr == destNode)
              {
                unsigned long endTime = micros();
                unsigned long latency = endTime - startTime;
                totalLatency += latency;
                successCount++;
                responseReceived = true;

                Serial.print("Sample ");
                Serial.print(i + 1);
                Serial.print(": ");
                Serial.print(latency);
                Serial.println(" us");
              }
            }
          }

          if (!responseReceived)
          {
            Serial.print("Sample ");
            Serial.print(i + 1);
            Serial.println(": Timeout");
          }
        }
        else
        {
          Serial.print("Sample ");
          Serial.print(i + 1);
          Serial.println(": Send failed");
        }

        delay(100); // Wait between samples
      }

      Serial.println("Latency measurement complete");
      if (successCount > 0)
      {
        float avgLatency = (float)totalLatency / successCount;
        Serial.print("Average round-trip latency: ");
        Serial.print(avgLatency, 2);
        Serial.println(" us");
      }
      else
      {
        Serial.println("No successful measurements");
      }
      Serial.println("ack");
      return;
    }
  }

  //-----------------------------------------------------------------------------
  // LUMINAIRE STATE COMMANDS
  //-----------------------------------------------------------------------------

  // "st <i> <state>" => set luminaire state (off/unoccupied/occupied)
  else if (strcmp(tokens[0], "st") == 0)
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }
    char stateStr[TOKEN_MAX_LENGTH];
    strcpy(stateStr, tokens[2]);
    uint8_t targetNode = atoi(tokens[1]);
    float val = strtof(tokens[2], NULL);

    // Convert state string to value for CAN transmission
    int stateVal = 0;
    if (strcmp(stateStr, "off") == 0)
      stateVal = 0;
    else if (strcmp(stateStr, "unoccupied") == 0)
      stateVal = 1;
    else if (strcmp(stateStr, "occupied") == 0)
      stateVal = 2;
    else
    {
      Serial.println("err");
      return;
    }

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Forward to specific node - control type 13 = luminaire state
      if (sendControlCommand(targetNode, 13, stateVal))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle broadcast case (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 13, stateVal))
      {
        // Also apply locally since broadcast includes this node
        if (strcmp(stateStr, "off") == 0)
          changeState(STATE_OFF);
        else if (strcmp(stateStr, "unoccupied") == 0)
          changeState(STATE_UNOCCUPIED);
        else if (strcmp(stateStr, "occupied") == 0)
          changeState(STATE_OCCUPIED);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    // Apply locally
    if (strcmp(stateStr, "off") == 0)
      changeState(STATE_OFF);
    else if (strcmp(stateStr, "unoccupied") == 0)
      changeState(STATE_UNOCCUPIED);
    else if (strcmp(stateStr, "occupied") == 0)
      changeState(STATE_OCCUPIED);
    Serial.println("ack");
    return;
  }
  // Default response for unrecognized commands
  Serial.println("ack");
  return;
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

/**
 * Print a comprehensive list of all available commands
 * Organizes commands by category and provides descriptions
 */
static void printHelp()
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