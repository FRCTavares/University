#include "CommandInterface.h"
#include "Storage.h"
#include "Metrics.h"
#include "CANComm.h"
#include <Arduino.h>
#include <math.h>
#include "LEDDriver.h"
#include "Globals.h"
#include "PIController.h"

// External variable declarations
extern float ledGain;
extern float baselineIlluminance;
extern PIController pid;
extern const float K;
extern const float BETA;
extern bool filterEnabled;

static void printHelp();
//-----------------------------------------------------------------------------
// COMMAND PARSING HELPERS
//-----------------------------------------------------------------------------

/**
 * Split a command line into space-separated tokens
 *
 * @param cmd The command string to parse
 * @param tokens Output array to store the parsed tokens
 * @param maxTokens Maximum number of tokens to extract
 * @param numFound Output parameter for number of tokens found
 */
static void parseTokens(const String &cmd, String tokens[], int maxTokens, int &numFound)
{
  numFound = 0;
  int startIdx = 0;
  while (numFound < maxTokens)
  {
    int spaceIdx = cmd.indexOf(' ', startIdx);
    if (spaceIdx == -1)
    {
      if (startIdx < (int)cmd.length())
      {
        tokens[numFound++] = cmd.substring(startIdx);
      }
      break;
    }
    tokens[numFound++] = cmd.substring(startIdx, spaceIdx);
    startIdx = spaceIdx + 1;
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
 * Process a single command line from Serial
 * Parses and executes commands for control, metrics, and CAN operations
 *
 * @param cmdLine The command string to process
 */
static void processCommandLine(const String &cmdLine)
{
  String trimmed = cmdLine;
  trimmed.trim();
  if (trimmed.length() == 0)
    return;

  // Tokenize the command line
  const int MAX_TOKENS = 6;
  String tokens[MAX_TOKENS];
  int numTokens = 0;
  parseTokens(trimmed, tokens, MAX_TOKENS, numTokens);
  if (numTokens == 0)
    return;

  String c0 = tokens[0];

  //-----------------------------------------------------------------------------
  // LED CONTROL COMMANDS
  //-----------------------------------------------------------------------------

  // "u <i> <val>" => set duty cycle (0.0-1.0)
  if (c0 == "u")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    float val = tokens[2].toFloat();

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
  else if (c0 == "p")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    float val = tokens[2].toFloat();

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
    setLEDPercentage(val);
    Serial.println("ack");
    return;
  }

  // "w <i> <watts>" => set LED by power in watts
  else if (c0 == "w")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    float val = tokens[2].toFloat();

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

  // "o <i> <val>" => set occupancy state (0=unoccupied, 1=occupied)
  else if (c0 == "o")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    int val = tokens[2].toInt();

    if (val != 0 && val != 1)
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
        occupancy = (val == 1);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    occupancy = (val == 1);
    Serial.println("ack");
    return;
  }

  // "a <i> <val>" => set anti-windup on/off (0=off, 1=on)
  else if (c0 == "a")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    int val = tokens[2].toInt();

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
        antiWindup = (val == 1);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    antiWindup = (val == 1);
    Serial.println("ack");
    return;
  }
  // "fi <i> <val>" => set filter enable/disable (0=off, 1=on)
  else if (c0 == "fi")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    int val = tokens[2].toInt();

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
        filterEnabled = (val == 1);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    filterEnabled = (val == 1);
    Serial.println("ack");
    return;
  }

  // "f <i> <val>" => set feedback control on/off (0=off, 1=on)
  else if (c0 == "f")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    int val = tokens[2].toInt();

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
        feedbackControl = (val == 1);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    feedbackControl = (val == 1);
    Serial.println("ack");
    return;
  }

  // "r <i> <val>" => set illuminance reference (lux)
  else if (c0 == "r")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    float val = tokens[2].toFloat();

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
        refIlluminance = val;
        setpointLux = val;
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    refIlluminance = val;
    setpointLux = val;
    Serial.println("ack");
    return;
  }
  // "h" => Print help information
  else if (c0 == "h")
  {
    printHelp();
    return;
  }

  //-----------------------------------------------------------------------------
  // DATA STREAMING COMMANDS
  //-----------------------------------------------------------------------------

  // "s <x> <i>" => start stream of real-time variable <x> for desk <i>
  else if (c0 == "s")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    String var = tokens[1];
    uint8_t targetNode = tokens[2].toInt();

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Stream start = control type 11
      // Encode the variable type in the value field
      float varCode = 0; // Default for 'y' (lux)

      // Map variable names to numeric codes for CAN transmission
      if (var.equalsIgnoreCase("u"))
        varCode = 1; // Duty cycle
      else if (var.equalsIgnoreCase("p"))
        varCode = 2; // Power percentage
      else if (var.equalsIgnoreCase("o"))
        varCode = 3; // Occupancy
      else if (var.equalsIgnoreCase("a"))
        varCode = 4; // Anti-windup
      else if (var.equalsIgnoreCase("f"))
        varCode = 5; // Feedback control
      else if (var.equalsIgnoreCase("r"))
        varCode = 6; // Reference illuminance
      else if (var.equalsIgnoreCase("y"))
        varCode = 0; // Illuminance (lux)
      else if (var.equalsIgnoreCase("v"))
        varCode = 7; // LDR voltage
      else if (var.equalsIgnoreCase("d"))
        varCode = 8; // External illuminance
      else if (var.equalsIgnoreCase("t"))
        varCode = 9; // Elapsed time
      else if (var.equalsIgnoreCase("V"))
        varCode = 10; // Visibility error
      else if (var.equalsIgnoreCase("F"))
        varCode = 11; // Flicker
      else if (var.equalsIgnoreCase("E"))
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
    return;
  }

  // "S <x> <i>" => stop stream of variable <x> for desk <i>
  else if (c0 == "S")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    String var = tokens[1];
    uint8_t targetNode = tokens[2].toInt();

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Stream stop = control type 12
      // Encode the variable type in the value field
      float varCode = 0; // Default for 'y' (lux)

      // Map variable names to numeric codes for CAN transmission
      if (var.equalsIgnoreCase("u"))
        varCode = 1; // Duty cycle
      else if (var.equalsIgnoreCase("p"))
        varCode = 2; // Power percentage
      else if (var.equalsIgnoreCase("o"))
        varCode = 3; // Occupancy
      else if (var.equalsIgnoreCase("a"))
        varCode = 4; // Anti-windup
      else if (var.equalsIgnoreCase("f"))
        varCode = 5; // Feedback control
      else if (var.equalsIgnoreCase("r"))
        varCode = 6; // Reference illuminance
      else if (var.equalsIgnoreCase("y"))
        varCode = 0; // Illuminance (lux)
      else if (var.equalsIgnoreCase("v"))
        varCode = 7; // LDR voltage
      else if (var.equalsIgnoreCase("d"))
        varCode = 8; // External illuminance
      else if (var.equalsIgnoreCase("t"))
        varCode = 9; // Elapsed time
      else if (var.equalsIgnoreCase("V"))
        varCode = 10; // Visibility error
      else if (var.equalsIgnoreCase("F"))
        varCode = 11; // Flicker
      else if (var.equalsIgnoreCase("E"))
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
  else if (c0 == "g")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }
    String subCommand = tokens[1];
    String originalCase = tokens[1];
    subCommand.toLowerCase();

    uint8_t targetNode = tokens[2].toInt();
    String idx = tokens[2];

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Map the get command to a CAN query message type (using code 20-32)
      uint8_t queryType = 20; // Default query type

      // Map variable types to query codes
      if (originalCase == "V")
        queryType = 20; // Visibility error
      else if (originalCase == "F")
        queryType = 21; // Flicker
      else if (originalCase == "E")
        queryType = 22; // Energy
      else if (subCommand == "u")
        queryType = 23; // Duty cycle
      else if (subCommand == "o")
        queryType = 24; // Occupancy
      else if (subCommand == "a")
        queryType = 25; // Anti-windup
      else if (subCommand == "f")
        queryType = 26; // Feedback control
      else if (subCommand == "r")
        queryType = 27; // Reference illuminance
      else if (subCommand == "y")
        queryType = 28; // Current illuminance
      else if (subCommand == "p")
        queryType = 29; // Power consumption
      else if (subCommand == "t")
        queryType = 30; // Elapsed time
      else if (subCommand == "v")
        queryType = 31; // LDR voltage
      else if (subCommand == "d")
        queryType = 32; // External illuminance
      else
      {
        Serial.println("err: Unsupported remote variable query");
        return;
      }

      // Send the query to the remote node
      if (sendControlCommand(targetNode, queryType, 0.0f))
      {
        Serial.println("Query sent to node " + String(targetNode));

        // Wait for response with timeout
        unsigned long timeout = millis() + 500; // 500ms timeout
        bool responseReceived = false;

        while (millis() < timeout && !responseReceived)
        {
          can_frame frame;
          if (readCANMessage(&frame) == MCP2515::ERROR_OK)
          {
            // Parse message and check if it's a response from our target
            uint8_t msgType, destAddr, priority;
            parseCANId(frame.can_id, msgType, destAddr, priority);

            uint8_t senderNodeID = frame.data[0];

            if (msgType == CAN_TYPE_RESPONSE && senderNodeID == targetNode &&
                (destAddr == nodeID || destAddr == CAN_ADDR_BROADCAST))
            {
              // Extract the response value and display
              float value = bytesToFloat(&frame.data[2]);

              // Format the response based on the original query type
              if (originalCase == "V" || originalCase == "F" || originalCase == "E")
              {
                Serial.print(originalCase);
              }
              else
              {
                Serial.print(subCommand);
              }
              Serial.print(" ");
              Serial.print(idx);
              Serial.print(" ");

              // Format the value with appropriate precision
              if (subCommand == "u" || originalCase == "F" || originalCase == "E")
                Serial.println(value, 4); // 4 decimal places
              else if (originalCase == "V" || subCommand == "y" ||
                       subCommand == "p" || subCommand == "d")
                Serial.println(value, 2); // 2 decimal places
              else if (subCommand == "v")
                Serial.println(value, 3); // 3 decimal places
              else if (subCommand == "o" || subCommand == "a" ||
                       subCommand == "f" || subCommand == "t")
                Serial.println((int)value); // Integer values
              else
                Serial.println(value); // Default format

              responseReceived = true;
            }
          }
          delay(5); // Small delay to prevent hammering the CAN bus
        }

        if (!responseReceived)
        {
          Serial.println("err: No response from node " + String(targetNode));
        }
      }
      else
      {
        Serial.println("err: Failed to send query to node " + String(targetNode));
      }
      return;
    }

    // Handle local metric queries

    // Quality metrics
    // "g V <i>" => "V <i> <val>" (Visibility error metric)
    if (originalCase == "V")
    {
      float V = computeVisibilityErrorFromBuffer();
      Serial.print("V ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(V, 2);
      return;
    }
    // "g F <i>" => "F <i> <val>" (Flicker metric)
    else if (originalCase == "F")
    {
      float F = computeFlickerFromBuffer();
      Serial.print("F ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(F, 4);
      return;
    }
    // "g E <i>" => "E <i> <val>" (Energy metric)
    else if (originalCase == "E")
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
    if (subCommand == "u")
    {
      Serial.print("u ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(dutyCycle, 4);
      return;
    }
    // "g o <i>" => "o <i> <val>" (occupancy)
    else if (subCommand == "o")
    {
      int occVal = occupancy ? 1 : 0;
      Serial.print("o ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(occVal);
      return;
    }
    // "g a <i>" => "a <i> <val>" (anti-windup)
    else if (subCommand == "a")
    {
      int awVal = antiWindup ? 1 : 0;
      Serial.print("a ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(awVal);
      return;
    }
    // "g f <i>" => "f <i> <val>" (feedback control)
    else if (subCommand == "f")
    {
      int fbVal = feedbackControl ? 1 : 0;
      Serial.print("f ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(fbVal);
      return;
    }
    // "g r <i>" => "r <i> <val>" (reference illuminance)
    else if (subCommand == "r")
    {
      Serial.print("r ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(refIlluminance, 4);
      return;
    }
    // "g y <i>" => "y <i> <val>" (current illuminance)
    else if (subCommand == "y")
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
    else if (subCommand == "v")
    {
      float vLdr = getVoltageAtLDR();
      Serial.print("v ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(vLdr, 3);
      return;
    }
    // "g d <i>" => external illuminance => "d <i> <val>"
    else if (subCommand == "d")
    {
      float dVal = getExternalIlluminance();
      Serial.print("d ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(dVal, 2);
      return;
    }
    // "g p <i>" => instantaneous power => "p <i> <val>"
    else if (subCommand == "p")
    {
      float pVal = getPowerConsumption();
      Serial.print("p ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(pVal, 2);
      return;
    }
    // "g t <i>" => elapsed time => "t <i> <val>"
    else if (subCommand == "t")
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
    else if (subCommand == "b")
    {
      if (numTokens < 4)
      {
        Serial.println("err");
        return;
      }
      String xVar = tokens[2];
      int iDesk = tokens[3].toInt();
      String bufferData = getLastMinuteBuffer(xVar, iDesk);
      Serial.print("b ");
      Serial.print(xVar);
      Serial.print(" ");
      Serial.print(iDesk);
      Serial.print(" ");
      Serial.println(bufferData);
      return;
    }
    else if (subCommand == "bigdump")
    {
      int iDesk = tokens[2].toInt();

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
    else if (subCommand == "mdump")
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
  else if (c0 == "k")
  {
    if (numTokens < 4)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    String param = tokens[2];
    float value = tokens[3].toFloat();

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Map parameter to control code for CAN
      uint8_t paramCode = 0;
      if (param.equalsIgnoreCase("beta"))
        paramCode = 15;
      else if (param.equalsIgnoreCase("k")) // Changed from "kp" to "k"
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
    if (param.equalsIgnoreCase("beta"))
    {
      if (value < 0.0 || value > 1.0)
      {
        Serial.println("err: beta must be between 0.0 and 1.0");
        return;
      }
      pid.setWeighting(value);
      Serial.println("ack");
    }
    else if (param.equalsIgnoreCase("k")) // Changed from "kp" to "k"
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
  else if (c0 == "c")
  {
    // "c m <0|1>" => Enable/disable CAN message monitoring
    if (tokens[1] == "m")
    {
      if (numTokens < 3)
      {
        Serial.println("err");
        return;
      }

      canMonitorEnabled = (tokens[2].toInt() == 1);

      Serial.print("CAN monitoring ");
      Serial.println(canMonitorEnabled ? "enabled" : "disabled");
      Serial.println("ack");
      return;
    }

    // "c st" => Display CAN communication statistics
    else if (tokens[1] == "st")
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
    else if (tokens[1] == "r")
    {
      resetCANStats();
      Serial.println("CAN statistics reset");
      Serial.println("ack");
      return;
    }

    // "c sc" => Scan for active nodes on the CAN network
    else if (tokens[1] == "sc")
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
    else if (tokens[1] == "l")
    {
      if (numTokens < 4)
      {
        Serial.println("err");
        return;
      }

      uint8_t destNode = tokens[2].toInt();
      int count = tokens[3].toInt();

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
  else if (c0 == "st")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    String stateStr = tokens[2];
    stateStr.toLowerCase();

    // Convert state string to value for CAN transmission
    int stateVal = 0;
    if (stateStr == "off")
      stateVal = 0;
    else if (stateStr == "unoccupied")
      stateVal = 1;
    else if (stateStr == "occupied")
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
        if (stateStr == "off")
          changeState(STATE_OFF);
        else if (stateStr == "unoccupied")
          changeState(STATE_UNOCCUPIED);
        else if (stateStr == "occupied")
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
    if (stateStr == "off")
      changeState(STATE_OFF);
    else if (stateStr == "unoccupied")
      changeState(STATE_UNOCCUPIED);
    else if (stateStr == "occupied")
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
  if (Serial.available() > 0)
  {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0)
    {
      processCommandLine(input);
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