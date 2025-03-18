#include "CommandInterface.h"
#include "Storage.h"     
#include "Metrics.h"     
#include "CANComm.h"     
#include <Arduino.h>
#include <math.h>
#include "LEDDriver.h"
#include "Globals.h" 


// Helper to split a command line into tokens by space
static void parseTokens(const String &cmd, String tokens[], int maxTokens, int &numFound) {
  numFound = 0;
  int startIdx = 0;
  while (numFound < maxTokens) {
    int spaceIdx = cmd.indexOf(' ', startIdx);
    if (spaceIdx == -1) {
      if (startIdx < (int)cmd.length()) {
        tokens[numFound++] = cmd.substring(startIdx);
      }
      break;
    }
    tokens[numFound++] = cmd.substring(startIdx, spaceIdx);
    startIdx = spaceIdx + 1;
  }
}

// Processes a single command line from Serial
static void processCommandLine(const String &cmdLine) {
  String trimmed = cmdLine;
  trimmed.trim();
  if (trimmed.length() == 0) return;

  // Tokenize
  const int MAX_TOKENS = 6;
  String tokens[MAX_TOKENS];
  int numTokens = 0;
  parseTokens(trimmed, tokens, MAX_TOKENS, numTokens);
  if (numTokens == 0) return;

  String c0 = tokens[0];
  c0.toLowerCase();

  //Commands:

  // 1) Set duty cycle: "u <i> <val>"
  // 2) Get duty cycle: "g u <i>"
  // 3) Set illuminance reference: "r <i> <val>"
  // 4) Get illuminance reference: "g r <i>"
  // 5) Measure actual illuminance: "g y <i>" => "y <i> <val>"
  // 6) Measure the voltage level at the LDR: "g v <i>" => "v <i> <val>"
  // 7) Set occupancy: "o <i> <val>"
  // 8) Get occupancy: "g o <i>"
  // 9) Set anti-windup: "a <i> <val>"
  // 10) Get anti-windup: "g a <i>"
  // 11) Set feedback control: "f <i> <val>"
  // 12) Get feedback control: "g f <i>"
  // 13) Get external illuminance: "g d <i>" => "d <i> <val>"
  // 14) Get instantaneous power: "g p <i>" => "p <i> <val>"
  // 15) Get elapsed time: "g t <i>" => "t <i> <val>"
  // 16) Start the stream: "s <x> <i>"
  // 17) Stop the stream: "S <x> <i>"
  // 18) Get the last minute buffer: "g b <x> <i>" => "b <x> <i> <val1>,<val2>..."
  // 19) For Table 2: "g E <i>" => "E <i> <val>", "g V <i>" => "V <i> <val>", "g F <i>" => "F <i> <val>"
  // 20) For CAN commands: "can <cmd> <args>"

  // "u <i> <val>" => set duty cycle
  if (c0 == "u") {
    if (numTokens < 3) { Serial.println("err"); return; }
    float val = tokens[2].toFloat();
    if (val < 0.0f || val > 1.0f) { Serial.println("err"); return; }
    setLEDDutyCycle(val);
    Serial.println("ack");
    return;
  }
  // Add an additional command for setting by percentage
  else if (c0 == "p") {
    // "p <i> <percentage>" => set LED by percentage
    if (numTokens < 3) { Serial.println("err"); return; }
    float percentage = tokens[2].toFloat();
    if (percentage < 0.0f || percentage > 100.0f) { Serial.println("err"); return; }
    setLEDPercentage(percentage); 
    Serial.println("ack");
    return;
}
  // Add an additional command for setting by power
  else if (c0 == "w") {
    // "w <i> <watts>" => set LED by power in watts
    if (numTokens < 3) { Serial.println("err"); return; }
    float watts = tokens[2].toFloat();
    if (watts < 0.0f || watts > MAX_POWER_WATTS) { Serial.println("err"); return; }
    setLEDPower(watts);
    Serial.println("ack");
    return;
}
  // Set occupancy: "o <i> <val>"
  else if (c0 == "o") {
    if (numTokens < 3) { Serial.println("err"); return; }
    int val = tokens[2].toInt();
    if (val != 0 && val != 1) { Serial.println("err"); return; }
    occupancy = (val == 1);
    Serial.println("ack");
    return;
  }
  // "a <i> <val>" => set anti-windup on/off
  else if (c0 == "a") {
    if (numTokens < 3) { Serial.println("err"); return; }
    int val = tokens[2].toInt();
    if (val != 0 && val != 1) { Serial.println("err"); return; }
    antiWindup = (val == 1);
    Serial.println("ack");
    return;
  }
  // "f <i> <val>" => set feedback control on/off
  else if (c0 == "f") {
    if (numTokens < 3) { Serial.println("err"); return; }
    int val = tokens[2].toInt();
    if (val != 0 && val != 1) { Serial.println("err"); return; }
    feedbackControl = (val == 1);
    Serial.println("ack");
    return;
  }
  // "r <i> <val>" => set illuminance reference
  else if (c0 == "r") {
    if (numTokens < 3) { Serial.println("err"); return; }
    float val = tokens[2].toFloat();
    refIlluminance = val;
    setpointLux = val;
    Serial.println("ack");
    return;
  }
  // "y <i>" => measure actual illuminance => respond "y <i> <val>"
  else if (c0 == "y") {
    if (numTokens < 2) { Serial.println("err"); return; }
    float lux = readLux();
    Serial.print("y ");
    Serial.print(tokens[1]);
    Serial.print(" ");
    Serial.println(lux, 2);
    return;
  }
  // "s <x> <i>" => start stream of real-time variable <x> for desk <i>
  else if (c0 == "s") { 
    if (numTokens < 3) { Serial.println("err"); return; }
    String var = tokens[1];
    int index = tokens[2].toInt();
    startStream(var, index);
    return;
  }
  // "S <x> <i>" => stop stream
  else if (c0 == "S") {
    if (numTokens < 3) { Serial.println("err"); return; }
    String var = tokens[1];
    int index = tokens[2].toInt();
    stopStream(var, index);
    Serial.println("ack");
    return;
  }
  // If first token is 'g' => "g <sub> <i>"
  else if (c0 == "g") {
    if (numTokens < 3) { Serial.println("err"); return; }
    String subCommand = tokens[1];
    String originalCase = tokens[1];
    
    subCommand.toLowerCase();
    String idx = tokens[2];

    // Metric commands
    // "g V <i>" => "V <i> <val>" (Visibility error metric)
    if (originalCase == "V") {
      float V = computeVisibilityErrorFromBuffer();
      Serial.print("V ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(V, 2);
      return;
    }
    // "g F <i>" => "F <i> <val>" (Flicker metric)
    else if (originalCase == "F") {
      float F = computeFlickerFromBuffer();
      Serial.print("F ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(F, 4);
      return;
    }
    // "g E <i>" => "E <i> <val>" (Energy metric)
    else if (originalCase == "E") {
      float E = computeEnergyFromBuffer();
      Serial.print("E ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(E, 4);
      return;
    }

    // "g u <i>" => "u <i> <val>"
    if (subCommand == "u") {
      Serial.print("u ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(dutyCycle, 4);
      return;
    }
    // "g o <i>" => "o <i> <val>"
    else if (subCommand == "o") {
      int occVal = occupancy ? 1 : 0;
      Serial.print("o ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(occVal);
      return;
    }
    // "g a <i>" => "a <i> <val>"
    else if (subCommand == "a") {
      int awVal = antiWindup ? 1 : 0;
      Serial.print("a ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(awVal);
      return;
    }
    // "g f <i>" => "f <i> <val>"
    else if (subCommand == "f") {
      int fbVal = feedbackControl ? 1 : 0;
      Serial.print("f ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(fbVal);
      return;
    }
    // "g r <i>" => "r <i> <val>"
    else if (subCommand == "r") {
      Serial.print("r ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(refIlluminance, 4);
      return;
    }
    // "g y <i>" => "y <i> <val>"
    else if (subCommand == "y") {
      float lux = readLux();
      Serial.print("y ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(lux, 2);
      return;
    }
    // "g v <i>" => measure voltage level at LDR => "v <i> <val>"
    else if (subCommand == "v") {
      float vLdr = getVoltageAtLDR();
      Serial.print("v ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(vLdr, 3);
      return;
    }
    // "g d <i>" => external illuminance => "d <i> <val>"
    else if (subCommand == "d") {
      float dVal = getExternalIlluminance(); 
      Serial.print("d ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(dVal, 2);
      return;
    }
    // "g p <i>" => instantaneous power => "p <i> <val>"
    else if (subCommand == "p") {
      float pVal = getPowerConsumption(); 
      Serial.print("p ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(pVal, 2);
      return;
    }
    // "g t <i>" => elapsed time => "t <i> <val>"
    else if (subCommand == "t") {
      unsigned long sec = getElapsedTime();
      Serial.print("t ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(sec);
      return;
    }
    // "g b <x> <i>" => "b <x> <i> <val1>,<val2>..."
    else if (subCommand == "b") {
      if (numTokens < 4) { 
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
    else {
      Serial.println("err");
      return;
    }
  }
  // "debug <option> <value>" => set debug options
  else if (c0 == "debug") {
    if (numTokens < 3) { Serial.println("err"); return; }
    
    String option = tokens[1];
    int value = tokens[2].toInt();
    bool enable = (value == 1);
    
    if (option.equalsIgnoreCase("all")) {
      DEBUG_MODE = enable;
      Serial.print("Debug mode: ");
      Serial.println(enable ? "ON" : "OFF");
    }
    else if (option.equalsIgnoreCase("led")) {
      DEBUG_LED = enable;
      Serial.print("LED debug: ");
      Serial.println(enable ? "ON" : "OFF");
    }
    else if (option.equalsIgnoreCase("sensor")) {
      DEBUG_SENSOR = enable;
      Serial.print("Sensor debug: ");
      Serial.println(enable ? "ON" : "OFF");
    }
    else if (option.equalsIgnoreCase("pid")) {
      DEBUG_PID = enable;
      Serial.print("PID debug: ");
      Serial.println(enable ? "ON" : "OFF");
    }
    else if (option.equalsIgnoreCase("plot")) {
      DEBUG_PLOTTING = enable;
      Serial.print("Plotting: ");
      Serial.println(enable ? "ON" : "OFF");
    }
    else if (option.equalsIgnoreCase("status")) {
      // Print current debug settings
      Serial.println("Debug Status:");
      Serial.print("  Master Debug: ");
      Serial.println(DEBUG_MODE ? "ON" : "OFF");
      Serial.print("  LED Debug: ");
      Serial.println(DEBUG_LED ? "ON" : "OFF");
      Serial.print("  Sensor Debug: ");
      Serial.println(DEBUG_SENSOR ? "ON" : "OFF");
      Serial.print("  PID Debug: ");
      Serial.println(DEBUG_PID ? "ON" : "OFF");
      Serial.print("  Plotting: ");
      Serial.println(DEBUG_PLOTTING ? "ON" : "OFF");
    }
    else {
      Serial.println("err");
      return;
    }
    
    Serial.println("ack");
    return;
  }
  

  
  // CAN Commands handled if c0 == "can"
  // "can send <destNode> <msgType> <value>" => Send a CAN message
  else if (c0 == "can" && tokens[1] == "send") {
    if (numTokens < 5) { Serial.println("err"); return; }
    
    uint8_t destNode = tokens[2].toInt();
    uint8_t msgType = tokens[3].toInt();
    float value = tokens[4].toFloat();
    
    bool success = false;
    
    if (msgType == 0) { // Control message
      success = sendControlCommand(destNode, 0, value); // Type 0 = setpoint
    } else if (msgType == 1) { // Sensor reading
      success = sendSensorReading(destNode, 0, value); // Type 0 = lux
    } else {
      Serial.println("err: Invalid message type");
      return;
    }
    
    if (success) {
      Serial.println("ack");
    } else {
      Serial.println("err: Send failed");
    }
    return;
  }
  // "can periodic <0|1>" => Enable/disable periodic CAN transmission
  else if (c0 == "can" && tokens[1] == "periodic") {
    if (numTokens < 3) { Serial.println("err"); return; }
    
    periodicCANEnabled = (tokens[2].toInt() == 1);
    
    Serial.print("Periodic CAN transmission ");
    Serial.println(periodicCANEnabled ? "enabled" : "disabled");
    Serial.println("ack");
    return;
  }
  // "can monitor <0|1>" => Enable/disable printing of received CAN messages
  else if (c0 == "can" && tokens[1] == "monitor") {
    if (numTokens < 3) { Serial.println("err"); return; }
    
    canMonitorEnabled = (tokens[2].toInt() == 1);
    
    Serial.print("CAN monitoring ");
    Serial.println(canMonitorEnabled ? "enabled" : "disabled");
    Serial.println("ack");
    return;
  }
  // "can stats" => Display CAN communication statistics
  else if (c0 == "can" && tokens[1] == "stats") {
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
  // "can reset" => Reset CAN statistics
  else if (c0 == "can" && tokens[1] == "reset") {
    resetCANStats();
    Serial.println("CAN statistics reset");
    Serial.println("ack");
    return;
  }
  // "can heartbeat" => Send a heartbeat message manually
  else if (c0 == "can" && tokens[1] == "heartbeat") {
    if(sendHeartbeat()) {
      Serial.println("Heartbeat sent successfully");
      Serial.println("ack");
    } else {
      Serial.println("Failed to send heartbeat");
      Serial.println("err");
    }
    return;
  }
  // "can broadcast <value>" => Broadcast a value to all nodes
  else if (c0 == "can" && tokens[1] == "broadcast") {
    if (numTokens < 3) { Serial.println("err"); return; }
    
    float value = tokens[2].toFloat();
    if(sendSensorReading(CAN_ADDR_BROADCAST, 0, value)) {
      Serial.println("Broadcast sent successfully");
      Serial.println("ack");
    } else {
      Serial.println("Failed to broadcast message");
      Serial.println("err");
    }
    return;
  }
  // "can latency <destNode> <count>" => Measure round-trip latency
  // "can scan" => Scan for active nodes on the network
  else if (c0 == "can" && tokens[1] == "scan") {
    Serial.println("Scanning for active CAN nodes...");
    
    // We'll track which nodes respond
    bool nodeFound[64] = {false};
    int foundCount = 0;
    
    // Send ping messages to all possible node addresses
    for(uint8_t node = 1; node < 64; node++) {
      // Send a special ping message
      if(sendControlCommand(node, 3, 0)) {
        // Give some time for node to respond
        delay(50);
        
        // Process any responses that came in
        for(int i = 0; i < 5; i++) {
          can_frame frame;
          if(readCANMessage(&frame) == MCP2515::ERROR_OK) {
            uint8_t msgType, srcAddr, priority;
            parseCANId(frame.can_id, msgType, srcAddr, priority);
            
            if(!nodeFound[srcAddr]) {
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
    for(int i = 0; i < 20; i++) {
      can_frame frame;
      if(readCANMessage(&frame) == MCP2515::ERROR_OK) {
        uint8_t msgType, srcAddr, priority;
        parseCANId(frame.can_id, msgType, srcAddr, priority);
        
        if(!nodeFound[srcAddr]) {
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
    
    for(uint8_t node = 1; node < 64; node++) {
      if(nodeFound[node]) {
        Serial.print("  Node ");
        Serial.println(node);
      }
    }
    
    Serial.println("Network scan complete");
    Serial.println("ack");
    return;
  }
  
  // "can latency <destNode> <count>" => Measure round-trip latency
  else if (c0 == "can" && tokens[1] == "latency") {
    if (numTokens < 4) { Serial.println("err"); return; }
    
    uint8_t destNode = tokens[2].toInt();
    int count = tokens[3].toInt();
    
    Serial.print("Measuring round-trip latency to node ");
    Serial.print(destNode);
    Serial.print(" (");
    Serial.print(count);
    Serial.println(" samples)");
    
    unsigned long totalLatency = 0;
    int successCount = 0;
    
    for(int i = 0; i < count; i++) {
      unsigned long startTime = micros();
      
      // Send echo request (using control message type 2)
      if(sendControlCommand(destNode, 2, startTime)) {
        // Wait for response with timeout
        unsigned long timeout = millis() + 500; // 500ms timeout
        bool responseReceived = false;
        
        while(millis() < timeout && !responseReceived) {
          can_frame frame;
          if(readCANMessage(&frame) == MCP2515::ERROR_OK) {
            // Parse message and check if it's an echo response
            uint8_t msgType, srcAddr, priority;
            parseCANId(frame.can_id, msgType, srcAddr, priority);
            
            if(msgType == CAN_TYPE_RESPONSE && srcAddr == destNode) {
              unsigned long endTime = micros();
              unsigned long latency = endTime - startTime;
              totalLatency += latency;
              successCount++;
              responseReceived = true;
              
              Serial.print("Sample ");
              Serial.print(i+1);
              Serial.print(": ");
              Serial.print(latency);
              Serial.println(" us");
            }
          }
        }
        
        if(!responseReceived) {
          Serial.print("Sample ");
          Serial.print(i+1);
          Serial.println(": Timeout");
        }
      } else {
        Serial.print("Sample ");
        Serial.print(i+1);
        Serial.println(": Send failed");
      }
      
      delay(100); // Wait between samples
    }
    
    Serial.println("Latency measurement complete");
    if(successCount > 0) {
      float avgLatency = (float)totalLatency / successCount;
      Serial.print("Average round-trip latency: ");
      Serial.print(avgLatency, 2);
      Serial.println(" us");
    } else {
      Serial.println("No successful measurements");
    }
    Serial.println("ack");
    return;
  }
  else if (c0 == "can" && tokens[1] == "loopback") {
    Serial.println("Testing CAN controller in loopback mode...");
    
    // Switch to loopback mode (internal testing)
    can0.reset();
    can0.setBitrate(CAN_125KBPS);
    can0.setLoopbackMode();
    
    // Send test message to self
    can_frame testFrame;
    testFrame.can_id = buildCANId(CAN_TYPE_CONTROL, nodeID, CAN_PRIO_NORMAL);
    testFrame.can_dlc = 8;
    testFrame.data[0] = nodeID;
    testFrame.data[1] = 99;
    floatToBytes(12345.67, &testFrame.data[2]);
    testFrame.data[6] = 0;
    testFrame.data[7] = 0;
    
    Serial.print("Sending test message: ");
    MCP2515::ERROR sendResult = can0.sendMessage(&testFrame);
    Serial.println(sendResult == MCP2515::ERROR_OK ? "SUCCESS" : "FAILED");
    
    // Try to receive it
    delay(50);
    can_frame rxFrame;
    MCP2515::ERROR rxResult = can0.readMessage(&rxFrame);
    
    if (rxResult == MCP2515::ERROR_OK) {
      Serial.println("Received loopback message - MCP2515 is working!");
      // Return to normal mode
      can0.reset();
      can0.setBitrate(CAN_125KBPS);
      can0.setNormalMode();
      Serial.println("ack");
    } else {
      Serial.println("Failed to receive loopback message - hardware issue!");
      Serial.println("err");
    }
    return;
  }
    // If none matched:
    Serial.println("err");
  }

// Called repeatedly in loop() to process any serial input
void processSerialCommands() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      processCommandLine(input);
    }
  }
}
