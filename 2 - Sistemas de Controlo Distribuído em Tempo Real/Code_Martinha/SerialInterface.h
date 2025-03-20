#pragma once

// Initialize serial interface
void initSerialInterface();

// Process commands from serial port
void processSerialCommands();

// Streaming interface
void startStream(const String &var, int index);
void stopStream(const String &var, int index);
void handleStreaming();