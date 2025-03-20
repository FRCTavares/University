#pragma once
#include <SPI.h>
#include <mcp2515.h>

typedef void (*CANMessageCallback)(const can_frame &msg);

// Initialize CAN communication
void initCANComm();

// CAN message handling
MCP2515::ERROR sendCANMessage(const can_frame &frame);
MCP2515::ERROR readCANMessage(struct can_frame *frame);
void canCommLoop();

// Helper functions
uint32_t buildCANId(uint8_t msgType, uint8_t destAddr, uint8_t priority);
void parseCANId(uint32_t canId, uint8_t &msgType, uint8_t &destAddr, uint8_t &priority);
void floatToBytes(float value, uint8_t *bytes);
float bytesToFloat(const uint8_t *bytes);

// Send specific message types
bool sendSensorReading(uint8_t destAddr, uint8_t sensorType, float value);
bool sendControlCommand(uint8_t destAddr, uint8_t controlType, float value);
bool sendQueryResponse(uint8_t destNode, float value);
bool sendHeartbeat();

// Callback registration
void setCANMessageCallback(CANMessageCallback callback);

// Statistics
void getCANStats(uint32_t &sent, uint32_t &received, uint32_t &errors, float &avgLatency);
void resetCANStats();