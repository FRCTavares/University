#ifndef CANCOMM_H
#define CANCOMM_H

#include <Arduino.h>
#include "mcp2515.h"

// CAN message callback function type
typedef void (*CANMessageCallback)(const can_frame &msg);

// Initialization
void initCANComm();

// Message building and parsing
uint32_t buildCANId(uint8_t msgType, uint8_t destAddr, uint8_t priority);
void parseCANId(uint32_t canId, uint8_t& msgType, uint8_t& destAddr, uint8_t& priority);

// Data conversion
void floatToBytes(float value, uint8_t* bytes);
float bytesToFloat(const uint8_t* bytes);

// Higher-level message functions
bool sendSensorReading(uint8_t destAddr, uint8_t sensorType, float value);
bool sendControlCommand(uint8_t destAddr, uint8_t controlType, float value);
bool sendHeartbeat();

// Core CAN functions
void canCommLoop();
MCP2515::ERROR sendCANMessage(const can_frame &frame);
MCP2515::ERROR readCANMessage(struct can_frame *frame);
void setCANMessageCallback(CANMessageCallback callback);

// Statistics
void getCANStats(uint32_t &sent, uint32_t &received, uint32_t &errors, float &avgLatency);
void resetCANStats();

#endif // CANCOMM_H