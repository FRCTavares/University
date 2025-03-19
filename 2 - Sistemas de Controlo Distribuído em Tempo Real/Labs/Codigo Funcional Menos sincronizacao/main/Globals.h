#ifndef GLOBALS_H
#define GLOBALS_H

#define MAX_ILLUMINANCE 2000.0f

#include <Arduino.h>

// --- Global Constants ---
extern const float MAX_POWER_WATTS;

// --- Control System State ---
extern float setpointLux;    // Desired lux (setpoint)
extern float dutyCycle;      // Current duty cycle [0..1]
extern float refIlluminance; // Reference illuminance
extern bool occupancy;       // Occupancy flag
extern bool antiWindup;      // Anti-windup flag for PID controller
extern bool feedbackControl; // Enable/disable feedback control

// Define luminaire states
enum LuminaireState
{
    STATE_OFF = 0,        // No one in office
    STATE_UNOCCUPIED = 1, // No one at desk, low light
    STATE_OCCUPIED = 2    // Desk is busy, full light
};

// Declare in global section
extern LuminaireState luminaireState;

// --- Debug Flags ---
extern bool DEBUG_MODE;     // Master debug switch
extern bool DEBUG_LED;      // LED driver debug messages
extern bool DEBUG_SENSOR;   // Sensor readings debug
extern bool DEBUG_PID;      // PID control debug
extern bool DEBUG_PLOTTING; // Serial plotter output

// --- CAN Communication Flags ---
extern bool periodicCANEnabled; // Enable periodic message sending
extern bool canMonitorEnabled;  // Display received messages
extern uint8_t nodeID;          // This node's identifier

// --- CAN Message Types ---
#define CAN_TYPE_CONTROL 0x00      // Control messages (setpoints, modes)
#define CAN_TYPE_SENSOR 0x01       // Sensor data (illuminance, duty cycle)
#define CAN_TYPE_STATUS 0x02       // Status information (power, mode)
#define CAN_TYPE_CONFIG 0x03       // Configuration parameters
#define CAN_TYPE_ERROR 0x04        // Error reports
#define CAN_TYPE_QUERY 0x05        // Data requests
#define CAN_TYPE_RESPONSE 0x06     // Responses to queries
#define CAN_TYPE_HEARTBEAT 0x07    // Node presence signals
#define CAN_CTRL_STATE_CHANGE 0x10 // Or choose another appropriate value that doesn't conflict

// CAN priority levels
#define CAN_PRIO_HIGH 0x00
#define CAN_PRIO_NORMAL 0x01
#define CAN_PRIO_LOW 0x02
#define CAN_PRIO_LOWEST 0x03

// CAN node addresses
#define CAN_ADDR_BROADCAST 0x00 // Broadcast to all nodes

// --- Function Declarations ---
float readLux();
float getVoltageAtLDR();
float getExternalIlluminance();
float getPowerConsumption();
unsigned long getElapsedTime();
void startStream(const String &var, int index);
void stopStream(const String &var, int index);
String getLastMinuteBuffer(const String &var, int index);
void changeState(LuminaireState newState);

#endif // GLOBALS_H