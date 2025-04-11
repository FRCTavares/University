#ifndef GLOBALS_H
#define GLOBALS_H

/**
 * Globals.h - Shared Definitions and Global Variables
 * 
 * This header centralizes all global variables, constants, and shared function
 * declarations used across the distributed lighting control system. It enables
 * modules to access common state information without circular dependencies.
 */

#include <Arduino.h>

//=============================================================================
// SYSTEM CONSTANTS
//=============================================================================

/**
 * Maximum supported illuminance level in lux
 * Defines the upper limit for illuminance measurements and setpoints
 */
#define MAX_ILLUMINANCE 2000.0f

/**
 * Maximum power consumption of LED at full brightness (watts)
 * Used for power consumption calculations and energy metrics
 */
extern const float MAX_POWER_WATTS;

//=============================================================================
// SYSTEM STATE ENUMERATIONS
//=============================================================================

/**
 * Luminaire operating states
 * Defines the possible operational modes of each lighting node
 */
enum LuminaireState
{
    STATE_OFF = 0,        // Complete shutdown, no illumination
    STATE_UNOCCUPIED = 1, // Low ambient lighting when area is unoccupied
    STATE_OCCUPIED = 2    // Full task lighting when workspace is in use
};

//=============================================================================
// CONTROL SYSTEM STATE
//=============================================================================

/**
 * Current target illuminance level in lux
 * This is the desired light level the system attempts to maintain
 */
extern float setpointLux;

/**
 * Current LED brightness as duty cycle [0.0-1.0]
 * Represents the proportion of time the LED is on during PWM cycle
 */
extern float dutyCycle;

/**
 * Reference illuminance for quality metrics
 * Used to evaluate lighting quality relative to desired level
 */
extern float refIlluminance;

/**
 * Workspace occupancy state flag
 * true = occupied, false = unoccupied
 */
extern bool occupancy;

/**
 * PID controller anti-windup flag
 * Enables/disables integral windup protection
 */
extern bool antiWindup;

/**
 * Feedback control mode flag
 * true = automatic control using illuminance feedback
 * false = manual control using direct duty cycle setting
 */
extern bool feedbackControl;

/**
 * Current luminaire operating state
 * Determines overall behavior mode of the lighting node
 */
extern LuminaireState luminaireState;

//=============================================================================
// DEBUG FLAGS
//=============================================================================

/**
 * Master debug switch
 * Enables/disables all debug output
 */
extern bool DEBUG_MODE;

/**
 * LED driver debug messages
 * Shows detailed information about LED control operations
 */
extern bool DEBUG_LED;

/**
 * Sensor readings debug
 * Shows raw and processed sensor values
 */
extern bool DEBUG_SENSOR;

/**
 * PID controller debug
 * Shows setpoint, measurement, error, and control terms
 */
extern bool DEBUG_PID;

/**
 * Serial plotter output
 * Formats output for Arduino Serial Plotter visualization
 */
extern bool DEBUG_PLOTTING;

//=============================================================================
// CAN COMMUNICATION
//=============================================================================

/**
 * Enable periodic CAN transmission flag
 * When true, node regularly broadcasts its state
 */
extern bool periodicCANEnabled;

/**
 * CAN message monitoring flag
 * When true, all CAN messages are printed to serial
 */
extern bool canMonitorEnabled;

/**
 * This node's CAN ID
 * Unique identifier for this node on the CAN network
 */
extern uint8_t nodeID;

//-----------------------------------------------------------------------------
// CAN Message Types
//-----------------------------------------------------------------------------

/** Control messages (setpoints, modes) */
#define CAN_TYPE_CONTROL 0x00

/** Sensor data (illuminance, duty cycle) */
#define CAN_TYPE_SENSOR 0x01

/** Status information (power, mode) */
#define CAN_TYPE_STATUS 0x02

/** Configuration parameters */
#define CAN_TYPE_CONFIG 0x03

/** Error reports */
#define CAN_TYPE_ERROR 0x04

/** Data requests */
#define CAN_TYPE_QUERY 0x05

/** Responses to queries */
#define CAN_TYPE_RESPONSE 0x06

/** Node presence signals */
#define CAN_TYPE_HEARTBEAT 0x07

/** State change notifications */
#define CAN_CTRL_STATE_CHANGE 0x10

//-----------------------------------------------------------------------------
// CAN Priority Levels
//-----------------------------------------------------------------------------

/** High priority messages (emergency, critical control) */
#define CAN_PRIO_HIGH 0x00

/** Normal priority messages (regular control) */
#define CAN_PRIO_NORMAL 0x01

/** Low priority messages (status updates) */
#define CAN_PRIO_LOW 0x02

/** Lowest priority messages (diagnostics) */
#define CAN_PRIO_LOWEST 0x03

//-----------------------------------------------------------------------------
// CAN Node Addresses
//-----------------------------------------------------------------------------

/** Broadcast address (all nodes) */
#define CAN_ADDR_BROADCAST 0x00

//=============================================================================
// SHARED FUNCTION DECLARATIONS
//=============================================================================

//-----------------------------------------------------------------------------
// Sensor Functions
//-----------------------------------------------------------------------------

/**
 * Read filtered illuminance value
 * @return Current illuminance in lux
 */
float readLux();

/**
 * Get raw voltage at the LDR sensor
 * @return Voltage in volts
 */
float getVoltageAtLDR();

/**
 * Calculate external illuminance contribution
 * @return Estimated external light in lux
 */
float getExternalIlluminance();

//-----------------------------------------------------------------------------
// System Management Functions
//-----------------------------------------------------------------------------

/**
 * Get current power consumption
 * @return Power in watts
 */
float getPowerConsumption();

/**
 * Get system uptime
 * @return Elapsed time in seconds
 */
unsigned long getElapsedTime();

/**
 * Change luminaire operating state
 * @param newState Target state (OFF, UNOCCUPIED, OCCUPIED)
 */
void changeState(LuminaireState newState);

//-----------------------------------------------------------------------------
// Data Streaming Functions
//-----------------------------------------------------------------------------

/**
 * Start streaming a variable to serial
 * @param var Variable identifier (y=illuminance, u=duty, etc.)
 * @param index Node index
 */
void startStream(const String &var, int index);

/**
 * Stop streaming a variable
 * @param var Variable to stop streaming
 * @param index Node index
 */
void stopStream(const String &var, int index);

/**
 * Get historical data as CSV
 * @param var Variable type
 * @param index Node index
 * @return CSV string of historical values
 */
String getLastMinuteBuffer(const String &var, int index);

#define MAX_STREAM_REQUESTS 5
struct StreamRequest {
  uint8_t requesterNode;
  uint8_t variableType;
  bool active;
  unsigned long lastSent;
};
extern StreamRequest remoteStreamRequests[MAX_STREAM_REQUESTS];


#endif // GLOBALS_H