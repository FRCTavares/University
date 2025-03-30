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
#include "pico/critical_section.h" 

//=============================================================================
// SYNCHRONIZATION PRIMITIVES
//=============================================================================

/**
 * Global critical section lock for thread synchronization
 * Used to protect shared data access between cores
 */
extern critical_section_t commStateLock;

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
// DEVICE CONFIGURATION STRUCTURE
//=============================================================================

/**
 * DeviceConfig holds configuration parameters for the device.
 * Replaces the globals:
 *   - ledGain
 *   - calibrationOffset
 */
struct DeviceConfig
{
  float ledGain;           // LED gain value
  float calibrationOffset; // Calibration offset for sensor
};

extern DeviceConfig deviceConfig;

//=============================================================================
// SENSOR STATE STRUCTURE
//=============================================================================

/**
 * SensorState holds the current sensor readings.
 * Replaces the globals:
 *   - rawLux
 *   - baselineIlluminance
 *   - filterEnabled
 */
struct SensorState
{
  float rawLux;              // Raw illuminance value from sensor
  float baselineIlluminance; // Baseline illuminance measurement
  bool filterEnabled;        // Flag to indicate if filtering is enabled
};

extern SensorState sensorState;

//=============================================================================
// CONTROL SYSTEM STATE STRUCTURE
//=============================================================================

/**
 * ControlState holds control parameters for lighting.
 * Replaces the globals:
 *   - setpointLux
 *   - feedbackControl
 *   - antiWindup
 *   - luminaireState
 */
struct ControlState
{
  float setpointLux;             // Target illuminance level in lux
  bool feedbackControl;          // Flag for automatic feedback control
  bool antiWindup;               // PID controller anti-windup flag
  LuminaireState luminaireState; // Current luminaire operating state
};

extern ControlState controlState;

//=============================================================================
// COMMUNICATION STATE STRUCTURE
//=============================================================================

/**
 * CommState holds state parameters for communication.
 * Replaces the globals:
 *   - streamingEnabled
 *   - streamingVar
 *   - streamingIndex
 *   - lastStreamTime
 */
struct CommState
{
  bool streamingEnabled;        // Enable periodic streaming
  String streamingVar;          // Identifier of variable being streamed
  int streamingIndex;           // Index of the node for streaming
  unsigned long lastStreamTime; // Last time a stream was sent
};

extern CommState commState;

//=============================================================================
// OTHER GLOBAL VARIABLES (unchanged)
//=============================================================================

/**
 * Current LED brightness as duty cycle [0.0-1.0]
 * Represents the proportion of time the LED is on during PWM cycle
 */
extern float dutyCycle;

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

// Add this near the other function declarations:
/**
 * Get estimated external illuminance (without LED contribution)
 * @return Estimated external illuminance in lux
 */
float getExternalIlluminance();

#endif // GLOBALS_H
