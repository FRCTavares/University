#ifndef GLOBALS_H
#define GLOBALS_H

#include <Arduino.h>
#include "pico/critical_section.h"
#include <atomic>

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

#define VCC 3.3                  // Supply voltage for analog reference
#define MY_ADC_RESOLUTION 4095.0 // 12-bit ADC resolution
#define FIXED_RESISTOR 10000.0   // Fixed resistor in voltage divider (ohms)

/**
 * Maximum number of stream requests that can be tracked simultaneously
 */
#define MAX_STREAM_REQUESTS 5

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
// STRUCTURE DEFINITIONS
//=============================================================================

/**
 * Structure to track a request to stream data to another node
 */
struct StreamRequest
{
  bool active;            // Is this request currently active?
  uint8_t requesterNode;  // Node ID requesting the data
  int variableType;       // Type of variable to stream (maps to sensor types)
  unsigned long lastSent; // Timestamp of last transmission
};

/**
 * DeviceConfig holds configuration parameters for the device.
 */
struct DeviceConfig
{
  uint8_t nodeId;           // Unique ID for this device (used for CAN communication)
  float ledGain;            // Calibrated LED contribution gain
  float calibrationOffset;  // Calibration offset for sensor
  
  // PID tuning parameters
  float pidKp;              // Proportional gain
  float pidKi;              // Integral gain
  float pidBeta;            // Setpoint weighting factor
};

/**
 * SensorState holds the current sensor readings and processing state.
 */
struct SensorState
{
  float rawLux;              // Raw unfiltered illuminance value from sensor
  float filteredLux;         // Current filtered illuminance value
  float lastFilteredLux;     // Previous filtered reading for EMA calculation
  float baselineIlluminance; // Baseline illuminance measurement with LED off
  float externalLuxAverage;  // Estimated external light contribution
  bool filterEnabled;        // Flag to indicate if filtering is enabled
};

/**
 * ControlState holds control parameters for lighting.
 */
struct ControlState
{
  float setpointLux;             // Target illuminance level in lux
  float dutyCycle;               // Current LED duty cycle [0-1]
  LuminaireState luminaireState; // Current luminaire operating state
  bool feedbackControl;          // Flag for automatic feedback control (true) vs manual (false)
  bool antiWindup;               // PID controller anti-windup flag
};

/**
 * CommState holds state parameters for communication between cores and nodes.
 */
struct CommState
{
  bool streamingEnabled;        // Enable periodic streaming
  char* streamingVar;           // Identifier of variable being streamed
  int streamingIndex;           // Index of the node for streaming
  unsigned long lastStreamTime; // Last time a stream was sent
  
  // Remote streaming requests from other nodes
  StreamRequest remoteStreamRequests[MAX_STREAM_REQUESTS]; 
  
  // Atomic flag to indicate new CAN messages are available
  std::atomic_flag hasNewMessages; 
};

//=============================================================================
// GLOBAL VARIABLE DECLARATIONS
//=============================================================================

/**
 * Synchronization primitive for thread-safe access to shared data
 */
extern critical_section_t commStateLock;

/**
 * Global device configuration
 * Contains calibration parameters and device-specific settings
 */
extern DeviceConfig deviceConfig;

/**
 * Global sensor state
 * Contains current and processed sensor readings
 */
extern SensorState sensorState;

/**
 * Global control system state
 * Contains control parameters and operating mode information
 */
extern ControlState controlState;

/**
 * Global communication state
 * Contains state parameters for inter-core and inter-node communication
 */
extern CommState commState;

/**
 * Global PI controller instance
 * Handles feedback control of illuminance
 */
extern class PIController pid;

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

//=============================================================================
// FUNCTION DECLARATIONS
//=============================================================================

/**
 * Get estimated external illuminance (without LED contribution)
 * @return Estimated external illuminance in lux
 */
float getExternalIlluminance();

/**
 * Get elapsed time since boot
 * @return Time in seconds
 */
unsigned long getElapsedTime();

/**
 * Set LED brightness using PWM duty cycle
 * @param newDutyCycle Duty cycle value between 0.0 (off) and 1.0 (fully on)
 */
void setLEDDutyCycle(float newDutyCycle);

/**
 * Set LED brightness using percentage
 * @param percentage Brightness percentage between 0.0 (off) and 100.0 (fully on)
 */
void setLEDPercentage(float percentage);

/**
 * Set LED brightness based on desired power consumption
 * @param powerWatts Desired power in watts from 0.0 to MAX_POWER_WATTS
 */
void setLEDPower(float powerWatts);

/**
 * Change luminaire operating state
 * @param newState New operating state to set
 */
void changeState(LuminaireState newState);

#endif // GLOBALS_H