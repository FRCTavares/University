#ifndef GLOBALS_H
#define GLOBALS_H

#include <Arduino.h>
#include "pico/critical_section.h"
#include <atomic>

//==========================================================================================================================================================
// SYSTEM CONSTANTS
//==========================================================================================================================================================

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
/**
 * Illuminance setpoints for different states
 */
extern const float SETPOINT_OFF;        // Off state target (lux)
extern const float SETPOINT_UNOCCUPIED; // Unoccupied state target (lux)
extern const float SETPOINT_OCCUPIED;   // Occupied state target (lux)

#define MAX_NEIGHBORS 5


//==========================================================================================================================================================
// SYSTEM STATE ENUMERATIONS
//==========================================================================================================================================================

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

enum WakeUpState
{
  WAKEUP_IDLE = 0,        // No wake-up in progress
  WAKEUP_RESET = 1,       // Reset phase - All nodes reset their state
  WAKEUP_ACK_WAIT = 2,    // Waiting for acknowledgments from all nodes
  WAKEUP_CALIBRATE = 3,   // Calibration phase - Measure sensor parameters
  WAKEUP_TRANSITION = 4,  // Transitioning to normal operation
  WAKEUP_COMPLETE = 5     // Wake-up complete, normal operation
};



//==========================================================================================================================================================
// CAN MESSAGE TYPES
//==========================================================================================================================================================

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

/** Wake up protocol messages */
#define CAN_CTRL_WAKEUP_INIT 0x20
#define CAN_CTRL_WAKEUP_ACK 0x21
#define CAN_CTRL_WAKEUP_CALIBRATE 0x22
#define CAN_CTRL_WAKEUP_COMPLETE 0x23


//==========================================================================================================================================================
// STRUCTURE DEFINITIONS
//==========================================================================================================================================================


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
 * Structure to track neighboring nodes in the network
 */
struct NeighborInfo {
  uint8_t nodeId;           // CAN node ID
  float lastLux;            // Last reported illuminance
  float lastDuty;           // Last reported duty cycle
  LuminaireState state;     // Current operating state
  unsigned long lastUpdate; // Last update timestamp
  unsigned long firstSeen;  // When this node was first discovered
  bool isActive;            // Is node currently active
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
  
  // Wake-up protocol state machine fields
  WakeUpState wakeUpState;        // Current wake-up protocol state
  unsigned long wakeUpStateTime;  // Timestamp when current wake-up state started
  bool isWakeUpMaster;            // Whether this node is coordinating the wake-up
};

/**
 * CommState holds state parameters for communication between cores and nodes.
 */
struct CommState
{
  // Streaming parameters
  bool streamingEnabled;         // Enable periodic streaming
  char* streamingVar;            // Identifier of variable being streamed
  int streamingIndex;            // Index of the node for streaming
  unsigned long lastStreamTime;  // Last time a stream was sent
  
  // Remote streaming requests from other nodes
  StreamRequest remoteStreamRequests[MAX_STREAM_REQUESTS]; 
  
  // Atomic flag to indicate new CAN messages are available
  std::atomic_flag hasNewMessages;
  
  // CAN communication flags
  bool periodicCANEnabled;       // Enable periodic CAN transmissions
  bool canMonitorEnabled;        // Enable monitoring of CAN messages
};


//==========================================================================================================================================================
// GLOBAL VARIABLE DECLARATIONS
//==========================================================================================================================================================

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
 * Node fitness variables for coordinator selection
 */
extern float myFitness;
extern float maxFitness;



//==========================================================================================================================================================
// FUNCTION DECLARATIONS
//==========================================================================================================================================================

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