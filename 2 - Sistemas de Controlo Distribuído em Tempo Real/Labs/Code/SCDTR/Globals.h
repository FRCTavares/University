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

#define MAX_CALIB_NODES 6 // Maximum number of nodes in calibration matrix

#define MAX_TRACKED_NODES 16 // Maximum number of nodes to track in the network

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

// Define calibration control command types
#define CAL_CMD_INIT 101         // Initialize calibration
#define CAL_CMD_ACK 102          // Acknowledge calibration
#define CAL_CMD_START_NODE 103   // Start calibrating a specific node
#define CAL_CMD_SEND_READING 104 // Send light reading
#define CAL_CMD_NEXT_NODE 105    // Move to next node
#define CAL_CMD_COMPLETE 106     // Calibration complete

// Constants for calibration process
const int CAL_TIMEOUT_MS = 15000;           // Wait time for acknowledgments
const int CAL_STABILIZE_TIME_MS = 1000;     // Wait time for light stabilization
const int CAL_CHECK_INTERVAL_MS = 100;      // Interval to check stabilization
const float CAL_STABILITY_THRESHOLD = 0.05; // Threshold for determining stability

/**
 * Illuminance setpoints for different states
 */
extern const float SETPOINT_OFF;        // Off state target (lux)
extern const float SETPOINT_UNOCCUPIED; // Unoccupied state target (lux)
extern const float SETPOINT_OCCUPIED;   // Occupied state target (lux)

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
  uint8_t nodeId;          // Unique ID for this device (used for CAN communication)
  float ledGain;           // Calibrated LED contribution gain
  float calibrationOffset; // Calibration offset for sensor

  // PID tuning parameters
  float pidKp;   // Proportional gain
  float pidKi;   // Integral gain
  float pidBeta; // Setpoint weighting factor
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
  float setpointLux;                // Target illuminance level in lux
  float dutyCycle;                  // Current LED duty cycle [0-1]
  LuminaireState luminaireState;    // Current luminaire operating state
  bool feedbackControl;             // Flag for automatic feedback control (true) vs manual (false)
  bool antiWindup;                  // PID controller anti-windup flag
  bool standbyMode;                 // Flag to indicate if node is in standby mode waiting for wake up
  bool systemAwake;                 // System is in wake-up discovery phase
  bool systemReady;                 // System is fully initialized and operational
  unsigned long discoveryStartTime; // When node discovery phase started
};

// Calibration matrix and parameters

struct CalibrationMatrix
{
  float gains[MAX_CALIB_NODES][MAX_CALIB_NODES]; // [i][j] = effect of node j on node i
  float externalLight[MAX_CALIB_NODES];          // External light contribution at each node
  uint8_t nodeIds[MAX_CALIB_NODES];              // Node IDs corresponding to matrix indices
  int numNodes;                                  // Number of nodes in calibration
};

/**
 * CommState holds state parameters for communication between cores and nodes.
 */
struct CommState
{
  // Streaming parameters
  bool streamingEnabled;        // Enable periodic streaming
  char *streamingVar;           // Identifier of variable being streamed
  int streamingIndex;           // Index of the node for streaming
  unsigned long lastStreamTime; // Last time a stream was sent

  // Remote streaming requests from other nodes
  StreamRequest remoteStreamRequests[MAX_STREAM_REQUESTS];

  // Atomic flag to indicate new CAN messages are available
  std::atomic_flag hasNewMessages;

  // CAN communication flags
  bool periodicCANEnabled; // Enable periodic CAN transmissions
  bool canMonitorEnabled;  // Enable monitoring of CAN messages

  // Calibration status
  bool isCalibrationMaster;             // Is this node the calibration master
  bool calibrationInProgress;           // Is calibration currently in progress
  uint8_t calibrationStep;              // Current step in the calibration sequence
  unsigned long calLastStepTime;        // Timestamp of last calibration step
  uint8_t currentCalNode;               // Current node being calibrated
  bool waitingForAcks;                  // Waiting for acknowledgments flag
  uint8_t acksReceived;                 // Number of acknowledgments received
  CalibrationMatrix calibMatrix;        // Matrix of calibration gains
  float luxReadings[MAX_CALIB_NODES];   // Temporary storage for lux readings during calibration
  unsigned long stabilizationStartTime; // When we started waiting for light levels to stabilize
  bool measurementsStable;              // Flag indicating if measurements are stable
  float previousReadings[5];            // Store previous readings to check for stability
  int readingIndex;                     // Index for circular buffer of readings
  int ourNodeIndex;                     // Index of this node in calibration matrix
};

// Maximum number of neighbors to track
#define MAX_NEIGHBORS 5

// Structure to store neighbor information
struct NeighborInfo
{
  uint8_t nodeId;           // CAN node ID
  float lastLux;            // Last reported illuminance
  float lastDuty;           // Last reported duty cycle
  LuminaireState state;     // Current operating state
  unsigned long lastUpdate; // Last update timestamp
  bool isActive;            // Is node currently active
};

// Array of neighbor information
extern NeighborInfo neighbors[MAX_NEIGHBORS];

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