/**
 * Distributed Lighting Control System
 *
 * This system implements an intelligent distributed lighting controller with:
 * - Illuminance sensing and filtering
 * - PID-based feedback control
 * - CAN bus communication between nodes
 * - Neighbor state coordination for energy efficiency
 * - Multiple operating modes (off, unoccupied, occupied)
 * - Serial command interface for configuration and monitoring
 * - Data logging and streaming capabilities
 */
#include <Arduino.h>
#include <math.h>
#include <map>
#include "pico/multicore.h"

#include "Globals.h"
#include "CANComm.h"
#include "CommandInterface.h"
#include "DataLogger.h"
#include "LEDDriver.h"
#include "Metrics.h"
#include "PIController.h"
#include "SensorManager.h"

//=============================================================================
// INTER-CORE MESSAGING
//=============================================================================

// Message types for inter-core communication
#define MSG_SEND_SENSOR 1
#define MSG_SEND_CONTROL 2
#define MSG_UPDATE_STATE 3

// Constants for consensus protocol
#define CONVERGENCE_THRESHOLD 0.1   // Max difference allowed for convergence
#define MAX_CONSENSUS_ROUNDS 3       // Maximum consensus rounds before forcing update
#define CONSENSUS_TIMEOUT_MS 2000    // Timeout for waiting on proposals

// Message structure for inter-core communication
struct CoreMessage
{
  uint8_t msgType;  // Message type identifier
  uint8_t dataType; // Type of data being sent
  float value;      // Value to send
  uint8_t nodeId;   // Target or source node ID
};

// Queues for communication between cores
queue_t core0to1queue;
queue_t core1to0queue;

//=============================================================================
// GLOBAL SYSTEM STATE
//=============================================================================

// System state structures (see Globals.h for definitions)
critical_section_t commStateLock;
DeviceConfig deviceConfig;
SensorState sensorState;
ControlState controlState;
CommState commState;

//=============================================================================
// HARDWARE CONFIGURATION
//=============================================================================

// Pin assignments
const int LED_PIN = 15; // PWM output for LED driver

// PID controller configuration
const float Kp = 20;    // Initial proportional gain (will be replaced by ledGain)
const float Ki = 400;   // Integral gain
const float DT = 0.01;  // Sampling period (seconds)
const float BETA = 0.8; // Setpoint weighting factor (0.0-1.0)
PIController pid(Kp, Ki, BETA, DT);

//=============================================================================
// SYSTEM SETTINGS
//=============================================================================

// Illuminance setpoints for different states
const float SETPOINT_OFF = 0.0;
const float SETPOINT_UNOCCUPIED = 20.0;
const float SETPOINT_OCCUPIED = 30.0;

// CAN monitoring flag
bool canMonitorEnabled = false;

//=============================================================================
// NEIGHBOR MANAGEMENT
//=============================================================================

NeighborInfo neighbors[MAX_NEIGHBORS];

// =============================================================================
// ADMM CONSENSUS PROTOCOL VARIABLES
// =============================================================================

// These variables track the ADMM consensus state
static bool is_running_admm_consensus = false;
static bool is_first_admm_iteration = false;
static int admm_consensus_iteration = 0;
static int max_admm_iterations = 10;
static std::map<std::pair<int, uint8_t>, float> d_received_values;
enum admm_consensus_stage_t { ADMM_CONSENSUS_ITERATION, ADMM_WAIT_FOR_MESSAGES };
static admm_consensus_stage_t admm_consensus_stage;
const float maxGain = 30.0f; // Maximum expected gain value


//=============================================================================
// SYSTEM UTILITY FUNCTIONS
//=============================================================================

/**
 * Change the luminaire operating state
 * Updates system state and broadcasts change to network
 *
 * @param newState New state to set (OFF, UNOCCUPIED, OCCUPIED)
 */
void changeState(LuminaireState newState)
{
  // Don't do anything if state is unchanged
  critical_section_enter_blocking(&commStateLock);
  if (newState == controlState.luminaireState)
  {
    critical_section_exit(&commStateLock);
    return;
  }

  controlState.luminaireState = newState;

  // Update setpoint based on new state
  switch (controlState.luminaireState)
  {
  case STATE_OFF:
    controlState.setpointLux = SETPOINT_OFF;
    controlState.feedbackControl = false;
    // Only set duty cycle to 0 if we're actually transitioning to OFF state
    controlState.dutyCycle = 0.0;
    break;

  case STATE_UNOCCUPIED:
    controlState.setpointLux = SETPOINT_UNOCCUPIED;
    controlState.feedbackControl = true;
    break;

  case STATE_OCCUPIED:
    controlState.setpointLux = SETPOINT_OCCUPIED;
    controlState.feedbackControl = true;
    break;
  }
  critical_section_exit(&commStateLock);

  // Reset PID controller to avoid integral windup during transitions
  pid.reset();

  // Send to CAN bus using core0
  CoreMessage msg;
  msg.msgType = MSG_SEND_CONTROL;
  msg.nodeId = CAN_ADDR_BROADCAST;
  msg.dataType = CAN_CTRL_STATE_CHANGE;
  msg.value = (float)controlState.luminaireState;
  queue_add_blocking(&core1to0queue, &msg);
}

/**
 * Get estimated power consumption based on current duty cycle
 *
 * @return Estimated power in watts
 */
float getPowerConsumption()
{
  critical_section_enter_blocking(&commStateLock);
  float duty = controlState.dutyCycle;
  critical_section_exit(&commStateLock);

  return duty * MAX_POWER_WATTS;
}

/**
 * Get elapsed time since boot
 *
 * @return Time in seconds
 */
unsigned long getElapsedTime()
{
  return millis() / 1000;
}

/**
 * Apply device configuration settings
 * Updates global variables and configures controller
 *
 * @param config Configuration to apply
 */
void applyDeviceConfig(const DeviceConfig &config)
{
  deviceConfig = config; // Store the configuration

  // Configure PID controller
  pid.setGains(config.pidKp, config.pidKi);
  pid.setWeighting(config.pidBeta);

  Serial.print("Configured as device #");
  Serial.print(config.nodeId);
  Serial.print(", Kp=");
  Serial.print(config.pidKp);
  Serial.print(", Ki=");
  Serial.print(config.pidKi);
  Serial.print(", Beta=");
  Serial.print(config.pidBeta);
  Serial.println();
}

/**
 * Configure device based on hardware ID
 * Sets node ID and PID parameters based on unique board ID
 */
void configureDeviceFromID()
{
  // Read hardware ID
  pico_unique_board_id_t board_id;
  pico_get_unique_board_id(&board_id);

  // Generate unique node ID from the last byte (1-63)
  uint8_t nodeId = board_id.id[7] & 0x3F; // Use last 6 bits for node ID (0-63)

  // Save existing calibration values
  float existingGain = deviceConfig.ledGain;
  float existingOffset = deviceConfig.calibrationOffset;

  // Start with default configuration
  DeviceConfig config;

  config.nodeId = nodeId;

  // Apply device-type specific PID parameters
  switch (config.nodeId)
  {
  case 33: // First device type
    config.pidKp = 20.0;
    config.pidKi = 400.0;
    config.pidBeta = 0.8;
    break;

  case 40: // Second device type
    config.pidKp = 25.0;
    config.pidKi = 350.0;
    config.pidBeta = 0.75;
    break;

  case 52: // Third device type
    config.pidKp = 22.0;
    config.pidKi = 380.0;
    config.pidBeta = 0.82;
    break;
  }

  // Preserve calibration values
  config.ledGain = existingGain;
  config.calibrationOffset = existingOffset;

  // Apply the configuration
  applyDeviceConfig(config);
}

//===========================================================================================================================================================
// Distributed Consensus Protocol NO OPTIMIZATION
//===========================================================================================================================================================

/**
 * Check if enough iterations have passed to apply the duty cycle
 * 
 * @return true if iteration count has reached the target
 */
bool shouldApplyProposal() {
  critical_section_enter_blocking(&commStateLock);
  uint8_t iterations = controlState.iterationCounter;
  critical_section_exit(&commStateLock);
  return iterations >= 10;
}

/**
 * Increment the iteration counter only if neighbor proposals have changed
 */
void incrementIterationCounterIfStale() {
  static float lastNeighborSum = -1.0;
  float currentSum = 0.0;

  critical_section_enter_blocking(&commStateLock);
  for (int i = 0; i < MAX_NEIGHBORS; i++) {
    if (neighbors[i].isActive && neighbors[i].hasProposal) {
      currentSum += neighbors[i].pendingDuty;
    }
  }
  critical_section_exit(&commStateLock);

  if (fabs(currentSum - lastNeighborSum) > 0.01f) {
    lastNeighborSum = currentSum;
    critical_section_enter_blocking(&commStateLock);
    controlState.iterationCounter++;
    critical_section_exit(&commStateLock);
  }
}

/**
 * Reset consensus state for the next round
 */
void resetConsensusState() {
  critical_section_enter_blocking(&commStateLock);
  controlState.consensusRound = 0;
  controlState.iterationCounter = 0;
  controlState.consensusReached = false;
  for (int i = 0; i < MAX_NEIGHBORS; i++) {
    if (neighbors[i].isActive) {
      neighbors[i].hasProposal = false;
    }
  }
  critical_section_exit(&commStateLock);
}

/**
 * Update luminaire duty cycle using coordinated control with iteration-based consensus
 * This function implements a decentralized control strategy with a two-phase approach:
 * 1. Calculate and propose duty cycle values
 * 2. Wait for 10 iterations before committing the change
 * 
 * @return true if duty cycle was updated, false otherwise
 */
bool updateCoordinatedControl() {
  static unsigned long lastUpdateTime = 0;
  unsigned long currentTime = millis();

  if (currentTime - lastUpdateTime < 1000) {
    critical_section_enter_blocking(&commStateLock);
    float pendingDuty = controlState.pendingDutyCycle;
    uint8_t consensusRound = controlState.consensusRound;
    unsigned long lastProposalTime = controlState.lastProposalTime;
    critical_section_exit(&commStateLock);

    if (consensusRound > 0 && (shouldApplyProposal() || (currentTime - lastProposalTime > CONSENSUS_TIMEOUT_MS))) {
      setLEDDutyCycle(pendingDuty);

      float currentLux, setpointLux;
      critical_section_enter_blocking(&commStateLock);
      currentLux = sensorState.filteredLux;
      setpointLux = controlState.setpointLux;
      controlState.dutyCycle = pendingDuty;
      critical_section_exit(&commStateLock);
      resetConsensusState();
      return true;
    }

    if (consensusRound > 0) {
      incrementIterationCounterIfStale();
    }

    return false;
  }

  uint8_t myNodeId;
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  bool feedbackEnabled = controlState.feedbackControl;
  LuminaireState currentState = controlState.luminaireState;
  uint8_t consensusRound = controlState.consensusRound;
  critical_section_exit(&commStateLock);

  critical_section_enter_blocking(&commStateLock);
  float currentDuty = controlState.dutyCycle;
  critical_section_exit(&commStateLock);
  sendSensorReading(CAN_ADDR_BROADCAST, 1, currentDuty);

  if (currentState == STATE_OFF) {
    lastUpdateTime = currentTime;
    return false;
  }

  int timeSlot = (currentTime / 1000) % 3;
  bool isMyTurn = (myNodeId == 33 && timeSlot == 0) || (myNodeId == 40 && timeSlot == 1) || (myNodeId == 52 && timeSlot == 2);

  
  if (isMyTurn) {
    Serial.print("Node ");
    Serial.print(myNodeId);
    Serial.print(" active in time slot ");
    Serial.println(timeSlot);
  }

  if (isMyTurn && consensusRound == 0) {
    int myIndex = -1;
    critical_section_enter_blocking(&commStateLock);
    for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
      if (commState.calibMatrix.nodeIds[i] == myNodeId) {
        myIndex = i;
        break;
      }
    }
    critical_section_exit(&commStateLock);

    if (myIndex < 0) {
      lastUpdateTime = currentTime;
      return false;
    }

    float setpoint, selfGain;
    critical_section_enter_blocking(&commStateLock);
    setpoint = controlState.setpointLux;
    selfGain = commState.calibMatrix.gains[myIndex][myIndex];
    critical_section_exit(&commStateLock);

    float disturbance = sensorState.baselineIlluminance;

    float neighborSum = 0.0;
    critical_section_enter_blocking(&commStateLock);
    for (int j = 0; j < commState.calibMatrix.numNodes; j++) {
      if (j != myIndex) {
        uint8_t neighborId = commState.calibMatrix.nodeIds[j];
        float gain_ij = commState.calibMatrix.gains[myIndex][j];

        float neighbor_duty = 0.0;
        for (int n = 0; n < MAX_NEIGHBORS; n++) {
          if (neighbors[n].nodeId == neighborId && neighbors[n].isActive) {
            neighbor_duty = neighbors[n].lastDuty;
            break;
          }
        }
        neighborSum += gain_ij * neighbor_duty;
      }
    }
    critical_section_exit(&commStateLock);

    float currentLux = sensorState.filteredLux;

    Serial.println("===== Duty Cycle Calculation =====");
    Serial.print("Node: "); Serial.println(myNodeId);
    Serial.print("Setpoint: "); Serial.print(setpoint);
    Serial.print(" lux, Current: "); Serial.print(currentLux);
    Serial.print(" lux, Baseline: "); Serial.println(disturbance);
    Serial.print("Self gain: "); Serial.println(selfGain);
    Serial.print("Neighbor contribution: "); Serial.println(neighborSum);

    float u_ff = (setpoint - disturbance - neighborSum) / selfGain;
    u_ff = constrain(u_ff, 0.0f, 1.0f);
  
    
    // Step 1: Run the feedback controller
    float u_fb_pwm = pid.computeWithFeedforward(setpoint, currentLux, u_ff);

    // Step 2: Add the feedforward component (convert u_ff to PWM range)
    float u_total_pwm = u_fb_pwm + (u_ff * 4095.0f);

    // Step 3: Convert the total PWM output to [0,1] duty cycle
    float u_total = u_total_pwm / 4095.0f;
    u_total = constrain(u_total, 0.0f, 1.0f);

    // After calculating final duty cycle
    Serial.print("Feedforward (u_ff): "); Serial.println(u_ff, 4);
    Serial.print("Feedback (u_fb_pwm): "); Serial.println(u_fb_pwm);
    Serial.print("Total PWM: "); Serial.println(u_total_pwm);
    Serial.print("Final duty cycle: "); Serial.println(u_total, 4);
    Serial.println("===============================");
  
    // Store pending duty cycle and update state
    critical_section_enter_blocking(&commStateLock);
    controlState.pendingDutyCycle = u_total;
    controlState.lastProposalTime = currentTime;
    controlState.consensusRound = 1;
    controlState.iterationCounter = 0;
    for (int i = 0; i < MAX_NEIGHBORS; i++) {
      if (neighbors[i].nodeId == myNodeId) {
        neighbors[i].pendingDuty = u_total;
        neighbors[i].hasProposal = true;
        break;
      }
    }
    critical_section_exit(&commStateLock);

    sendSensorReading(CAN_ADDR_BROADCAST, CAN_SENSOR_DUTY_PROPOSAL, u_total);
  }

  lastUpdateTime = currentTime;
  return false;
}

//===================================================================================================================================
// Distributed Consensus Protocol with Optimization (ADMM)
//===================================================================================================================================

/**
 * Check if ADMM solution is feasible
 * 
 * @param d Map of duty cycles to check
 * @return true if solution is feasible
 */
bool checkFeasibility(const std::map<int, float>& d) {
  const float tol = 0.001; // Tolerance for rounding errors
  
  uint8_t myNodeId;
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  critical_section_exit(&commStateLock);
  
  // DEBUG: Print constraint checking details
  Serial.print("Checking feasibility - duty cycle bounds: ");
  Serial.print(d.at(myNodeId));
  Serial.print(" in [0,1]? ");
  Serial.println((d.at(myNodeId) >= -tol && d.at(myNodeId) <= 1.0 + tol) ? "Yes" : "No");

  // Check if duty cycle is within bounds
  if (d.at(myNodeId) < -tol || d.at(myNodeId) > 1.0 + tol)
    return false;
    
  // Check illuminance constraint
  float d_dot_k = 0.0;
  critical_section_enter_blocking(&commStateLock);
  for (const auto& item : controlState.nodeInfo) {
    d_dot_k += d.at(item.first) * item.second.k;
  }
  float L = controlState.L;
  float o = controlState.o;
  critical_section_exit(&commStateLock);
  
  // DEBUG: Print illuminance constraint details
  Serial.print("Illuminance constraint: ");
  Serial.print(d_dot_k);
  Serial.print(" >= ");
  Serial.print(L - o - tol);
  Serial.print("? ");
  Serial.println((d_dot_k >= L - o - tol) ? "Yes" : "No");

  if (d_dot_k < L - o - tol)
    return false;
    
  return true;
}

/**
 * Evaluate the cost of a solution
 * 
 * @param d Map of duty cycles to evaluate
 * @return Cost value
 */
float evaluateCost(const std::map<int, float>& d) {
  float cost = 0.0;
  
  critical_section_enter_blocking(&commStateLock);
  float rho = controlState.rho;
  for (const auto& item : controlState.nodeInfo) {
    cost += item.second.c * d.at(item.first) + 
            item.second.y * (d.at(item.first) - item.second.d_av) + 
            rho / 2.0 * pow(d.at(item.first) - item.second.d_av, 2);
  }
  critical_section_exit(&commStateLock);
  
  return cost;
}

/**
 * Print a map for debugging
 */
void printMap(const std::map<int, float>& m) {
  Serial.print("{");
  int i = 0;
  for (const auto& pair : m) {
    Serial.print(pair.first);
    Serial.print(": ");
    Serial.print(pair.second, 4);
    i++;
    if (i < m.size())
      Serial.print(", ");
  }
  Serial.print("}");
}

/**
 * Core ADMM optimization algorithm
 * Computes the optimal duty cycle values for all nodes
 * 
 * @return true if a feasible solution was found
 */
bool runADMMIteration() {
  uint8_t myNodeId;
  float n, m, o, L, rho;
  std::map<int, float> d_best;
  std::map<int, float> z;
  float z_dot_k = 0.0;
  
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  n = controlState.n;
  m = controlState.m;
  o = controlState.o;
  L = controlState.L;
  rho = controlState.rho;
  
  // Initialize computation variables
  for (const auto& item : controlState.nodeInfo) {
    d_best[item.first] = -1.0;
    z[item.first] = rho * item.second.d_av - item.second.y - item.second.c;
    z_dot_k += z[item.first] * item.second.k;
  }
  critical_section_exit(&commStateLock);
  
  float cost_best = 1e6; // Large initial value
  
  // -------- Solution 1: Unconstrained minimum --------
  std::map<int, float> d_u;
  for (auto& item : z) {
    d_u[item.first] = 1.0 / rho * item.second;
  }
  
  Serial.print("Unconstrained: ");
  printMap(d_u);
  
  if (checkFeasibility(d_u)) {
    Serial.println(" -> Feasible");
    
    // Store unconstrained solution and we're done
    critical_section_enter_blocking(&commStateLock);
    for (auto& item : controlState.nodeInfo) {
      item.second.d = d_u[item.first];
    }
    critical_section_exit(&commStateLock);
    
    return true;
  } else {
    Serial.println(" -> Not feasible");
  }
  
  // -------- Solution 2: Linear boundary constrained --------
  std::map<int, float> d_bl;
  
  critical_section_enter_blocking(&commStateLock);
  for (const auto& item : controlState.nodeInfo) {
    d_bl[item.first] = 1.0 / rho * z[item.first] - 
                       item.second.k / n * (o - L + 1.0 / rho * z_dot_k);
  }
  critical_section_exit(&commStateLock);
  
  Serial.print("Boundary linear: ");
  printMap(d_bl);
  
  if (checkFeasibility(d_bl)) {
    Serial.print(" -> Feasible");
    float cost = evaluateCost(d_bl);
    if (cost < cost_best) {
      d_best = d_bl;
      cost_best = cost;
      Serial.println(" -> Best so far");
    } else {
      Serial.println();
    }
  } else {
    Serial.println(" -> Not feasible");
  }
  
  // -------- Solution 3: Constrained to 0 --------
  std::map<int, float> d_b0 = d_u;
  d_b0[myNodeId] = 0.0;
  
  Serial.print("0-constrained: ");
  printMap(d_b0);
  
  if (checkFeasibility(d_b0)) {
    Serial.print(" -> Feasible");
    float cost = evaluateCost(d_b0);
    if (cost < cost_best) {
      d_best = d_b0;
      cost_best = cost;
      Serial.println(" -> Best so far");
    } else {
      Serial.println();
    }
  } else {
    Serial.println(" -> Not feasible");
  }
  
  // -------- Solution 4: Constrained to 1.0 --------
  std::map<int, float> d_b1 = d_u;
  d_b1[myNodeId] = 1.0;
  
  Serial.print("1-constrained: ");
  printMap(d_b1);
  
  if (checkFeasibility(d_b1)) {
    Serial.print(" -> Feasible");
    float cost = evaluateCost(d_b1);
    if (cost < cost_best) {
      d_best = d_b1;
      cost_best = cost;
      Serial.println(" -> Best so far");
    } else {
      Serial.println();
    }
  } else {
    Serial.println(" -> Not feasible");
  }
  
  // -------- Solution 5: Linear and 0 bounded --------
  std::map<int, float> d_l0;
  
  critical_section_enter_blocking(&commStateLock);
  for (const auto& item : controlState.nodeInfo) {
    float k_idx = controlState.nodeInfo[myNodeId].k;
    d_l0[item.first] = 1.0 / rho * z[item.first] - 
                        1.0 / m * item.second.k * (o - L) + 
                        1.0 / rho / m * item.second.k * (k_idx * z[myNodeId] - z_dot_k);
  }
  d_l0[myNodeId] = 0.0;
  critical_section_exit(&commStateLock);
  
  Serial.print("Linear+0: ");
  printMap(d_l0);
  
  if (checkFeasibility(d_l0)) {
    Serial.print(" -> Feasible");
    float cost = evaluateCost(d_l0);
    if (cost < cost_best) {
      d_best = d_l0;
      cost_best = cost;
      Serial.println(" -> Best so far");
    } else {
      Serial.println();
    }
  } else {
    Serial.println(" -> Not feasible");
  }
  
  // -------- Solution 6: Linear and 1.0 bounded --------
  std::map<int, float> d_l1;
  
  critical_section_enter_blocking(&commStateLock);
  for (const auto& item : controlState.nodeInfo) {
    float k_idx = controlState.nodeInfo[myNodeId].k;
    d_l1[item.first] = 1.0 / rho * z[item.first] - 
                        1.0 / m * item.second.k * (o - L + 1.0 * k_idx) + 
                        1.0 / rho / m * item.second.k * (k_idx * z[myNodeId] - z_dot_k);
  }
  d_l1[myNodeId] = 1.0;
  critical_section_exit(&commStateLock);
  
  Serial.print("Linear+1: ");
  printMap(d_l1);
  
  if (checkFeasibility(d_l1)) {
    Serial.print(" -> Feasible");
    float cost = evaluateCost(d_l1);
    if (cost < cost_best) {
      d_best = d_l1;
      cost_best = cost;
      Serial.println(" -> Best so far");
    } else {
      Serial.println();
    }
  } else {
    Serial.println(" -> Not feasible");
  }
  
  // Update with the best solution found
  if (cost_best < 1e6) {
    critical_section_enter_blocking(&commStateLock);

  // Normalize values to ensure they're in [0,1] range
  for (auto& item : controlState.nodeInfo) {
    float normalized = constrain(d_best[item.first], 0.0f, 1.0f);
    if (fabs(normalized - d_best[item.first]) > 0.01f) {
      Serial.print("WARNING: Normalizing node ");
      Serial.print(item.first);
      Serial.print(" from ");
      Serial.print(d_best[item.first], 4);
      Serial.print(" to ");
      Serial.println(normalized, 4);
    }
    item.second.d = normalized;
  }
  
  critical_section_exit(&commStateLock);
  
  Serial.print("Selected solution: ");
  printMap(d_best);
  Serial.println();
  
  return true;
}
  
  // No feasible solution found
  return false;
}

/**
 * Initialize the ADMM algorithm parameters
 */
void initADMMConsensus() {
  uint8_t myNodeId;
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  controlState.usingADMM = true;
  controlState.rho = 0.07; // ADMM step size parameter
  
  // Cost weight should reflect energy savings priority
  controlState.cost = 1.0;
  Serial.print("Cost mode: ");
  Serial.println(controlState.equalCosts ? "EQUAL" : "DIFFERENT");
  critical_section_exit(&commStateLock);
  
  // Clear any previous state
  Serial.println("Initializing node information from calibration matrix:");
  critical_section_enter_blocking(&commStateLock);
  controlState.nodeInfo.clear();
  critical_section_exit(&commStateLock);
  
  // Parse the calibration matrix to set up ADMM
  critical_section_enter_blocking(&commStateLock);
  for (int i = 0; i < commState.calibMatrix.numNodes; i++) {
    uint8_t nodeId = commState.calibMatrix.nodeIds[i];
    NodeInfo info;
    info.d = 0.0;
    info.d_av = 0.0;
    info.y = 0.0;
    info.k = 0.0;
    info.c = 0.0;
    
    // Find our row in the calibration matrix
    for (int j = 0; j < commState.calibMatrix.numNodes; j++) {
      if (commState.calibMatrix.nodeIds[j] == myNodeId) {
        // Get the gain effect of node i on our node
        info.k = commState.calibMatrix.gains[j][i] / 100.0; // Convert to proportion
        Serial.print("Node ");
        Serial.print(nodeId);
        Serial.print(" k = ");
        Serial.println(info.k, 4);
        break;
      }
    }

    // Calculate maxGain for cost normalization
    float maxGain = 0.0;
    for (int j = 0; j < commState.calibMatrix.numNodes; j++) {
      if (commState.calibMatrix.gains[j][j] > maxGain) 
        maxGain = commState.calibMatrix.gains[j][j];
    }

    // Use a minimum value to avoid division by zero
    if (maxGain < 0.1f) maxGain = 1.0f;

    // Set cost weight based on mode
    if (controlState.equalCosts){
      // All nodes have the same cost
      info.c = controlState.cost;
    }
    else{
      // Based on illumination gain (higher gain = higher cost priority)
      info.c = controlState.cost * (1.0 + info.k / maxGain);
    }
    Serial.print("Node ");
    Serial.print(nodeId);
    Serial.print(" cost = ");
    Serial.println(info.c, 4);

    // Set the cost for this node
    controlState.nodeInfo[nodeId] = info;
  }
  
  // Calculate n and m parameters
  float n = 0.0;
  for (const auto& item : controlState.nodeInfo) {
    n += item.second.k * item.second.k;
  }
  controlState.n = n;
  
  float self_gain_squared = pow(controlState.nodeInfo[myNodeId].k, 2);
  controlState.m = n - self_gain_squared;
  
  // Get external illuminance
  controlState.o = sensorState.baselineIlluminance;
  
  // Set lower bound based on setpoint
  controlState.L = controlState.setpointLux;
  critical_section_exit(&commStateLock);
  
  // Initialize the consensus process
  is_running_admm_consensus = true;
  is_first_admm_iteration = true;
  admm_consensus_stage = ADMM_CONSENSUS_ITERATION;
  d_received_values.clear();
  admm_consensus_iteration = 0;
  
  Serial.println("ADMM consensus initialized");
}

/**
 * Send our duty cycle proposals to all nodes
 */
void broadcastADMMValues() {
  uint8_t myNodeId;
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  
  // For each node in our node info, send our proposal
  for (const auto& item : controlState.nodeInfo) {
    // Prepare duty cycle data: first byte is target node ID
    uint8_t data[5];
    data[0] = item.first;
    
    // Next 4 bytes are float value
    float value = item.second.d;
    memcpy(data+1, &value, sizeof(value));
    
    // Send the message
    sendSensorReading(CAN_ADDR_BROADCAST, CAN_SENSOR_DUTY_PROPOSAL, *((float*)data));
    
    Serial.print("Sent ADMM proposal to node ");
    Serial.print(item.first);
    Serial.print(": ");
    Serial.println(value, 4);
  }
  critical_section_exit(&commStateLock);
}

/**
 * Process received ADMM consensus values from other nodes
 * 
 * @param sourceNodeId Node ID that sent the message
 * @param data Byte array containing the message data
 */
void processADMMMessage(uint8_t sourceNodeId, const uint8_t* data) {
  if (!is_running_admm_consensus) return;
  
  // First byte is the target node ID
  uint8_t targetNodeId = data[0];
  
  // Next 4 bytes are the float value
  float value;
  memcpy(&value, data+1, sizeof(value));
  
  // Store in our map with pair key
  d_received_values[{sourceNodeId, targetNodeId}] = value;
  
  Serial.print("ADMM MSG: Node ");
  Serial.print(sourceNodeId);
  Serial.print(" proposes for node ");
  Serial.print(targetNodeId);
  Serial.print(": d=");
  Serial.println(value, 4);
}

/**
 * Check if we have received all expected ADMM messages
 * 
 * @return true if all messages received
 */
bool haveAllADMMMessages() {
  int expected = 0;
  int active_nodes = 0;
  
  critical_section_enter_blocking(&commStateLock);
  active_nodes = controlState.nodeInfo.size();
  critical_section_exit(&commStateLock);
  
  // Each node sends proposals for all nodes
  expected = (active_nodes - 1) * active_nodes;
  
  return d_received_values.size() >= expected;
}

void testADMMConvergence() {
  Serial.println("\n==== ADMM CONVERGENCE TEST ====");
  
  // Test with different costs
  bool testModes[] = {true, false};  // true = equal costs, false = different costs
  
  for (int mode = 0; mode < 2; mode++) {
    // Set cost mode
    critical_section_enter_blocking(&commStateLock);
    controlState.equalCosts = testModes[mode];
    critical_section_exit(&commStateLock);
    
    Serial.print("\nTesting with ");
    Serial.print(testModes[mode] ? "EQUAL" : "DIFFERENT");
    Serial.println(" costs");
    
    // Init ADMM and run for several iterations
    initADMMConsensus();
    
    for (int iter = 0; iter < 10; iter++) {
      Serial.print("\nIteration ");
      Serial.println(iter+1);
      
      bool result = runADMMIteration();
      
      Serial.print("Found feasible solution: ");
      Serial.println(result ? "YES" : "NO");
      
      // Print current state
      critical_section_enter_blocking(&commStateLock);
      Serial.println("Current duty cycles:");
      for (const auto& item : controlState.nodeInfo) {
        Serial.print("Node ");
        Serial.print(item.first);
        Serial.print(": ");
        Serial.println(item.second.d, 4);
      }
      critical_section_exit(&commStateLock);
      
      delay(500);
    }
  }
  
  Serial.println("\n==== TEST COMPLETE ====");
}

/**
 * Update our duty cycle averages based on received values
 */
void updateADMMDutyAverages() {
  uint8_t myNodeId;
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  
  // For each node ID in our node info
  for (auto& item : controlState.nodeInfo) {
    uint8_t targetNodeId = item.first;
    
    // Start with our own value
    float sum = item.second.d;
    int count = 1;
    
    // Add values from all other nodes
    for (const auto& received : d_received_values) {
      if (received.first.second == targetNodeId) { // If this is for the current target
        sum += received.second;
        count++;
      }
    }
    
    // Calculate average
    if (count > 0) {
      item.second.d_av = sum / count;
      
      Serial.print("Updated average for node ");
      Serial.print(targetNodeId);
      Serial.print(": ");
      Serial.println(item.second.d_av, 4);
    }
  }
  critical_section_exit(&commStateLock);
}

/**
 * Update Lagrangian multipliers for ADMM
 */
void updateADMMLagrangianMultipliers() {
  critical_section_enter_blocking(&commStateLock);
  float rho = controlState.rho;
  
  // For each node in our node info
  for (auto& item : controlState.nodeInfo) {
    // Update Lagrangian multiplier
    item.second.y = item.second.y + rho * (item.second.d - item.second.d_av);
    
    Serial.print("Updated Lagrangian for node ");
    Serial.print(item.first);
    Serial.print(": ");
    Serial.println(item.second.y, 4);
  }
  critical_section_exit(&commStateLock);
}

/**
 * Apply the ADMM consensus result to our control system
 */
void applyADMMResult() {
  uint8_t myNodeId;
  float duty;
  
  critical_section_enter_blocking(&commStateLock);
  myNodeId = deviceConfig.nodeId;
  duty = controlState.nodeInfo[myNodeId].d_av;
  
  // Ensure value is within bounds
  duty = constrain(duty, 0.0f, 1.0f);
  
  // Set as our new duty cycle
  controlState.dutyCycle = duty;
  controlState.pendingDutyCycle = duty;
  
  // Convert to PWM value for LED
  float pwm_duty = duty * 4095.0f;
  critical_section_exit(&commStateLock);
  
  // Apply the duty cycle
  setLEDDutyCycle(duty);
  
  Serial.print("ADMM Final duty cycle: ");
  Serial.println(duty, 4);
  
  // Reset consensus state for next time
  is_running_admm_consensus = false;
}

/**
 * Main ADMM consensus loop function
 * Call this periodically to run the ADMM optimization
 * 
 * @return true if duty cycle was updated
 */
bool updateADMMConsensus() {
  static unsigned long last_run_time = 0;
  unsigned long current_time = millis();
  
  // Only run once per second
  if (current_time - last_run_time < 1000) {
    return false;
  }
  
  last_run_time = current_time;
  
  // Only proceed if we're doing ADMM
  if (!is_running_admm_consensus) {
    // Check if we should initialize ADMM
    critical_section_enter_blocking(&commStateLock);
    bool should_use_admm = controlState.usingADMM && 
                           controlState.luminaireState != STATE_OFF &&
                           commState.calibMatrix.numNodes > 0;
    critical_section_exit(&commStateLock);
    
    if (should_use_admm) {
      initADMMConsensus();
    } else {
      return false;
    }
  }
  
  // Process the current consensus stage
  switch (admm_consensus_stage) {
    case ADMM_CONSENSUS_ITERATION:
      admm_consensus_iteration++;
      Serial.print("\n=== ADMM Iteration ");
      Serial.print(admm_consensus_iteration);
      Serial.println(" ===");
      
      if (is_first_admm_iteration) {
        // First iteration already initialized in initADMMConsensus
        is_first_admm_iteration = false;
      }
      
      // Run the ADMM optimization iteration
      if (runADMMIteration()) {
        // Broadcast our values to other nodes
        broadcastADMMValues();
      } else {
        Serial.println("No feasible solution found!");
      }
      
      // Move to wait state
      admm_consensus_stage = ADMM_WAIT_FOR_MESSAGES;
      d_received_values.clear();
      break;
      
    case ADMM_WAIT_FOR_MESSAGES:
      // Check if we have all messages or timeout
      if (haveAllADMMMessages() || (current_time - last_run_time > 1500)) {
        // Update our averages based on received values
        updateADMMDutyAverages();
        
        // Update Lagrangian multipliers
        updateADMMLagrangianMultipliers();
        
        // Check if we've reached max iterations
        if (admm_consensus_iteration >= max_admm_iterations) {
          // Consensus complete - apply result
          applyADMMResult();
          return true;
        } else {
          // Continue to next iteration
          admm_consensus_stage = ADMM_CONSENSUS_ITERATION;
        }
      }
      break;
  }
  
  return false;
}

//=============================================================================
// MAIN PROGRAM
//=============================================================================

/**
 * Arduino setup function
 * Initializes hardware and subsystems
 */
void setup()
{
  Serial.begin(115200);
  delay(2000); // Allow time for serial connection to establish

  // Configure hardware
  analogReadResolution(12);
  analogWriteFreq(30000);
  analogWriteRange(4095); // Use the constant value directly

  // Initialize subsystems
  initLEDDriver(LED_PIN);
  initStorage();

  // Initialize synchronization primitives
  critical_section_init(&commStateLock);
  queue_init(&core0to1queue, sizeof(CoreMessage), 10);
  queue_init(&core1to0queue, sizeof(CoreMessage), 10);

  // Set initial state values
  sensorState.filterEnabled = true;
  sensorState.lastFilteredLux = -1.0;

  controlState.setpointLux = SETPOINT_OFF;
  controlState.luminaireState = STATE_OFF;
  controlState.feedbackControl = false;
  controlState.antiWindup = true;
  controlState.dutyCycle = 0.0;
  commState.streamingEnabled = false;
  commState.isCalibrationMaster = false;
  commState.calibrationInProgress = false;
  commState.calibrationStep = 0;
  commState.calLastStepTime = 0;
  controlState.systemAwake = false;
  controlState.systemReady = false;
  controlState.discoveryStartTime = 0;

  commState.useDefaultGains = true;  // Default to real calibration

  // Initialize the consensus-related fields
  controlState.pendingDutyCycle = 0.0;
  controlState.consensusRound = 0;
  controlState.consensusReached = false;
  controlState.lastProposalTime = 0;
  controlState.iterationCounter = 0;  // Initialize the iteration counter

  // Initialize ADMM-related state
  controlState.usingADMM = false; // Start with sequential control by default
  controlState.rho = 0.07;
  controlState.cost = 1.0;
  
  // Initialize neighbor proposal flags
  for (int i = 0; i < MAX_NEIGHBORS; i++) {
      neighbors[i].pendingDuty = 0.0;
      neighbors[i].hasProposal = false;
  }

  commState.readingIndex = 0;
  commState.measurementsStable = false;
  for (int i = 0; i < 5; i++)
  {
    commState.previousReadings[i] = 0.0f;
  }

  // Calibrate the system
  deviceConfig.ledGain = 0;
  Serial.println();

  // Configure device based on unique ID
  configureDeviceFromID();

  initCANComm();
  initPendingQueries();

  // Launch core 0 for CAN communication
  multicore_launch_core1(core1_main);

  controlState.standbyMode = true; // Start in standby mode by default
  Serial.println("System starting in standby mode. Send 'wakeup' command to activate.");
}

/**
 * Arduino main loop
 * Processes sensor readings, controls, and communication
 */
void loop()
{
  // ===============================================================
  // SYSTEM STATE MANAGEMENT
  // ===============================================================
  
  // Get current system state
  critical_section_enter_blocking(&commStateLock);
  bool inStandby = controlState.standbyMode;
  bool inWakeUpDiscovery = controlState.systemAwake && !controlState.systemReady;
  bool isSystemReady = controlState.systemReady;
  unsigned long discoveryStartTime = controlState.discoveryStartTime;
  critical_section_exit(&commStateLock);
  
  float lux = readLux();

  // Process serial commands in all states (required for standby wake-up)
  processSerialCommands();

  // Handle standby mode - minimal processing
  if (inStandby) {
    delay(100); // Reduced CPU usage while in standby
    return;     // Skip the rest of the loop
  }

  // Handle wake-up discovery phase
  if (inWakeUpDiscovery) {
    // Check if discovery period has ended (5 seconds)
    if (millis() - discoveryStartTime > 5000) {
      Serial.println("\nNode discovery phase complete");
      displayDiscoveredNodes();

      // Transition to calibration phase
      Serial.println("Starting system calibration as coordinated...");
      critical_section_enter_blocking(&commStateLock);
      controlState.systemAwake = false;
      controlState.systemReady = false; // Will be set true after calibration
      critical_section_exit(&commStateLock);

      // Trigger calibration
      startCalibration();
    }

    // Continue with minimal processing during discovery
    sendHeartbeat();
    delay(50);
    return;
  }

  // ===============================================================
  // INPUT PROCESSING
  // ===============================================================
  
  // Read sensor data - do this early to have fresh data for all functions
  
  
  // Update filtered lux value in the sensor state
  critical_section_enter_blocking(&commStateLock);
  sensorState.filteredLux = lux;
  critical_section_exit(&commStateLock);

  // ===============================================================
  // CALIBRATION MANAGEMENT
  // ===============================================================
  
  // Check calibration status
  critical_section_enter_blocking(&commStateLock);
  bool calibInProgress = commState.calibrationInProgress;
  bool wasCalibrating = commState.wasCalibrating;
  critical_section_exit(&commStateLock);

  // Handle active calibration process
  if (calibInProgress) {
    updateCalibrationState();
    // Skip control during calibration
    logData(millis(), lux, 0.0f);
    delay((int)(pid.getSamplingTime() * 1000));
    return;
  }
  
  // Handle post-calibration initialization
  if (wasCalibrating) {
    handlePostCalibrationSetup();
    return;
  }

  // ===============================================================
  // CONTROL SYSTEM EXECUTION
  // ===============================================================
  
  // Determine which control method to use
  critical_section_enter_blocking(&commStateLock);
  bool useADMM = controlState.usingADMM;
  critical_section_exit(&commStateLock);

  // Execute selected control algorithm
  bool controlUpdateApplied = false;
  if (useADMM) {
    controlUpdateApplied = updateADMMConsensus();
  } else {
    controlUpdateApplied = updateCoordinatedControl();
  }

  // Handle streaming requests
  handleStreaming();
  handleRemoteStreamRequests();

  // ===============================================================
  // DATA LOGGING AND TIMING
  // ===============================================================
  
  // Log data for monitoring and metrics
  logData(millis(), lux, controlState.dutyCycle);

  // Wait for next control cycle
  delay((int)(pid.getSamplingTime() * 1000));
}

/**
 * Handle post-calibration system setup
 * Initializes control state after calibration is complete
 */
void handlePostCalibrationSetup() {
  // Reset calibration flags and set system as ready
  critical_section_enter_blocking(&commStateLock);
  commState.wasCalibrating = false;
  controlState.systemReady = true;
  critical_section_exit(&commStateLock);

  // Set initial state to UNOCCUPIED
  changeState(STATE_UNOCCUPIED);

  // Calculate and apply initial duty cycle
  float initialDuty = SETPOINT_UNOCCUPIED / deviceConfig.ledGain;
  initialDuty = constrain(initialDuty, 0.1f, 0.7f);
  setLEDDutyCycle(initialDuty);

  // Reset PID controller to avoid transients
  pid.reset();

  // Update control state with initial settings
  critical_section_enter_blocking(&commStateLock);
  controlState.dutyCycle = initialDuty;
  critical_section_exit(&commStateLock);

  // Broadcast state change and initial duty cycle to network
  sendControlCommand(CAN_ADDR_BROADCAST, CAN_CTRL_STATE_CHANGE, (float)STATE_UNOCCUPIED);
  sendSensorReading(CAN_ADDR_BROADCAST, 1, initialDuty);

  Serial.println("Calibration complete!");
}


/**
 * Core 0: CAN Communication processing function
 * Runs on the second core and handles all CAN-related operations
 */
void core1_main()
{
  // Add these variables at the top of core1_main
  unsigned long lastHeartbeatTime = 0;
  const unsigned long HEARTBEAT_INTERVAL = 3000; // Send heartbeat every 3 seconds

  // Also declare other timing variables at the top
  unsigned long lastStatusUpdateTime = 0;

  while (true)
  {
    // Process incoming CAN messages
    canCommLoop();

    // Handle periodic heartbeat transmission
    unsigned long currentTime = millis();

    if (currentTime - lastHeartbeatTime >= HEARTBEAT_INTERVAL)
    {
      sendHeartbeat();
      lastHeartbeatTime = currentTime;

      // While we're at it, also update node status (mark inactive nodes)
      updateNodeStatus();
    }

    // Process outgoing messages from Core 0
    CoreMessage msg;
    if (queue_try_remove(&core1to0queue, &msg))
    {
      bool sendResult = false;
      switch (msg.msgType)
      {
      case MSG_SEND_SENSOR:
        sendResult = sendSensorReading(msg.nodeId, msg.dataType, msg.value);
        break;

      case MSG_SEND_CONTROL:
        sendResult = sendControlCommand(msg.nodeId, msg.dataType, msg.value);
        break;

      case MSG_UPDATE_STATE:
        // Handle state updates - this might update local state based on Core 0 request
        critical_section_enter_blocking(&commStateLock);
        switch (msg.dataType)
        {
        case CAN_CTRL_STATE_CHANGE:
          controlState.luminaireState = (LuminaireState)(int)msg.value;
          break;
          // Add other state update types as needed
        }
        critical_section_exit(&commStateLock);
        sendResult = true;
        break;
      }

      // Could report back to Core 0 if needed
    }

    // Update node status every second
    if (currentTime - lastStatusUpdateTime >= 1000)
    {
      updateNodeStatus();
      lastStatusUpdateTime = currentTime;
    }

    // Brief delay to prevent core hogging
    sleep_ms(1);
  }
}