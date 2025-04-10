#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <Arduino.h>
#include "Globals.h"

//==========================================================================================================================================================
// CALIBRATION INITIALIZATION FUNCTIONS
//==========================================================================================================================================================

/**
 * Start a calibration sequence as the calibration master
 * This node takes control of the calibration process across the network
 *
 * @return true if calibration started successfully
 */
bool startCalibration();

/**
 * Acknowledge a calibration initialization request
 * Sent by nodes when they receive a calibration initialization command
 *
 * @param masterNodeId ID of the calibration master
 */
void acknowledgeCalibration(uint8_t masterNodeId);

//==========================================================================================================================================================
// CALIBRATION PROCESS FUNCTIONS
//==========================================================================================================================================================

/**
 * Process a calibration acknowledgment message
 * Updates the master's tracking of which nodes have acknowledged
 *
 * @param nodeId Node ID that sent the acknowledgment
 * @return true if all expected nodes have acknowledged
 */
bool processCalibrationAck(uint8_t nodeId);

/**
 * Calibrate a specific column of the gain matrix
 * Each column corresponds to one node's LED effect on all nodes
 *
 * @param columnIndex The column index to calibrate (node index in the matrix)
 * @return Average gain value
 */
float calibrateColumn(int columnIndex);

/**
 * Process a calibration reading received from another node
 * Stores the reading in the calibration matrix for processing
 * 
 * @param sourceNodeId ID of the node that sent the reading
 * @param value The illuminance reading value
 */
void processCalibrationReading(uint8_t sourceNodeId, float value);

//==========================================================================================================================================================
// CALIBRATION STATE MACHINE
//==========================================================================================================================================================

/**
 * Update calibration state machine
 * Called periodically from the main loop to advance the calibration process
 */
void updateCalibrationState();

//==========================================================================================================================================================
// DISPLAY FUNCTIONS
//==========================================================================================================================================================

/**
 * Handle an incoming calibration control message
 *
 * @param sourceNodeId ID of the node that sent the message
 * @param controlType Type of calibration command
 * @param value Command parameter value
 */
void handleCalibrationMessage(uint8_t sourceNodeId, uint8_t controlType, float value);

#endif // CALIBRATION_H