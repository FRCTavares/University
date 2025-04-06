#ifndef SENSORMANAGER_H
#define SENSORMANAGER_H

#include <Arduino.h>
#include "Globals.h"

/**
 * Read and process illuminance with multi-stage filtering:
 * 1. Multiple samples to reduce noise
 * 2. Statistical outlier rejection
 * 3. EMA filtering for temporal smoothing
 * 4. Calibration offset application
 *
 * @return Processed illuminance value in lux
 */
float readLux();

/**
 * Calibrate LUX sensor using a reference measurement
 *
 * @param knownLux Reference illuminance from trusted external meter
 */
void calibrateLuxSensor(float knownLux);

/**
 * Get raw voltage at LDR sensing pin
 *
 * @return Voltage at LDR pin (0-VCC)
 */
float getVoltageAtLDR();

/**
 * Get estimated external illuminance (without LED contribution)
 * Calculates background illuminance by subtracting LED contribution from total
 *
 * @return Estimated external illuminance in lux
 */
float getExternalIlluminance();

/**
 * Adapt control system to external light changes
 * Uses a feedforward approach to assist the PID controller
 */
void adaptToExternalLight();

/**
 * Calibrate illuminance model by measuring LED contribution
 * Measures illuminance with LED off and on to calculate system gain
 *
 * @return Calibrated gain value (y2-y1)
 */
float calibrateIlluminanceModel();

/**
 * Perform comprehensive system calibration:
 * 1. Calibrate LDR sensor accuracy
 * 2. Measure LED contribution for external illuminance calculation
 *
 * @param referenceValue The reference illuminance value
 * @return Calibrated LED gain value (G)
 */
float calibrateSystem(float referenceValue);

#endif // SENSORMANAGER_H