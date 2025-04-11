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

#endif // SENSORMANAGER_H