#ifndef METRICS_H
#define METRICS_H

#include <Arduino.h>

/**
 * Compute and display all lighting quality metrics
 * Calculates energy usage, visibility error, and flicker from logged data
 * and outputs the results to the serial console
 */
void computeAndPrintMetrics();

//=============================================================================
// ENERGY EFFICIENCY METRICS
//=============================================================================

/**
 * Calculate energy consumption from duty cycle history
 *
 * Energy is computed by integrating power over time:
 * E = ∫ P(t) dt
 *
 * Since we have discrete samples, we use:
 * E = Σ (P × Δt)
 *
 * where P = Pmax × duty_cycle
 *
 * @return Total energy consumption in joules
 */
float computeEnergyFromBuffer();

//=============================================================================
// LIGHTING QUALITY METRICS
//=============================================================================

/**
 * Calculate visibility error metric from illuminance history
 *
 * Visibility error measures how much the illuminance falls below
 * the setpoint over time. It's the average of (setpoint - measured)
 * when measured < setpoint, otherwise 0.
 *
 * This metric represents insufficient lighting conditions.
 *
 * @return Average visibility error in lux
 */
float computeVisibilityErrorFromBuffer();

/**
 * Calculate flicker metric from duty cycle history
 *
 * Flicker is computed by detecting direction changes in the
 * duty cycle signal, which indicate oscillations. The method uses
 * three consecutive points to detect when the slope changes sign
 * (indicating a potential oscillation), and measures the magnitude
 * of these changes.
 *
 * @return Average flicker magnitude when direction changes
 */
float computeFlickerFromBuffer();

#endif // METRICS_H