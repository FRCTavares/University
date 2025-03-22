#ifndef METRICS_H
#define METRICS_H

/**
 * Lighting System Performance Metrics Module
 * 
 * This module provides tools for evaluating lighting system performance through
 * various metrics that capture energy efficiency, lighting quality, and comfort.
 * It processes historical data from the circular buffer to calculate:
 * - Energy consumption (power over time)
 * - Visibility error (insufficient illuminance detection)
 * - Flicker (unwanted lighting oscillations)
 * - Overall quality metrics
 * 
 * These metrics enable objective evaluation of system performance and can guide
 * parameter tuning and optimization.
 */

//=============================================================================
// PRIMARY METRICS FUNCTIONS
//=============================================================================

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

/**
 * Calculate average power consumption over the logged period
 * 
 * @return Average power consumption in watts
 */
float computeAveragePowerFromBuffer();

/**
 * Calculate peak power consumption in the logged period
 * 
 * @return Maximum power consumption in watts
 */
float computePeakPowerFromBuffer();

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
 * Calculate illuminance stability metric
 * 
 * Measures how stable the illuminance level remains over time
 * Lower values indicate more stable illuminance
 * 
 * @return Standard deviation of illuminance
 */
float computeIlluminanceStabilityFromBuffer();

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

//=============================================================================
// COMBINED QUALITY METRICS
//=============================================================================

/**
 * Calculate duty cycle stability metric
 * 
 * Measures how stable the duty cycle remains over time
 * Lower values indicate better stability
 * 
 * @return Standard deviation of duty cycle
 */
float computeDutyStabilityFromBuffer();

/**
 * Calculate overall lighting quality index
 * 
 * Combines energy, visibility error, and flicker into a single metric
 * Higher values indicate better overall performance
 * 
 * @return Quality index from 0 (worst) to 100 (best)
 */
float computeQualityIndex();

/**
 * Calculate comfort metric based on illuminance stability and flicker
 * 
 * @return Comfort rating from 0 (poor) to 100 (excellent)
 */
float computeComfortMetric();

//=============================================================================
// STATISTICAL ANALYSIS FUNCTIONS
//=============================================================================

/**
 * Calculate how closely illuminance matches the setpoint over time
 * 
 * @return Mean absolute error between setpoint and measured illuminance
 */
float computeSetpointTrackingError();

/**
 * Calculate illuminance dip ratio
 * Measures the frequency and magnitude of illuminance drops below the setpoint
 * 
 * @return Ratio of samples where illuminance is below setpoint
 */
float computeIlluminanceDipRatio();

#endif // METRICS_H