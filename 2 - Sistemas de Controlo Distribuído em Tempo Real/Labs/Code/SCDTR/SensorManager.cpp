#include <Arduino.h>
#include <math.h>
#include "pico/multicore.h"

#include "Globals.h"
#include "CANComm.h"
#include "CommandInterface.h"
#include "DataLogger.h"
#include "LEDDriver.h"
#include "Metrics.h"
#include "PIController.h"
#include "SensorManager.h"

// External dependencies
extern void setLEDDutyCycle(float dutyCycle);


//-----------------------------------------------------------------------------
// Sensor Configuration
//-----------------------------------------------------------------------------
// LDR Calibration parameters (for lux conversion)
const float R10 = 225000.0;       // LDR resistance at ~10 lux (ohms)
const float LDR_M = -1.0;         // Slope of log-log resistance vs. illuminance
float LDR_B = log10(R10) - LDR_M; // Y-intercept for log-log conversion

// Pin assignments
const int LDR_PIN = A0; // Analog input for light sensor

// Measurement Filter Configuration
const int NUM_SAMPLES = 10;          // Samples for averaging
const float OUTLIER_THRESHOLD = 2.0; // Standard deviations for outlier detection
const float ALPHA = 0.3;             // EMA filter coefficient (0=slow, 1=fast)

// External light tracking
const float EXT_LUX_ALPHA = 0.05;                   // Slow-moving average coefficient
const float EXTERNAL_LIGHT_CHANGE_THRESHOLD = 1.0f; // Minimum lux change to trigger adaptation

/**
 * Read and process illuminance with multi-stage filtering:
 * 1. Multiple samples to reduce noise
 * 2. Statistical outlier rejection
 * 3. EMA filtering for temporal smoothing
 * 4. Calibration offset application
 *
 * @return Processed illuminance value in lux
 */
float readLux()
{
  bool isFilterEnabled;
  float lastFiltered;

  critical_section_enter_blocking(&commStateLock);
  isFilterEnabled = sensorState.filterEnabled;
  lastFiltered = sensorState.lastFilteredLux;
  critical_section_exit(&commStateLock);

  float samples[NUM_SAMPLES];
  float sum = 0.0;
  float count = 0.0;

  // Always take at least one sample to update rawLux
  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    // Read the ADC value from the analog pin
    int adcValue = analogRead(LDR_PIN);
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;

    // Skip invalid readings
    if (voltage <= 0.0)
    {
      continue;
    }

    // Calculate resistance of LDR using voltage divider formula
    float resistance = FIXED_RESISTOR * (VCC / voltage - 1.0);

    // Convert resistance to LUX using calibration formula
    float logR = log10(resistance);
    float logLux = (logR - LDR_B) / LDR_M;
    float luxValue = pow(10, logLux);

    samples[i] = luxValue;
    sum += luxValue;
    count++;
  }

  if (count == 0)
    return 0.0; // No valid readings

  // Store the raw lux (average of all samples without further filtering)
  critical_section_enter_blocking(&commStateLock);
  sensorState.rawLux = sum / count;
  critical_section_exit(&commStateLock);

  // If filtering is disabled, return the raw value immediately
  if (!isFilterEnabled)
  {
    return sensorState.rawLux;
  }

  // 2. Calculate mean and standard deviation
  float mean = sum / count;
  float variance = 0.0;

  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    if (samples[i] > 0)
    { // Only consider valid samples
      variance += sq(samples[i] - mean);
    }
  }
  float stdDev = sqrt(variance);

  // 3. Filter outliers and recalculate mean
  float filteredSum = 0.0;
  float filteredCount = 0.0;

  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    if (samples[i] > 0 && abs(samples[i] - mean) <= OUTLIER_THRESHOLD * stdDev)
    {
      filteredSum += samples[i];
      filteredCount++;
    }
  }

  float filteredMean = (filteredCount > 0) ? filteredSum / filteredCount : mean;

  // 4. Apply exponential moving average (EMA) filter for temporal smoothing
  if (lastFiltered < 0)
  {
    lastFiltered = filteredMean; // First valid reading
  }
  else
  {
    lastFiltered = ALPHA * filteredMean + (1.0 - ALPHA) * lastFiltered;
  }

  // 5. Apply calibration offset and safety bounds check
  float calibratedLux = lastFiltered + deviceConfig.calibrationOffset;
  if (calibratedLux < 0.0)
    calibratedLux = 0.0;

  // Store in sensorState with proper synchronization
  critical_section_enter_blocking(&commStateLock);
  sensorState.rawLux = sum / count;
  sensorState.lastFilteredLux = calibratedLux;
  critical_section_exit(&commStateLock);

  return calibratedLux;
}

/**
 * Get raw voltage at LDR sensing pin
 *
 * @return Voltage at LDR pin (0-VCC)
 */
float getVoltageAtLDR()
{
  int adcValue = analogRead(LDR_PIN);
  return (adcValue / MY_ADC_RESOLUTION) * VCC;
}

/**
 * Get estimated external illuminance (without LED contribution)
 * Calculates background illuminance by subtracting LED contribution from total
 *
 * @return Estimated external illuminance in lux
 */
float getExternalIlluminance()
{
  float totalLux;
  float ledContribution;
  float currentDuty;

  // Access shared data safely
  critical_section_enter_blocking(&commStateLock);
  totalLux = sensorState.filteredLux;   // Current measured illuminance
  currentDuty = controlState.dutyCycle; // Current LED duty cycle
  float ledGain = deviceConfig.ledGain; // LED contribution factor
  critical_section_exit(&commStateLock);

  // Calculate LED's contribution using duty cycle and gain
  ledContribution = currentDuty * ledGain;

  // Subtract LED contribution from total measured illuminance
  float externalLux = totalLux - ledContribution;

  // Ensure we don't return negative values
  if (externalLux < 0.0f)
    externalLux = 0.0f;

  return externalLux;
}
