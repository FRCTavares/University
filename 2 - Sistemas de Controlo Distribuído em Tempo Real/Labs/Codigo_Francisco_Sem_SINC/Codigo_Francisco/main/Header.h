#include <Arduino.h>
#include <math.h>

// Global Definitions
#define STATE_OFF 0
#define STATE_UNOCCUPIED 1
#define STATE_OCCUPIED 2
#define SETPOINT_OFF 0.0
#define SETPOINT_OCCUPIED 20.0
#define SETPOINT_UNOCCUPIED 2.0
#define MIN_SETPOINT 1.0
#define EXTERNAL_LIGHT_THRESHOLD 1.0
#define EXTERNAL_SCALE_FACTOR 0.5
#define MIN_SETPOINT_CHANGE 0.5
#define SAMPLES 10
#define OUTLIER_THRESHOLD 2.0
#define ALPHA 0.3
#define VCC 3.3                  // Supply voltage for analog reference
#define MY_ADC_RESOLUTION 4095.0 // 12-bit ADC resolution
#define FIXED_RESISTOR 10000.0   // Fixed resistor in voltage divider (ohms)
#define LOG_SIZE 100             // Size of circular log buffer
#define MAX_POWER_WATTS 0.08695  // Maximum power consumption in watts

// LDR Calibration parameters (for lux conversion)
const float R10 = 225000.0;       // LDR resistance at ~10 lux (ohms)
const float LDR_M = -1.0;         // Slope of log-log resistance vs. illuminance
float LDR_B = log10(R10) - LDR_M; // Y-intercept for log-log conversion

//-----------------------------------------------------------------------------
// Pin Assignments
//-----------------------------------------------------------------------------
const int LED_PIN = 15; // PWM output for LED driver
const int LDR_PIN = A0; // Analog input for light sensor
const unsigned int PWM_FREQUENCY = 30000; // 30 kHz frequency

//-----------------------------------------------------------------------------
// PWM Configuration
//-----------------------------------------------------------------------------
const int PWM_MAX = 4095; // Maximum PWM value (12-bit)
const int PWM_MIN = 0;    // Minimum PWM value (off)

//-----------------------------------------------------------------------------
// LDR Sensor Calibration and Lux Conversion
//-----------------------------------------------------------------------------
// Global variables for light measurement and filtering
float lastMeasuredLux = 0.0;
float lastFilteredLux = -1.0;  // -1 indicates first reading
float rawLux = 0.0;
bool filterEnabled = true;
bool adaptiveSetpointEnabled = true;
const float EXT_LUX_ALPHA = 0.1;  // Smoothing factor for external illuminance

