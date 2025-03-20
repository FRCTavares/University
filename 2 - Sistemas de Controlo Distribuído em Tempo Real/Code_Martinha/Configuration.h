#pragma once

// System constants
#define MAX_ILLUMINANCE 2000.0f

// Pin assignments
#define LED_PIN 15
#define LDR_PIN A0

// PWM configuration
#define PWM_FREQUENCY 30000
#define PWM_MAX 4095
#define PWM_MIN 0

// Sensor calibration
#define VCC 3.3
#define MY_ADC_RESOLUTION 4095.0
#define FIXED_RESISTOR 10000.0
#define LDR_R10 225000.0
#define LDR_M -1.0

// Filter configuration
#define NUM_SAMPLES 10
#define OUTLIER_THRESHOLD 2.0
#define ALPHA 0.3

// PID parameters
#define PID_KP 28.0
#define PID_KI 230.0
#define PID_KD 0.0
#define PID_N 10.0
#define PID_DT 0.01

// Power model
#define MAX_POWER_WATTS 1.0

// Logging
#define LOG_SIZE 1200

// CAN configuration
#define CAN_CS_PIN 17
#define CAN_MOSI_PIN 19
#define CAN_MISO_PIN 16
#define CAN_SCK_PIN 18

// CAN message types
#define CAN_TYPE_CONTROL 0x00
#define CAN_TYPE_SENSOR 0x01
#define CAN_TYPE_STATUS 0x02
#define CAN_TYPE_CONFIG 0x03
#define CAN_TYPE_ERROR 0x04
#define CAN_TYPE_QUERY 0x05
#define CAN_TYPE_RESPONSE 0x06
#define CAN_TYPE_HEARTBEAT 0x07
#define CAN_CTRL_STATE_CHANGE 0x10

// CAN priority levels
#define CAN_PRIO_HIGH 0x00
#define CAN_PRIO_NORMAL 0x01
#define CAN_PRIO_LOW 0x02
#define CAN_PRIO_LOWEST 0x03

// CAN node addresses
#define CAN_ADDR_BROADCAST 0x00

// State setpoints
#define SETPOINT_OFF 0.0
#define SETPOINT_UNOCCUPIED 5.0
#define SETPOINT_OCCUPIED 15.0