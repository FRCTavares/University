#pragma once

// Initialize the sensor
void initSensor();

// Read illuminance in lux with filtering
float readLux();

// Calibrate sensor with known reference value
void calibrateLuxSensor(float knownLux);

// Get voltage at the LDR pin
float getVoltageAtLDR();

// Get estimated external illuminance
float getExternalIlluminance();

// Adapt to external light changes
void adaptToExternalLight();