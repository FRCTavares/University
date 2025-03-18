#include "SystemID.h"
#include "CommandInterface.h"
#include "Storage.h"     
#include "Metrics.h"     
#include "CANComm.h"     
#include <Arduino.h>
#include <math.h>
#include "LEDDriver.h"
#include "Globals.h" 


// Add to the top of your existing file:
void runSimulation(float gain, float timeConstant, float delay, float ambient, float dutyCycle) {
    Serial.println("Starting system simulation...");
    SystemSimulator simulator(gain, timeConstant, delay, ambient);
    simulator.simulateStep(dutyCycle, 100, 50);  // 100 steps, 50ms each = 5 seconds
}

// Add this implementation after your existing functions:
void identifyStaticGains() {
    Serial.println("Identifying static gains...");
    const int NUM_POINTS = 5;
    float dutyCycles[NUM_POINTS] = {0.0, 0.25, 0.5, 0.75, 1.0};
    float luxReadings[NUM_POINTS] = {0};
    float gains[NUM_POINTS-1] = {0};
    
    // Turn off LED and measure ambient light
    setLEDDutyCycle(0.0);
    delay(2000);  // Allow time to stabilize
    float ambientLux = readLux();
    Serial.print("Ambient lux: ");
    Serial.println(ambientLux);
    
    for (int i = 0; i < NUM_POINTS; i++) {
      setLEDDutyCycle(dutyCycles[i]);
      delay(2000);  // Allow system to stabilize
      
      // Take multiple readings to ensure accuracy
      float sumLux = 0;
      for (int j = 0; j < 10; j++) {
        sumLux += readLux();
        delay(100);
      }
      luxReadings[i] = sumLux / 10.0;
      
      Serial.print("Duty cycle: ");
      Serial.print(dutyCycles[i]);
      Serial.print(" → Lux: ");
      Serial.println(luxReadings[i]);
      
      // Calculate gain (delta lux / delta duty)
      if (i > 0) {
        float deltaDuty = dutyCycles[i] - dutyCycles[i-1];
        float deltaLux = luxReadings[i] - luxReadings[i-1];
        gains[i-1] = deltaLux / deltaDuty;
        
        Serial.print("Gain at ");
        Serial.print(dutyCycles[i]);
        Serial.print(": ");
        Serial.println(gains[i-1]);
      }
    }
    
    // Calculate average gain
    float avgGain = 0;
    for (int i = 0; i < NUM_POINTS-1; i++) {
      avgGain += gains[i];
    }
    avgGain /= (NUM_POINTS-1);
    
    Serial.print("Average system gain: ");
    Serial.print(avgGain);
    Serial.println(" lux/duty");
  }

// Add SystemSimulator implementation
SystemSimulator::SystemSimulator(float gain, float tau, float d, float ambient) {
    staticGain = gain;
    timeConstant = tau;
    delay = d;
    ambientLux = ambient;
    currentLux = ambient;
}


void SystemSimulator::simulateStep(float dutyCycle, int steps, int stepSize) {
    Serial.println("Simulating step response:");
    Serial.println("Time,Predicted,Actual");
    
    float finalLux = ambientLux + staticGain * dutyCycle;
    
    // Apply step input
    setLEDDutyCycle(dutyCycle);
    
    for (int i = 0; i < steps; i++) {
        // First-order system response: y(t) = final - (final-initial)*e^(-t/tau)
        unsigned long t = i * stepSize;
        float timeSeconds = t / 1000.0;
        float predicted;
        
        // Add delay to model
        if (timeSeconds < (delay / 1000.0)) {
            predicted = ambientLux;  // Before delay, output remains at initial value
        } else {
            // After delay, apply first-order step response 
            float adjustedTime = timeSeconds - (delay / 1000.0);
            predicted = ambientLux + (finalLux - ambientLux) * 
                       (1 - exp(-adjustedTime/(timeConstant/1000.0)));
        }
        
        // Get actual reading for comparison
        float actual = readLux();
        
        Serial.print(t);
        Serial.print(",");
        Serial.print(predicted);
        Serial.print(",");
        Serial.println(actual);
        
        delay(stepSize);
    }
}