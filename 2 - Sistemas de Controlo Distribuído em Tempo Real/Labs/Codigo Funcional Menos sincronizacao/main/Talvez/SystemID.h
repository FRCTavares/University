#ifndef SYSTEMID_H
#define SYSTEMID_H

#include <Arduino.h>

// Function to identify static gains at different operating points
void identifyStaticGains();

// Function to identify system dynamics (time constants, etc)
void identifyDynamics();

// Simulator class to predict system behavior
class SystemSimulator {
private:
    float staticGain;
    float timeConstant;
    float delay;
    float currentLux;
    float ambientLux;
    
public:
    SystemSimulator(float gain, float tau, float d, float ambient);
    void simulateStep(float dutyCycle, int steps, int stepSize);
};

// Helper function to create and run a simulation
void runSimulation(float gain, float timeConstant, float delay, float ambient, float dutyCycle);

#endif // SYSTEMID_H