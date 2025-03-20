#ifndef PIDCONTROLLER_H
#define PIDCONTROLLER_H

#include <Arduino.h>
#include <math.h>

class PIDController
{
public:
    PIDController(float kp, float ki, float kd, float n, float samplingTime);
    float compute(float setpoint, float measurement);
    void reset();
    void setGains(float kp, float ki, float kd);
    float getSamplingTime() const; // Added getter for sampling time
    // Set internal target for coordination purposes
    void setTarget(float newTarget);

private:
    float Kp;    // Proportional gain
    float Ki;    // Integral gain
    float Kd;    // Derivative gain
    float N;     // Filter coefficient for derivative
    float h;     // Sampling time
    float Iterm; // Integral term
    float Dterm; // Derivative term with filtering
    float e_old; // Previous error

    // Make sure these are declared in the header
    float internalTarget; // Target for coordination
    bool useInternalTarget;
};

#endif
