#ifndef PIDCONTROLLER_H
#define PIDCONTROLLER_H

#include <Arduino.h>
#include <math.h>

class PIDController {
public:
    PIDController(float kp, float ki, float kd, float n, float samplingTime);
    float compute(float setpoint, float measurement);
    void reset();
    void setGains(float kp, float ki, float kd);
    float getSamplingTime() const;  // Added getter for sampling time
    float readLux();

private:
    float Kp;
    float Ki;
    float Kd;
    float N;         // Filter coefficient for derivative term
    float h;         // Sampling time
    float Iterm;
    float Dterm;
    float e_old;
};

#endif
