#pragma once

class PIDController {
public:
    PIDController(float kp, float ki, float kd, float n, float samplingTime);
    float compute(float setpoint, float measurement);
    void reset();
    void setGains(float kp, float ki, float kd);
    float getSamplingTime() const;
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
    float internalTarget; // Target for coordination
    bool useInternalTarget;
};