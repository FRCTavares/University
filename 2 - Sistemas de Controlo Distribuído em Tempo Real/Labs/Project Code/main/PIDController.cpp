#include "PIDController.h"
#include "Globals.h"
#include <Arduino.h>
#

PIDController::PIDController(float kp, float ki, float kd, float n, float samplingTime)
    : Kp(kp), Ki(ki), Kd(kd), N(n), h(samplingTime), Iterm(0), Dterm(0), e_old(0) {}


float PIDController::compute(float setpoint, float measurement) {
    float e = setpoint - measurement;

    // Debug output
    if (DEBUG_MODE && DEBUG_PID) {
        Serial.print("PID: SP=");
        Serial.print(setpoint);
        Serial.print(" PV=");
        Serial.print(measurement);
        Serial.print(" e=");
        Serial.println(e);
    }
    
    // Proportional term
    float Pterm = Kp * e;
    
    // Derivative term with filtering
    float derivative = (e - e_old) / h;
    float alpha = N * h;
    Dterm = (alpha * derivative + Dterm) / (1 + alpha);
    float D_out = Kd * Dterm;
    
    // Compute unsaturated control action
    float u_unsat = Pterm + Iterm + D_out;
    
    // Anti-windup: Only integrate if control is not saturated
    const int PWM_MAX = 4095;
    const int PWM_MIN = 0;
    if ((u_unsat < PWM_MAX || e < 0) && (u_unsat > PWM_MIN || e > 0)) {
        Iterm += Ki * e * h;
    }
    
    e_old = e;
    return Pterm + Iterm + D_out;
}

void PIDController::reset() {
    Iterm = 0;
    Dterm = 0;
    e_old = 0;
}

void PIDController::setGains(float kp, float ki, float kd) {
    Kp = kp;
    Ki = ki;
    Kd = kd;
}

float PIDController::getSamplingTime() const {
    return h;
}
