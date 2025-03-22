#include "PIController.h"
#include "Globals.h"
#include <Arduino.h>

//=============================================================================
// CONSTRUCTOR AND INITIALIZATION
//=============================================================================

/**
 * PID Controller Constructor
 * Initializes a PID controller with filtering and anti-windup capabilities
 * 
 * @param kp Proportional gain coefficient
 * @param ki Integral gain coefficient
 * @param kd Derivative gain coefficient
 * @param n Derivative filter coefficient (higher = less filtering)
 * @param samplingTime Control loop sampling time in seconds
 */
PIDController::PIDController(float kp, float ki, float kd, float n, float samplingTime)
    : Kp(kp), Ki(ki), Kd(kd), N(n), h(samplingTime), 
      Iterm(0), Dterm(0), e_old(0),
      internalTarget(0), useInternalTarget(false) 
{
    // All variables are initialized in the initializer list
}

//=============================================================================
// CONTROL COMPUTATION
//=============================================================================

/**
 * Compute PID control action for the current sample
 * 
 * Implements a discretized PID controller:
 * u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt
 * 
 * Features:
 * - Filtered derivative action to reduce noise sensitivity
 * - Anti-windup on integral term to prevent saturation issues
 * - Support for setpoint coordination through internal target
 * 
 * @param setpoint Desired target value (setpoint)
 * @param measurement Current process value (feedback)
 * @return Control action value (typically PWM value)
 */
float PIDController::compute(float setpoint, float measurement)
{
    // Use internal target if set by coordination logic
    float actualSetpoint = useInternalTarget ? internalTarget : setpoint;

    // Calculate control error
    float e = actualSetpoint - measurement;

    // Debug output
    if (DEBUG_MODE && DEBUG_PID)
    {
        Serial.print("PID: SP=");
        Serial.print(actualSetpoint);
        Serial.print(" PV=");
        Serial.print(measurement);
        Serial.print(" e=");
        Serial.println(e);
    }

    //-------------------------------------------------------------------------
    // Proportional Term
    //-------------------------------------------------------------------------
    float Pterm = Kp * e;

    //-------------------------------------------------------------------------
    // Derivative Term with Low-Pass Filtering
    //-------------------------------------------------------------------------
    // Calculate raw derivative of error
    float derivative = (e - e_old) / h;
    
    // Apply first-order filter to derivative term
    // This reduces noise amplification while preserving phase characteristics
    // Filter coefficient: alpha = N*h / (1 + N*h)
    float alpha = N * h;
    Dterm = (alpha * derivative + Dterm) / (1 + alpha);
    
    // Scale by derivative gain
    float D_out = Kd * Dterm;

    //-------------------------------------------------------------------------
    // Calculate Unsaturated Control Action
    //-------------------------------------------------------------------------
    // This is used to check for saturation before applying integral term
    float u_unsat = Pterm + Iterm + D_out;

    //-------------------------------------------------------------------------
    // Anti-Windup for Integral Term
    //-------------------------------------------------------------------------
    // Only integrate if:
    // 1. Control action is not saturated at upper limit OR error is negative (would decrease integral)
    // 2. Control action is not saturated at lower limit OR error is positive (would increase integral)
    const int PWM_MAX = 4095;
    const int PWM_MIN = 0;
    
    if ((u_unsat < PWM_MAX || e < 0) && (u_unsat > PWM_MIN || e > 0))
    {
        // Accumulate integral term using rectangular integration
        Iterm += Ki * e * h;
    }
    // Otherwise, hold integral term at current value (conditional integration)

    // Store error for next iteration's derivative calculation
    e_old = e;
    
    // Return final control action (sum of all terms)
    return Pterm + Iterm + D_out;
}

//=============================================================================
// CONTROLLER MANAGEMENT FUNCTIONS
//=============================================================================

/**
 * Reset controller state
 * Clears integral and derivative terms and resets error history
 * Should be called when control is re-enabled after being off or when 
 * making large changes to setpoint to prevent integral windup
 */
void PIDController::reset()
{
    Iterm = 0;
    Dterm = 0;
    e_old = 0;
}

/**
 * Update controller gain parameters
 * Allows runtime modification of controller behavior
 * 
 * @param kp New proportional gain coefficient
 * @param ki New integral gain coefficient
 * @param kd New derivative gain coefficient
 */
void PIDController::setGains(float kp, float ki, float kd)
{
    Kp = kp;
    Ki = ki;
    Kd = kd;
}

/**
 * Get the controller's sampling time
 * 
 * @return Sampling time in seconds
 */
float PIDController::getSamplingTime() const
{
    return h;
}

/**
 * Set an internal coordination target
 * This allows coordination algorithms to temporarily adjust the target
 * without changing the user-defined setpoint
 * 
 * @param newTarget New internally managed setpoint value
 */
void PIDController::setTarget(float newTarget)
{
    internalTarget = newTarget;
    useInternalTarget = true;
}

/**
 * Restore using the external setpoint
 * Disables the internal target and reverts to using the setpoint
 * passed directly to the compute() method
 */
void PIDController::clearInternalTarget()
{
    useInternalTarget = false;
}

/**
 * Check if controller is currently using an internal target
 * 
 * @return true if using internal target, false if using external setpoint
 */
bool PIDController::isUsingInternalTarget() const
{
    return useInternalTarget;
}

/**
 * Get current values of all PID terms for diagnostics
 * 
 * @param p Output parameter for proportional term
 * @param i Output parameter for integral term
 * @param d Output parameter for derivative term
 */
void PIDController::getTerms(float& p, float& i, float& d) const
{
    // For the most recent control calculation:
    p = Kp * e_old;  // Proportional term
    i = Iterm;       // Integral term
    d = Kd * Dterm;  // Derivative term
}