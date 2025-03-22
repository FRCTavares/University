#include "PIController.h"
#include "Globals.h"
#include <Arduino.h>

//=============================================================================
// CONSTRUCTOR AND INITIALIZATION
//=============================================================================

/**
 * PI Controller Constructor
 * Initializes a PI controller with anti-windup capabilities
 * 
 * @param kp Proportional gain coefficient
 * @param ki Integral gain coefficient
 * @param beta Setpoint weighting factor (0.0-1.0)
 * @param samplingTime Control loop sampling time in seconds
 */
PIController::PIController(float kp, float ki, float beta, float samplingTime)
    : Kp(kp), Ki(ki), Beta(beta), h(samplingTime), 
      Iterm(0), e_old(0), Pterm(0),
      internalTarget(0), useInternalTarget(false) 
{
    // All variables are initialized in the initializer list
}

//=============================================================================
// CONTROL COMPUTATION
//=============================================================================

/**
 * Compute PI control action for the current sample
 * 
 * Implements a discretized PI controller with setpoint weighting:
 * u(t) = Kp·(β·r(t) - y(t)) + Ki·∫e(t)dt
 * 
 * Features:
 * - Setpoint weighting to reduce overshoot
 * - Back-calculation anti-windup for saturation handling
 * - Support for setpoint coordination through internal target
 * 
 * @param setpoint Desired target value (setpoint)
 * @param measurement Current process value (feedback)
 * @return Control action value (typically PWM value)
 */
float PIController::compute(float setpoint, float measurement)
{
    //-------------------------------------------------------------------------
    // Determine actual setpoint (internal or external)
    //-------------------------------------------------------------------------
    float actualSetpoint = useInternalTarget ? internalTarget : setpoint;
    
    //-------------------------------------------------------------------------
    // Calculate error and proportional term with setpoint weighting
    //-------------------------------------------------------------------------
    float e = actualSetpoint - measurement; // Standard error for integral term
    Pterm = Kp * (Beta * actualSetpoint - measurement); // Weighted proportional term
    
    //-------------------------------------------------------------------------
    // Calculate Unsaturated Control Action
    //-------------------------------------------------------------------------
    float u_unsat = Pterm + Iterm;
    
    //-------------------------------------------------------------------------
    // Anti-Windup using Back-Calculation
    //-------------------------------------------------------------------------
    const int PWM_MAX = 4095;
    const int PWM_MIN = 0;
    float u_sat;
    
    // Apply saturation limits
    if (u_unsat > PWM_MAX) {
        u_sat = PWM_MAX;
    } else if (u_unsat < PWM_MIN) {
        u_sat = PWM_MIN;
    } else {
        u_sat = u_unsat;
    }
    
    // Back-calculation: adjust integral when saturated
    float saturation_error = u_sat - u_unsat;
    
    // Update integral term using backward Euler integration and anti-windup
    Iterm += Ki * e * h + 0.1 * saturation_error; // 0.1 is the anti-windup gain
    
    // Store error for next iteration
    e_old = e;
    
    // Return saturated control action
    return u_sat;
}

//=============================================================================
// CONTROLLER MANAGEMENT FUNCTIONS
//=============================================================================

/**
 * Reset controller state
 * Clears integral term and resets error history
 * Should be called when control is re-enabled after being off or when 
 * making large changes to setpoint to prevent integral windup
 */
void PIController::reset()
{
    Iterm = 0;
    e_old = 0;
    Pterm = 0;
}

/**
 * Update controller gain parameters
 * Allows runtime modification of controller behavior
 * 
 * @param kp New proportional gain coefficient
 * @param ki New integral gain coefficient
 */
void PIController::setGains(float kp, float ki)
{
    Kp = kp;
    Ki = ki;
}

/**
 * Set setpoint weighting factor
 * Controls how setpoint changes affect proportional action
 * 
 * @param beta New setpoint weighting factor (0.0-1.0)
 */
void PIController::setWeighting(float beta)
{
    Beta = beta;
}

/**
 * Get the controller's sampling time
 * 
 * @return Sampling time in seconds
 */
float PIController::getSamplingTime() const
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
void PIController::setTarget(float newTarget)
{
    internalTarget = newTarget;
    useInternalTarget = true;
}

/**
 * Restore using the external setpoint
 * Disables the internal target and reverts to using the setpoint
 * passed directly to the compute() method
 */
void PIController::clearInternalTarget()
{
    useInternalTarget = false;
}

/**
 * Check if controller is currently using an internal target
 * 
 * @return true if using internal target, false if using external setpoint
 */
bool PIController::isUsingInternalTarget() const
{
    return useInternalTarget;
}

/**
 * Get current values of PI terms for diagnostics
 * 
 * @param p Output parameter for proportional term
 * @param i Output parameter for integral term
 */
void PIController::getTerms(float& p, float& i) const
{
    // For the most recent control calculation:
    p = Pterm;      // Proportional term
    i = Iterm;      // Integral term
}