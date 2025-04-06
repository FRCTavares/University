#ifndef PICONTROLLER_H
#define PICONTROLLER_H

#include <Arduino.h>
#include <math.h>

/**
 * PI Controller Class
 *
 * Implements a discrete-time PI controller with:
 * - Setpoint weighting for improved disturbance rejection
 * - Back-calculation anti-windup to prevent integral saturation
 * - Support for setpoint coordination between nodes
 * - Configurable gains and sampling time
 *
 * This controller is designed for real-time control applications where
 * robustness and predictable performance are required.
 */
class PIController
{
public:
  //=========================================================================
  // CONSTRUCTOR AND CONFIGURATION
  //=========================================================================

  /**
   * PI Controller Constructor
   * Initializes a PI controller with anti-windup capabilities
   *
   * @param kp Proportional gain coefficient
   * @param ki Integral gain coefficient
   * @param beta Setpoint weighting factor (0.0-1.0)
   * @param samplingTime Control loop sampling time in seconds
   */
  PIController(float kp, float ki, float beta, float samplingTime);

  /**
   * Update controller gain parameters
   * Allows runtime modification of controller behavior
   *
   * @param kp New proportional gain coefficient
   * @param ki New integral gain coefficient
   */
  void setGains(float kp, float ki);

  /**
   * Set setpoint weighting factor
   * Controls how setpoint changes affect proportional action
   *
   * @param beta New setpoint weighting factor (0.0-1.0)
   */
  void setWeighting(float beta);

  /**
   * Get the controller's sampling time
   *
   * @return Sampling time in seconds
   */
  float getSamplingTime() const;

  //=========================================================================
  // CONTROL COMPUTATION
  //=========================================================================

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
  float compute(float setpoint, float measurement);

  /**
   * Reset controller state
   * Clears integral term and resets error history
   * Should be called when control is re-enabled after being off or when
   * making large changes to setpoint to prevent integral windup
   */
  void reset();

  //=========================================================================
  // COORDINATION SUPPORT
  //=========================================================================

  /**
   * Set an internal coordination target
   * This allows coordination algorithms to temporarily adjust the target
   * without changing the user-defined setpoint
   *
   * @param newTarget New internally managed setpoint value
   */
  void setTarget(float newTarget);

  /**
   * Restore using the external setpoint
   * Disables the internal target and reverts to using the setpoint
   * passed directly to the compute() method
   */
  void clearInternalTarget();

  /**
   * Check if controller is currently using an internal target
   *
   * @return true if using internal target, false if using external setpoint
   */
  bool isUsingInternalTarget() const;

  //=========================================================================
  // DIAGNOSTIC FUNCTIONS
  //=========================================================================

  /**
   * Get current values of PI terms for diagnostics
   *
   * @param p Output parameter for proportional term
   * @param i Output parameter for integral term
   */
  void getTerms(float &p, float &i) const;

  /**
   * Enable or disable feedforward control
   *
   * @param enable True to enable feedforward, false to disable
   * @param ffGain Feedforward gain to be used when enabled
   */
  void enableFeedforward(bool enable, float ffGain);

private:
  //=========================================================================
  // CONTROLLER PARAMETERS
  //=========================================================================

  float Kp;   // Proportional gain
  float Ki;   // Integral gain
  float Beta; // Setpoint weighting factor
  float h;    // Sampling time in seconds

  //=========================================================================
  // CONTROLLER STATE
  //=========================================================================

  float Iterm; // Accumulated integral term
  float e_old; // Previous error for integration
  float Pterm; // Last proportional term (for diagnostics)

  //=========================================================================
  // COORDINATION VARIABLES
  //=========================================================================

  float internalTarget;   // Target value used for coordination
  bool useInternalTarget; // Flag to use internal or external target

  // Add these new variables
  float Kff;           // Feedforward gain
  float prevSetpoint;  // Previous setpoint for detecting changes
  bool useFeedforward; // Enable/disable feedforward
};

#endif // PICONTROLLER_H