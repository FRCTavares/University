#include "PIDController.h"
#include "Configuration.h"
#include <Arduino.h>

PIDController::PIDController(float kp, float ki, float kd, float n, float samplingTime)
    : Kp(kp), Ki(ki), Kd(kd), N(n), h(samplingTime), Iterm(0), Dterm(0), e_old(0),
      internalTarget(0), useInternalTarget(false) {}

float PIDController::compute(float setpoint, float measurement)
{
    // Usar alvo interno se definido pela lógica de coordenação
    float actualSetpoint = useInternalTarget ? internalTarget : setpoint;

    float e = actualSetpoint - measurement;

    // Saída de depuração
    if (DEBUG_MODE && DEBUG_PID)
    {
        Serial.print("PID: SP=");
        Serial.print(actualSetpoint);
        Serial.print(" PV=");
        Serial.print(measurement);
        Serial.print(" e=");
        Serial.println(e);
    }

    // Termo proporcional
    float Pterm = Kp * e;

    // Termo derivativo com filtragem
    float derivative = (e - e_old) / h;
    float alpha = N * h;
    Dterm = (alpha * derivative + Dterm) / (1 + alpha);
    float D_out = Kd * Dterm;

    // Calcular ação de controlo não saturada
    float u_unsat = Pterm + Iterm + D_out;

    // Anti-windup: Apenas integrar se o controlo não estiver saturado
    if ((u_unsat < PWM_MAX || e < 0) && (u_unsat > PWM_MIN || e > 0))
    {
        Iterm += Ki * e * h;
    }

    e_old = e;
    return Pterm + Iterm + D_out;
}

void PIDController::reset()
{
    Iterm = 0;
    Dterm = 0;
    e_old = 0;
}

void PIDController::setGains(float kp, float ki, float kd)
{
    Kp = kp;
    Ki = ki;
    Kd = kd;
}

float PIDController::getSamplingTime() const
{
    return h;
}

void PIDController::setTarget(float newTarget)
{
    // Isto permite que algoritmos de coordenação ajustem temporariamente o alvo
    // sem alterar o setpoint definido pelo utilizador
    internalTarget = newTarget;
    useInternalTarget = true;
}
