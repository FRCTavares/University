/**
 * PIDController.cpp - Implementação do controlador PI
 *
 * Este ficheiro implementa um controlador PI com funcionalidades avançadas:
 * - Controlo Proporcional e Integral
 * - Ponderação do setpoint no termo proporcional
 * - Proteção anti-windup por back-calculation
 * - Capacidade de ajuste dinâmico dos ganhos
 * - Suporte para alteração do setpoint interno para algoritmos de coordenação
 */

//============================================================================
// FICHEIROS INCLUÍDOS
//============================================================================

#include "PIDController.h"
#include "Configuration.h"
#include <Arduino.h>

//============================================================================
// IMPLEMENTAÇÃO DA CLASSE PIDCONTROLLER
//============================================================================

/**
 * Construtor da classe PIDController
 * @param kp Ganho proporcional
 * @param ki Ganho integral
 * @param beta Fator de ponderação do setpoint (0-1)
 * @param gamma Ganho de back-calculation para anti-windup
 */
PIDController::PIDController(float kp, float ki, float beta, float gamma)
    : Kp(kp), Ki(ki), Beta(beta), Gamma(gamma), h(0.01f), Pterm(0), Iterm(0), e_old(0),
      internalTarget(0), useInternalTarget(false) {}

/**
 * Calcula a ação de controlo do PI
 * @param setpoint Valor desejado da variável de processo
 * @param measurement Valor atual medido da variável de processo
 * @return Ação de controlo calculada
 */
float PIDController::compute(float setpoint, float measurement)
{
    // Usar alvo interno se definido pela lógica de coordenação
    float actualSetpoint = useInternalTarget ? internalTarget : setpoint;

    // Termo proporcional com ponderação de setpoint
    Pterm = Kp * (Beta * actualSetpoint - measurement);
    
    // Cálculo do erro para o termo integral
    float e = actualSetpoint - measurement;

    // Saída de debug
    if (DEBUG_MODE && DEBUG_PID)
    {
        Serial.print("PI: SP=");
        Serial.print(actualSetpoint);
        Serial.print(" PV=");
        Serial.print(measurement);
        Serial.print(" e=");
        Serial.println(e);
    }

    // Calcular ação de controlo não saturada
    float u = Pterm + Iterm;
    
    // Aplicar saturação
    float u_sat = u;
    if (u > PWM_MAX) u_sat = PWM_MAX;
    if (u < PWM_MIN) u_sat = PWM_MIN;
    
    // Anti-windup por back-calculation
    float windup_error = u_sat - u;
    
    // Atualizar termo integral usando método de Euler e back-calculation
    Iterm += Ki * e * h + Gamma * windup_error * h;
    
    e_old = e;
    return u_sat;
}

/**
 * Reinicia os estados internos do controlador
 * Útil quando o controlador é ligado/desligado ou ocorre uma mudança grande no setpoint
 */
void PIDController::reset()
{
    Pterm = 0;
    Iterm = 0;
    e_old = 0;
}

/**
 * Define novos ganhos para o controlador
 * @param kp Novo ganho proporcional
 * @param ki Novo ganho integral
 */
void PIDController::setGains(float kp, float ki)
{
    Kp = kp;
    Ki = ki;
}

/**
 * Define o fator de ponderação de setpoint
 * @param beta Novo fator de ponderação (0-1)
 */
void PIDController::setSetpointWeight(float beta)
{
    Beta = beta;
}

/**
 * Define o ganho de anti-windup
 * @param gamma Novo ganho de back-calculation
 */
void PIDController::setAntiWindupGain(float gamma)
{
    Gamma = gamma;
}

/**
 * Obtém o tempo de amostragem do controlador
 * @return Tempo de amostragem em segundos
 */
float PIDController::getSamplingTime() const
{
    return h;
}

/**
 * Define um alvo interno para o controlador
 * Útil para algoritmos de coordenação que precisam de ajustar temporariamente o setpoint
 * @param newTarget Novo alvo interno
 */
void PIDController::setTarget(float newTarget)
{
    // Isto permite que algoritmos de coordenação ajustem temporariamente o alvo
    // sem alterar o setpoint definido pelo utilizador
    internalTarget = newTarget;
    useInternalTarget = true;
}

/**
 * Restaura o uso do setpoint externo normal
 * Útil após concluir um ajuste temporário do setpoint
 */
void PIDController::restoreExternalTarget()
{
    useInternalTarget = false;
}

/**
 * Obtém os ganhos atuais do controlador
 * @param kp Referência para armazenar o ganho proporcional
 * @param ki Referência para armazenar o ganho integral
 */
void PIDController::getGains(float &kp, float &ki) const
{
    kp = Kp;
    ki = Ki;
}

/**
 * Obtém os termos atuais de controlo para análise
 * @param p Referência para armazenar o termo proporcional
 * @param i Referência para armazenar o termo integral
 */
void PIDController::getTerms(float &p, float &i) const
{
    p = Pterm;  // Termo proporcional atual
    i = Iterm;  // Termo integral acumulado
}