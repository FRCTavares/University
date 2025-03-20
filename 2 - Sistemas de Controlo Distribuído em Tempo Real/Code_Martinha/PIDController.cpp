/**
 * PIDController.cpp - Implementação do controlador PID
 *
 * Este ficheiro implementa um controlador PID com funcionalidades avançadas:
 * - Controlo Proporcional, Integral e Derivativo
 * - Proteção anti-windup para evitar saturação do integrador
 * - Filtragem do termo derivativo para reduzir sensibilidade ao ruído
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
 * @param kd Ganho derivativo
 * @param n Fator de filtragem do termo derivativo
 * @param samplingTime Tempo de amostragem em segundos
 */
PIDController::PIDController(float kp, float ki, float kd, float n, float samplingTime)
    : Kp(kp), Ki(ki), Kd(kd), N(n), h(samplingTime), Iterm(0), Dterm(0), e_old(0),
      internalTarget(0), useInternalTarget(false) {}

/**
 * Calcula a ação de controlo do PID
 * @param setpoint Valor desejado da variável de processo
 * @param measurement Valor atual medido da variável de processo
 * @return Ação de controlo calculada
 */
float PIDController::compute(float setpoint, float measurement)
{
    // Usar alvo interno se definido pela lógica de coordenação
    float actualSetpoint = useInternalTarget ? internalTarget : setpoint;

    float e = actualSetpoint - measurement;

    // Saída de debug
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

/**
 * Reinicia os estados internos do controlador
 * Útil quando o controlador é ligado/desligado ou ocorre uma mudança grande no setpoint
 */
void PIDController::reset()
{
    Iterm = 0;
    Dterm = 0;
    e_old = 0;
}

/**
 * Define novos ganhos para o controlador
 * @param kp Novo ganho proporcional
 * @param ki Novo ganho integral
 * @param kd Novo ganho derivativo
 */
void PIDController::setGains(float kp, float ki, float kd)
{
    Kp = kp;
    Ki = ki;
    Kd = kd;
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
 * @param kd Referência para armazenar o ganho derivativo
 */
void PIDController::getGains(float &kp, float &ki, float &kd) const
{
    kp = Kp;
    ki = Ki;
    kd = Kd;
}

/**
 * Obtém os termos atuais de controlo para análise
 * @param p Referência para armazenar o termo proporcional
 * @param i Referência para armazenar o termo integral
 * @param d Referência para armazenar o termo derivativo
 */
void PIDController::getTerms(float &p, float &i, float &d) const
{
    p = Kp * e_old; // Proporcional ao último erro
    i = Iterm;      // Termo integral acumulado
    d = Kd * Dterm; // Termo derivativo filtrado
}