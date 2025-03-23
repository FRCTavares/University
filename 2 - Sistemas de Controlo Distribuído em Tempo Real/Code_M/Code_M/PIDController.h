/**
 * PIDController.h - Definição do controlador PI
 *
 * Este ficheiro define a classe PIDController que implementa o algoritmo de controlo PI:
 * - Controlador PI com ponderação de setpoint
 * - Proteção anti-windup por back-calculation
 * - Reset do integrador
 * - Capacidade de coordenação através de referência interna
 * - Mecanismos para diagnóstico e monitorização
 */

#pragma once

//============================================================================
// CLASSE CONTROLADOR PI
//============================================================================

class PIDController
{
public:
    /**
     * Construtor do controlador PI
     * @param kp Ganho proporcional
     * @param ki Ganho integral
     * @param beta Fator de ponderação do setpoint (0-1)
     * @param gamma Ganho de back-calculation para anti-windup
     */
    PIDController(float kp, float ki, float beta = 1.0f, float gamma = 1.0f);
    
    /**
     * Calcula a ação de controlo
     * @param setpoint Referência desejada (setpoint)
     * @param measurement Valor medido da saída do processo
     * @return Sinal de controlo calculado
     */
    float compute(float setpoint, float measurement);
    
    /**
     * Reinicia o controlador, zerando o termo integral
     */
    void reset();
    
    /**
     * Altera os ganhos do controlador
     * @param kp Novo ganho proporcional
     * @param ki Novo ganho integral
     */
    void setGains(float kp, float ki);
    
    /**
     * Define o fator de ponderação de setpoint
     * @param beta Novo fator de ponderação (0-1)
     */
    void setSetpointWeight(float beta);
    
    /**
     * Define o ganho de anti-windup
     * @param gamma Novo ganho de back-calculation
     */
    void setAntiWindupGain(float gamma);
    
    /**
     * Obtém o tempo de amostragem
     * @return Tempo de amostragem em segundos
     */
    float getSamplingTime() const;
    
    /**
     * Define uma referência interna para coordenação
     * @param newTarget Nova referência interna
     */
    void setTarget(float newTarget);
    
    /**
     * Restaura a utilização da referência externa
     */
    void restoreExternalTarget();
    
    /**
     * Obtém os ganhos atuais do controlador
     * @param kp Referência para armazenar o ganho proporcional
     * @param ki Referência para armazenar o ganho integral
     */
    void getGains(float &kp, float &ki) const;
    
    /**
     * Obtém os termos individuais da ação de controlo para diagnóstico
     * @param p Referência para armazenar o termo proporcional
     * @param i Referência para armazenar o termo integral
     */
    void getTerms(float &p, float &i) const;

private:
    // Parâmetros do controlador
    float Kp;    // Ganho proporcional
    float Ki;    // Ganho integral
    float Beta;  // Fator de ponderação do setpoint
    float Gamma; // Ganho de back-calculation para anti-windup
    float h;     // Tempo de amostragem em segundos (fixo em 0.01)
    
    // Variáveis de estado do controlador
    float Pterm; // Termo proporcional atual
    float Iterm; // Termo integral acumulado
    float e_old; // Erro anterior
    
    // Suporte para coordenação
    float internalTarget; // Referência interna para coordenação
    bool useInternalTarget; // Flag para usar referência interna em vez da externa
};