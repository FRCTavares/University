/**
 * PIDController.h - Definição do controlador PID
 *
 * Este ficheiro define a classe PIDController que implementa o algoritmo de controlo PID:
 * - Controlador PID com filtro derivativo
 * - Proteção anti-windup
 * - Reset do integrador
 * - Capacidade de coordenação através de referência interna
 * - Mecanismos para diagnóstico e monitorização
 */

 #pragma once

 //============================================================================
 // CLASSE CONTROLADOR PID
 //============================================================================
 
 class PIDController
 {
 public:
     /**
      * Construtor do controlador PID
      * @param kp Ganho proporcional
      * @param ki Ganho integral
      * @param kd Ganho derivativo
      * @param n Coeficiente de filtragem derivativa (N > 0)
      * @param samplingTime Tempo de amostragem em segundos
      */
     PIDController(float kp, float ki, float kd, float n, float samplingTime);
     
     /**
      * Calcula a ação de controlo
      * @param setpoint Referência desejada (setpoint)
      * @param measurement Valor medido da saída do processo
      * @return Sinal de controlo calculado
      */
     float compute(float setpoint, float measurement);
     
     /**
      * Reinicia o controlador, zerando os termos integral e derivativo
      */
     void reset();
     
     /**
      * Altera os ganhos do controlador
      * @param kp Novo ganho proporcional
      * @param ki Novo ganho integral
      * @param kd Novo ganho derivativo
      */
     void setGains(float kp, float ki, float kd);
     
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
      * @param kd Referência para armazenar o ganho derivativo
      */
     void getGains(float &kp, float &ki, float &kd) const;
     
     /**
      * Obtém os termos individuais da ação de controlo para diagnóstico
      * @param p Referência para armazenar o termo proporcional
      * @param i Referência para armazenar o termo integral
      * @param d Referência para armazenar o termo derivativo
      */
     void getTerms(float &p, float &i, float &d) const;
 
 private:
     // Parâmetros do controlador
     float Kp;    // Ganho proporcional
     float Ki;    // Ganho integral
     float Kd;    // Ganho derivativo
     float N;     // Coeficiente de filtragem derivativa
     float h;     // Tempo de amostragem em segundos
     
     // Variáveis de estado do controlador
     float Iterm; // Termo integral acumulado
     float Dterm; // Termo derivativo filtrado
     float e_old; // Erro anterior
     
     // Suporte para coordenação
     float internalTarget; // Referência interna para coordenação
     bool useInternalTarget; // Flag para usar referência interna em vez da externa
 };