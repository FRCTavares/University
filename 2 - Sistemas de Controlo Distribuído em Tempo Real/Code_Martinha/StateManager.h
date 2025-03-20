/**
 * StateManager.h - Gestor de estados do LED
 *
 * Este ficheiro define a interface para gestão de estados operacionais do LED:
 * - Definição dos possíveis estados de funcionamento (desligado, não-ocupado, ocupado)
 * - Funções para transição entre estados
 * - Controlo dos setpoints de iluminação conforme o estado
 * - Interfaces para monitorização do estado atual
 * - Acesso aos parâmetros associados a cada estado
 */

 #pragma once
 #include "PIDController.h"
 
 //============================================================================
 // DEFINIÇÕES DE ESTADOS
 //============================================================================
 
 /**
  * Enumeração dos possíveis estados do LED
  */
 enum LuminaireState
 {
     STATE_OFF = 0,        // Desligado (sem ocupação no escritório)
     STATE_UNOCCUPIED = 1, // Não-ocupado (sem utilizador na secretária, iluminação reduzida)
     STATE_OCCUPIED = 2    // Ocupado (secretária em uso, iluminação completa)
 };
 
 //============================================================================
 // FUNÇÕES DE GESTÃO DE ESTADOS
 //============================================================================
 
 /**
  * Inicializa o gestor de estados do LED
  * Define o estado inicial e as configurações de controlo correspondentes
  */
 void initStateManager();
 
 /**
  * Muda o estado atual do LED
  * Atualiza os parâmetros de controlo conforme o novo estado
  * 
  * @param newState Novo estado para o qual o LED deve transitar
  */
 void changeState(LuminaireState newState);
 
 /**
  * Obtém o estado atual do LED
  * 
  * @return Estado atual do LED
  */
 LuminaireState getCurrentState();
 
 /**
  * Obtém o setpoint de iluminação correspondente a um determinado estado
  * 
  * @param state Estado para o qual se pretende obter o setpoint
  * @return Valor do setpoint de iluminação em lux
  */
 float getStateSetpoint(LuminaireState state);
 
 //============================================================================
 // VARIÁVEIS EXTERNAS
 //============================================================================
 
 /**
  * Controlador PID utilizado para controlo por realimentação
  * Declarado externamente para permitir acesso por outros módulos
  */
 extern PIDController pid;