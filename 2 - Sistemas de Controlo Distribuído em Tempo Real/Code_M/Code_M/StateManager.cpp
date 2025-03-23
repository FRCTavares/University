/**
 * StateManager.cpp - Implementação do gestor de estados do LED
 *
 * Este ficheiro implementa a máquina de estados que controla o comportamento do LED:
 * - Gestão dos estados de operação (desligado, não-ocupado, ocupado)
 * - Transições entre estados com atualização automática dos setpoints
 * - Reinicialização segura do controlador durante transições
 * - Notificação das mudanças de estado para outros nós na rede CAN
 * - Recuperação do estado após reinicialização
 */

//============================================================================
// FICHEIROS INCLUÍDOS
//============================================================================

#include "StateManager.h"
#include "Configuration.h"
#include "CANCom.h"
#include "PIDController.h"

//============================================================================
// VARIÁVEIS GLOBAIS
//============================================================================

// Estado atual do LED
LuminaireState luminaireState = STATE_UNOCCUPIED;

// Variáveis externas do sistema de controlo
extern float setpointLux;      // Setpoint de iluminação
extern bool feedbackControl;   // Ativa/desativa controlo em malha fechada

//============================================================================
// FUNÇÕES DE INICIALIZAÇÃO
//============================================================================

/**
 * Inicializa o gestor de estados do LED
 * Define o estado inicial e as configurações de controlo correspondentes
 */
void initStateManager()
{
    // Inicializar com o estado predefinido
    luminaireState = STATE_UNOCCUPIED;
    setpointLux = SETPOINT_UNOCCUPIED;
    feedbackControl = true;
}

//============================================================================
// FUNÇÕES DE CONTROLO DE ESTADO
//============================================================================

/**
 * Muda o estado do LED e atualiza os setpoints e modos de controlo
 * Também notifica outros nós na rede sobre a mudança de estado
 * 
 * @param newState Novo estado para o qual a LED deve transitar
 */
void changeState(LuminaireState newState)
{
    // Não fazer nada se o estado não mudar
    if (newState == luminaireState)
    {
        return;
    }

    luminaireState = newState;

    // Atualizar setpoint com base no novo estado
    switch (luminaireState)
    {
    case STATE_OFF:
        setpointLux = SETPOINT_OFF;
        feedbackControl = false; // Desligar controlo quando as luzes estão desligadas
        break;

    case STATE_UNOCCUPIED:
        setpointLux = SETPOINT_UNOCCUPIED;
        feedbackControl = true;
        break;

    case STATE_OCCUPIED:
        setpointLux = SETPOINT_OCCUPIED;
        feedbackControl = true;
        break;
    }

    // Reiniciar o controlador PID para evitar windup integral durante transições
    pid.reset();

    // Atualizar iluminação de referência para cálculo de métricas
    refIlluminance = setpointLux;

    // Transmitir mudança de estado para a rede
    sendControlCommand(CAN_ADDR_BROADCAST, CAN_CTRL_STATE_CHANGE, (float)luminaireState);
}

/**
 * Obtém o estado atual do LED
 * 
 * @return Estado atual (OFF, UNOCCUPIED ou OCCUPIED)
 */
LuminaireState getCurrentState()
{
    return luminaireState;
}

/**
 * Obtém o setpoint de iluminação correspondente a um determinado estado
 * 
 * @param state Estado para o qual se pretende saber o setpoint
 * @return Valor do setpoint de iluminação em lux
 */
float getStateSetpoint(LuminaireState state)
{
    switch (state)
    {
    case STATE_OFF:
        return SETPOINT_OFF;
    case STATE_UNOCCUPIED:
        return SETPOINT_UNOCCUPIED;
    case STATE_OCCUPIED:
        return SETPOINT_OCCUPIED;
    default:
        return SETPOINT_UNOCCUPIED;
    }
}
