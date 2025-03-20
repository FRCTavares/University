#include "StateManager.h"
#include "Configuration.h"
#include "CANCom.h"
#include "PIDController.h"

// Current state
LuminaireState luminaireState = STATE_UNOCCUPIED;

// Control system variables
extern float setpointLux;
extern bool feedbackControl;

void initStateManager()
{
    // Initialize with default state
    luminaireState = STATE_UNOCCUPIED;
    setpointLux = SETPOINT_UNOCCUPIED;
    feedbackControl = true;
}

// Função para mudar o estado e atualizar os setpoints de acordo
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

    // Atualizar iluminância de referência para cálculo de métricas
    refIlluminance = setpointLux;

    // Transmitir mudança de estado para a rede
    sendControlCommand(CAN_ADDR_BROADCAST, CAN_CTRL_STATE_CHANGE, (float)luminaireState);
}

LuminaireState getCurrentState()
{
    return luminaireState;
}

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