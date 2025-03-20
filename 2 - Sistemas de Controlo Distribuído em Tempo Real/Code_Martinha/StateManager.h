#pragma once
#include "PIDController.h"

// Define luminaire states
enum LuminaireState
{
    STATE_OFF = 0,        // No one in office
    STATE_UNOCCUPIED = 1, // No one at desk, low light
    STATE_OCCUPIED = 2    // Desk is busy, full light
};

// State management interface
void initStateManager();
void changeState(LuminaireState newState);
LuminaireState getCurrentState();
float getStateSetpoint(LuminaireState state);

extern PIDController pid; // PID controller for feedback control