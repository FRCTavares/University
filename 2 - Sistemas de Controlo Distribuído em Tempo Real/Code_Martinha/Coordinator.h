#pragma once
#include "StateManager.h"

// Neighbor node information
struct NeighborInfo {
    uint8_t nodeId;
    float lastLux;
    float lastDuty;
    LuminaireState state;
    unsigned long lastUpdate;
    bool isActive;
};

// Initialize coordinator
void initCoordinator();

// Update neighbor information
void updateNeighborInfo(uint8_t nodeId, uint8_t sensorType, float value);

// Get light contribution from neighbors
float getNeighborsContribution();

// Coordinate with neighbors to optimize energy
void coordinateWithNeighbors();