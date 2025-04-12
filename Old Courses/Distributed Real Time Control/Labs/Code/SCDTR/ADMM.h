#ifndef ADMM_H
#define ADMM_H

#include <Arduino.h>

#define MAX_NODES 3 // Now supports 3 nodes

// Node structure for ADMM algorithm
struct Node
{
    int index;               // Index of this node in the control vector
    float u[MAX_NODES];      // Control variables (u)
    float u_av[MAX_NODES];   // Average control variables (u_av)
    float lambda[MAX_NODES]; // Dual variables (lambda)
    float c[MAX_NODES];      // Cost coefficients
    float k[MAX_NODES];      // Gain values
    float L;                 // Illuminance lower bound
    float d;                 // External illuminance (disturbance)
    float n;                 // Precomputed value (sum of squares of all gains)
    float m;                 // Precomputed value (sum of squares excluding self gain)
};

// Make the global node variable accessible to other files
extern Node localNode;

// Function declarations
void initADMMNode(Node &node, int nodeIndex);
void processADMMMessage(uint8_t sourceNodeId, uint8_t msgType, float value);
bool updateADMMConsensus();
void dumpADMMState();
void updateExternalIlluminanceModel();
void updateExternalIlluminanceForControl();
float controlToOutputDuty(float u);
float outputToControlDuty(float duty);

#endif // ADMM_H