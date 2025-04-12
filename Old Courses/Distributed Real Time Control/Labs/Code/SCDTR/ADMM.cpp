#include <Arduino.h>
#include <math.h>
#include "pico/multicore.h"

#include "Globals.h"
#include "CANComm.h"
#include "ADMM.h"
#include "LEDDriver.h"
#include "SensorManager.h"

// Maximum number of neighboring nodes to track
#define MAX_ADMM_NEIGHBORS 3

// ADMM message types
#define ADMM_MSG_U_VAL 21      // u value
#define ADMM_MSG_U_AV_VAL 22   // u_av value
#define ADMM_MSG_LAMBDA_VAL 23 // lambda value

// Global node structure for this device
Node localNode;

// Track values from other nodes
struct ADMMNeighborData
{
    uint8_t nodeId;
    bool active;
    float u[MAX_NODES];      // Control variables
    float u_av[MAX_NODES];   // Average control variables
    float lambda[MAX_NODES]; // Dual variables
    unsigned long lastUpdate;
};

ADMMNeighborData admmNeighbors[MAX_ADMM_NEIGHBORS];

void updateExternalIlluminanceForControl()
{
    static unsigned long lastUpdateTime = 0;
    unsigned long currentTime = millis();

    // Update every 500ms even when not in ADMM iteration
    if (currentTime - lastUpdateTime >= 500)
    {
        updateExternalIlluminanceModel();
        lastUpdateTime = currentTime;
    }
}

void updateExternalIlluminanceModel()
{
    // Get current measured illuminance
    float measuredLux = readLux();

    // Calculate expected illuminance from all LEDs
    float expectedLEDContribution = 0.0f;
    for (int i = 0; i < MAX_NODES; i++)
    {
        expectedLEDContribution += localNode.k[i] * controlToOutputDuty(localNode.u[i]);
    }

    // Estimate external illuminance (measured - LED contribution)
    float newEstimate = measuredLux - expectedLEDContribution;
    if (newEstimate < 0)
        newEstimate = 0; // Cannot have negative illuminance

    // Apply low-pass filter to avoid oscillations from measurement noise
    static const float FILTER_ALPHA = 0.2f; // Smoothing factor
    localNode.d = (1.0f - FILTER_ALPHA) * localNode.d + FILTER_ALPHA * newEstimate;

    // Also update the target illuminance (might have changed)
    critical_section_enter_blocking(&commStateLock);
    localNode.L = controlState.setpointLux;
    critical_section_exit(&commStateLock);
}

// Convert between control variable (0-100) and output duty cycle (0-1)
float controlToOutputDuty(float u)
{
    return u / 100.0f;
}

float outputToControlDuty(float duty)
{
    return duty * 100.0f;
}

bool check_feasibility(Node &node, float u[MAX_NODES])
{
    float tol = 0.001;
    if (u[node.index] < -tol || u[node.index] > 100 + tol)
        return false;

    float dot = 0.0f;
    for (int i = 0; i < MAX_NODES; i++)
    {
        dot += u[i] * node.k[i];
    }

    if (dot < node.L - node.d - tol)
        return false;

    return true;
}

float evaluate_cost(Node &node, float u[MAX_NODES], float rho)
{
    float dot_cu = 0.0f;
    float dot_lambdau = 0.0f;
    float norm = 0.0f;

    for (int i = 0; i < MAX_NODES; i++)
    {
        // Cost term
        dot_cu += node.c[i] * u[i];

        // Lagrangian term and norm
        float diff = u[i] - node.u_av[i];
        dot_lambdau += node.lambda[i] * diff;
        norm += diff * diff;
    }

    return dot_cu + dot_lambdau + rho / 2.0 * norm;
}

void consensus_iterate(Node &node, float rho, float u_out[MAX_NODES], float &cost_out)
{
    float u_best[MAX_NODES];
    float cost_best = 1e6;

    // Initialize best solution to something invalid
    for (int i = 0; i < MAX_NODES; i++)
    {
        u_best[i] = -1;
    }

    // Calculate z = rho*u_av - lambda - c
    float z[MAX_NODES];
    for (int i = 0; i < MAX_NODES; i++)
    {
        z[i] = rho * node.u_av[i] - node.lambda[i] - node.c[i];
    }

    // Case 1: Unconstrained solution u = z/rho
    float u_u[MAX_NODES];
    for (int i = 0; i < MAX_NODES; i++)
    {
        u_u[i] = z[i] / rho;
    }

    if (check_feasibility(node, u_u))
    {
        float cost = evaluate_cost(node, u_u, rho);
        if (cost < cost_best)
        {
            for (int i = 0; i < MAX_NODES; i++)
            {
                u_best[i] = u_u[i];
            }
            cost_best = cost;
        }
    }

    // Case 2: Constrained to linear boundary (illuminance constraint)
    // Calculate dot product of k and z, and scale factor
    float k_dot_z = 0.0f;
    for (int i = 0; i < MAX_NODES; i++)
    {
        k_dot_z += node.k[i] * z[i];
    }

    float scalar = (node.d - node.L + (1.0f / rho) * k_dot_z) / node.n;

    float u_bl[MAX_NODES];
    for (int i = 0; i < MAX_NODES; i++)
    {
        u_bl[i] = u_u[i] - scalar * node.k[i];
    }

    if (check_feasibility(node, u_bl))
    {
        float cost = evaluate_cost(node, u_bl, rho);
        if (cost < cost_best)
        {
            for (int i = 0; i < MAX_NODES; i++)
            {
                u_best[i] = u_bl[i];
            }
            cost_best = cost;
        }
    }

    // Case 3: Constrained to 0 boundary for this node
    float u_b0[MAX_NODES];
    for (int i = 0; i < MAX_NODES; i++)
    {
        u_b0[i] = u_u[i];
    }
    u_b0[node.index] = 0;

    if (check_feasibility(node, u_b0))
    {
        float cost = evaluate_cost(node, u_b0, rho);
        if (cost < cost_best)
        {
            for (int i = 0; i < MAX_NODES; i++)
            {
                u_best[i] = u_b0[i];
            }
            cost_best = cost;
        }
    }

    // Case 4: Constrained to 100 boundary for this node
    float u_b1[MAX_NODES];
    for (int i = 0; i < MAX_NODES; i++)
    {
        u_b1[i] = u_u[i];
    }
    u_b1[node.index] = 100;

    if (check_feasibility(node, u_b1))
    {
        float cost = evaluate_cost(node, u_b1, rho);
        if (cost < cost_best)
        {
            for (int i = 0; i < MAX_NODES; i++)
            {
                u_best[i] = u_b1[i];
            }
            cost_best = cost;
        }
    }

    // Case 5: Linear boundary + 0 boundary
    float z_dot_k = 0.0f;
    for (int i = 0; i < MAX_NODES; i++)
    {
        z_dot_k += z[i] * node.k[i];
    }

    float u_l0[MAX_NODES];
    for (int i = 0; i < MAX_NODES; i++)
    {
        float term1 = node.k[i] / node.m * (node.d - node.L);
        float term2 = node.k[i] / (rho * node.m) * (node.k[node.index] * z[node.index] - z_dot_k);
        u_l0[i] = u_u[i] - term1 + term2;
    }
    u_l0[node.index] = 0;

    if (check_feasibility(node, u_l0))
    {
        float cost = evaluate_cost(node, u_l0, rho);
        if (cost < cost_best)
        {
            for (int i = 0; i < MAX_NODES; i++)
            {
                u_best[i] = u_l0[i];
            }
            cost_best = cost;
        }
    }

    // Case 6: Linear boundary + 100 boundary
    float u_l1[MAX_NODES];
    for (int i = 0; i < MAX_NODES; i++)
    {
        float term1 = node.k[i] / node.m * (node.d - node.L + 100 * node.k[node.index]);
        float term2 = node.k[i] / (rho * node.m) * (node.k[node.index] * z[node.index] - z_dot_k);
        u_l1[i] = u_u[i] - term1 + term2;
    }
    u_l1[node.index] = 100;

    if (check_feasibility(node, u_l1))
    {
        float cost = evaluate_cost(node, u_l1, rho);
        if (cost < cost_best)
        {
            for (int i = 0; i < MAX_NODES; i++)
            {
                u_best[i] = u_l1[i];
            }
            cost_best = cost;
        }
    }

    // Output the best solution
    for (int i = 0; i < MAX_NODES; i++)
    {
        u_out[i] = u_best[i];
    }
    cost_out = cost_best;
}

bool hasConverged()
{
    // Get current values from control state
    critical_section_enter_blocking(&commStateLock);
    float setpoint = controlState.setpointLux;
    critical_section_exit(&commStateLock);

    // Base tolerance on setpoint and problem scale
    float primalTolerance = max(0.5f, setpoint * 0.01f);
    float dualTolerance = max(0.5f, setpoint * 0.01f);

    // Calculate residuals as before
    float primalResidual = 0.0f;
    for (int i = 0; i < MAX_NODES; i++)
    {
        primalResidual += pow(localNode.u[i] - localNode.u_av[i], 2);
    }
    primalResidual = sqrt(primalResidual);

    static float prevLambda[MAX_NODES] = {0};
    float dualResidual = 0.0f;
    for (int i = 0; i < MAX_NODES; i++)
    {
        dualResidual += pow(localNode.lambda[i] - prevLambda[i], 2);
        prevLambda[i] = localNode.lambda[i];
    }
    dualResidual = sqrt(dualResidual);

    // Require multiple consecutive convergence checks
    static int convergedCount = 0;
    if (primalResidual < primalTolerance && dualResidual < dualTolerance)
    {
        convergedCount++;
    }
    else
    {
        convergedCount = 0;
    }

    // Only declare convergence after 3 consecutive checks
    return (convergedCount >= 3);
}

void updateRhoParameter()
{
    static float prevPrimalResidual = 0.0f;

    // Calculate primal and dual residuals
    float primalResidual = 0.0f;
    float dualResidual = 0.0f;

    for (int i = 0; i < MAX_NODES; i++)
    {
        primalResidual += pow(localNode.u[i] - localNode.u_av[i], 2);

        // For dual residual, use change in u_av (simplified approach)
        static float prev_u_av[MAX_NODES] = {0};
        dualResidual += pow(localNode.u_av[i] - prev_u_av[i], 2);
        prev_u_av[i] = localNode.u_av[i];
    }

    primalResidual = sqrt(primalResidual);
    dualResidual = sqrt(dualResidual);

    // Adjust rho based on residuals
    critical_section_enter_blocking(&commStateLock);
    float currentRho = controlState.rho;

    // Only adjust rho if we've made some progress (avoid oscillations)
    if (primalResidual < prevPrimalResidual * 0.95 ||
        primalResidual > prevPrimalResidual * 1.05)
    {

        const float mu = 10.0f; // Balance factor

        if (primalResidual > mu * dualResidual)
        {
            // Primal residual too large -> increase rho
            controlState.rho *= 1.2f;
        }
        else if (dualResidual > mu * primalResidual)
        {
            // Dual residual too large -> decrease rho
            controlState.rho /= 1.2f;
        }

        // Keep rho within reasonable bounds
        controlState.rho = constrain(controlState.rho, 0.01f, 10.0f);
    }
    critical_section_exit(&commStateLock);

    prevPrimalResidual = primalResidual;
}

// Initialize a node with calibration matrix data
void initADMMNode(Node &node, int nodeIndex)
{
    critical_section_enter_blocking(&commStateLock);

    // Find this node's index in the calibration matrix
    int nodeIdx = -1;
    for (int i = 0; i < commState.calibMatrix.numNodes; i++)
    {
        if (commState.calibMatrix.nodeIds[i] == deviceConfig.nodeId)
        {
            nodeIdx = i;
            break;
        }
    }

    // Use the found index or the passed index if not found
    node.index = (nodeIdx >= 0) ? nodeIdx : nodeIndex;

    // Initialize control variables with current duty cycle
    float currentDuty = controlState.dutyCycle;
    node.u[node.index] = outputToControlDuty(currentDuty);

    // Initialize other nodes' control values to 0
    for (int i = 0; i < MAX_NODES; i++)
    {
        if (i != node.index)
        {
            node.u[i] = 0.0f;
        }
    }

    // Initialize averaging and dual variables for all nodes
    for (int i = 0; i < MAX_NODES; i++)
    {
        node.u_av[i] = node.u[i];
        node.lambda[i] = 0.0f;
        // Set cost coefficients - only this node's coefficient is non-zero
        node.c[i] = (i == node.index) ? controlState.cost : 0.0f;
    }

    // Get calibration gains
    if (commState.calibMatrix.numNodes >= 2)
    {
        // Set k values from calibration matrix for all nodes
        for (int i = 0; i < MAX_NODES; i++)
        {
            if (i < commState.calibMatrix.numNodes)
            {
                node.k[i] = commState.calibMatrix.gains[node.index][i];
            }
            else
            {
                node.k[i] = 0.0f; // Zero gains for non-existent nodes
            }
        }

        // Compute n and m values (sum of squares)
        node.n = 0.0f;
        for (int i = 0; i < MAX_NODES; i++)
        {
            node.n += node.k[i] * node.k[i];
        }
        node.m = node.n - node.k[node.index] * node.k[node.index];

        // Set illumination target and disturbance
        node.L = controlState.setpointLux;
        node.d = sensorState.baselineIlluminance;
    }
    else
    {
        // Default values if calibration hasn't been done
        for (int i = 0; i < MAX_NODES; i++)
        {
            node.k[i] = (i == node.index) ? 20.0f : 5.0f;
        }
        node.n = 0.0f;
        for (int i = 0; i < MAX_NODES; i++)
        {
            node.n += node.k[i] * node.k[i];
        }
        node.m = node.n - node.k[node.index] * node.k[node.index];
        node.L = 30.0f;
        node.d = 5.0f;
    }

    critical_section_exit(&commStateLock);

    // Initialize neighbor tracking
    for (int i = 0; i < MAX_ADMM_NEIGHBORS; i++)
    {
        admmNeighbors[i].active = false;
    }
}

// Process incoming ADMM messages from other nodes
void processADMMMessage(uint8_t sourceNodeId, uint8_t msgType, float value)
{
    // First find node index in our structure
    int nodeIndex = -1;
    for (int i = 0; i < MAX_ADMM_NEIGHBORS; i++)
    {
        if (admmNeighbors[i].active && admmNeighbors[i].nodeId == sourceNodeId)
        {
            nodeIndex = i;
            break;
        }
    }

    // If not found, try to add this node
    if (nodeIndex < 0)
    {
        for (int i = 0; i < MAX_ADMM_NEIGHBORS; i++)
        {
            if (!admmNeighbors[i].active)
            {
                admmNeighbors[i].active = true;
                admmNeighbors[i].nodeId = sourceNodeId;
                nodeIndex = i;
                break;
            }
        }
    }

    // If we couldn't find or create space for this node, ignore message
    if (nodeIndex < 0)
    {
        return;
    }

    // Update node's last activity time
    admmNeighbors[nodeIndex].lastUpdate = millis();

    // Process message based on type
    if (msgType == ADMM_MSG_U_VAL)
    {
        // Process control variable
        int controlIndex = -1;
        critical_section_enter_blocking(&commStateLock);
        for (int i = 0; i < commState.calibMatrix.numNodes; i++)
        {
            if (commState.calibMatrix.nodeIds[i] == sourceNodeId)
            {
                controlIndex = i;
                break;
            }
        }
        critical_section_exit(&commStateLock);

        // If we couldn't map the source node, return
        if (controlIndex < 0 || controlIndex >= MAX_NODES)
        {
            return;
        }

        // Store their u value
        admmNeighbors[nodeIndex].u[controlIndex] = value;
    }
    else if (msgType == ADMM_MSG_U_AV_VAL)
    {
        // Map source node ID to control vector index
        int controlIndex = -1;
        critical_section_enter_blocking(&commStateLock);
        for (int i = 0; i < commState.calibMatrix.numNodes; i++)
        {
            if (commState.calibMatrix.nodeIds[i] == sourceNodeId)
            {
                controlIndex = i;
                break;
            }
        }
        critical_section_exit(&commStateLock);

        // If we couldn't map the source node, return
        if (controlIndex < 0 || controlIndex >= MAX_NODES)
        {
            return;
        }

        // Store their average value
        admmNeighbors[nodeIndex].u_av[controlIndex] = value;
    }
    else if (msgType == ADMM_MSG_LAMBDA_VAL)
    {
        // Similar mapping for lambda values
        int controlIndex = -1;
        critical_section_enter_blocking(&commStateLock);
        for (int i = 0; i < commState.calibMatrix.numNodes; i++)
        {
            if (commState.calibMatrix.nodeIds[i] == sourceNodeId)
            {
                controlIndex = i;
                break;
            }
        }
        critical_section_exit(&commStateLock);

        if (controlIndex < 0 || controlIndex >= MAX_NODES)
        {
            return;
        }

        // Store their lambda value
        admmNeighbors[nodeIndex].lambda[controlIndex] = value;
    }
}
// Update both u_av values based on received messages
void updateControlAverages()
{
    float sum[MAX_NODES] = {0.0f};
    int count[MAX_NODES] = {0};

    // Add our own values
    for (int i = 0; i < MAX_NODES; i++)
    {
        sum[i] += localNode.u[i];
        count[i]++;
    }

    // Add values from neighbors
    for (int i = 0; i < MAX_ADMM_NEIGHBORS; i++)
    {
        if (admmNeighbors[i].active)
        {
            for (int j = 0; j < MAX_NODES; j++)
            {
                // Don't check for positivity, just ensure the value exists
                sum[j] += admmNeighbors[i].u[j];
                count[j]++;
            }
        }
    }

    // Compute averages
    for (int i = 0; i < MAX_NODES; i++)
    {
        localNode.u_av[i] = (count[i] > 0) ? sum[i] / count[i] : localNode.u[i];
    }
}

// Update lambda values after each iteration
void updateLambdaValues()
{
    float rho;
    critical_section_enter_blocking(&commStateLock);
    rho = controlState.rho;
    critical_section_exit(&commStateLock);

    for (int i = 0; i < MAX_NODES; i++)
    {
        localNode.lambda[i] += rho * (localNode.u[i] - localNode.u_av[i]);
    }
}

// Send control values to other nodes
void broadcastControlValues()
{
    // Send u value
    sendSensorReading(CAN_ADDR_BROADCAST, ADMM_MSG_U_VAL, localNode.u[localNode.index]);

    // Also send u_av and lambda values
    sendSensorReading(CAN_ADDR_BROADCAST, ADMM_MSG_U_AV_VAL, localNode.u_av[localNode.index]);
    sendSensorReading(CAN_ADDR_BROADCAST, ADMM_MSG_LAMBDA_VAL, localNode.lambda[localNode.index]);
}

// Main ADMM consensus update function
bool updateADMMConsensus()
{
    static unsigned long lastIterationTime = 0;
    static int iterationCount = 0;
    static bool consensusInitialized = false;
    static bool convergenceReached = false;
    bool controlUpdated = false;
    unsigned long currentTime = millis();

    // Check if ADMM is enabled
    critical_section_enter_blocking(&commStateLock);
    bool admmEnabled = controlState.usingADMM;
    critical_section_exit(&commStateLock);

    if (!admmEnabled)
    {
        consensusInitialized = false;
        convergenceReached = false;
        return false;
    }

    // Initialize ADMM when first enabled
    if (!consensusInitialized)
    {
        // Find our node index based on calibration matrix
        int nodeIdx = -1;

        critical_section_enter_blocking(&commStateLock);
        for (int i = 0; i < commState.calibMatrix.numNodes; i++)
        {
            if (commState.calibMatrix.nodeIds[i] == deviceConfig.nodeId)
            {
                nodeIdx = i;
                break;
            }
        }
        critical_section_exit(&commStateLock);

        // If we couldn't find our index, use a default
        if (nodeIdx < 0)
        {
            // Default index based on node ID as fallback
            if (deviceConfig.nodeId == 33)
                nodeIdx = 0;
            else if (deviceConfig.nodeId == 40)
                nodeIdx = 1;
            else if (deviceConfig.nodeId == 52)
                nodeIdx = 2;
        }

        initADMMNode(localNode, nodeIdx);
        consensusInitialized = true;
        iterationCount = 0;
        lastIterationTime = currentTime;

        // Initial broadcast of our control value
        broadcastControlValues();
        return false;
    }
    // Only update at regular intervals or if we haven't converged
    if ((currentTime - lastIterationTime < 1000) && !convergenceReached)
    {
        return false;
    }

    // First, update external illuminance model to adapt to changing conditions
    updateExternalIlluminanceModel();

    Serial.print("Iteration ");
    Serial.print(iterationCount);
    Serial.print(": u=");
    for (int i = 0; i < MAX_NODES; i++)
    {
        Serial.print(localNode.u[i]);
        Serial.print(" ");
    }
    Serial.println();

    // Only proceed with iterations if we haven't converged
    if (!convergenceReached)
    {
        // Update averages and perform iteration
        updateControlAverages();

        float newControl[MAX_NODES];
        float newCost;
        consensus_iterate(localNode, controlState.rho, newControl, newCost);

        // Update our control values
        for (int i = 0; i < MAX_NODES; i++)
        {
            localNode.u[i] = newControl[i];
        }

        // Update lambda values
        updateLambdaValues();

        // Update step size parameter
        updateRhoParameter();

        // Check for convergence
        convergenceReached = hasConverged();

        // Broadcast updated values
        broadcastControlValues();

        // Log iteration
        Serial.print("ADMM Iteration ");
        Serial.print(iterationCount++);
        if (convergenceReached)
        {
            Serial.println(" - Convergence reached!");
        }
        else
        {
            Serial.println();
        }
    }
    else
    {
        // If we've already converged, just periodically broadcast values
        // to maintain consensus and ensure all nodes have latest values
        if (currentTime - lastIterationTime >= 5000)
        {
            broadcastControlValues();
        }
    }

    // We always return the current ADMM duty cycle - it will be used
    // as a feedforward term in the main control loop
    lastIterationTime = currentTime;

    // We don't directly apply control here anymore - that's handled
    // by the main control loop using PID with feedforward
    return true;
}

/**
 * Output current ADMM optimization data in row-based CSV format
 * Each variable gets its own row for easier plotting
 */
void dumpADMMState()
{
    // Check if ADMM is active
    critical_section_enter_blocking(&commStateLock);
    bool admmEnabled = controlState.usingADMM;
    critical_section_exit(&commStateLock);

    if (!admmEnabled)
    {
        Serial.println("ADMM optimization not active.");
        return;
    }

    Serial.println("\n===== ADMM Optimization State =====");

    // Node and illuminance info
    Serial.print("node_id,");
    Serial.println(deviceConfig.nodeId);

    Serial.print("node_index,");
    Serial.println(localNode.index);

    Serial.print("target_lux,");
    Serial.println(localNode.L);

    Serial.print("external_lux,");
    Serial.println(localNode.d);

    // Get step size
    critical_section_enter_blocking(&commStateLock);
    float rho = controlState.rho;
    critical_section_exit(&commStateLock);
    Serial.print("rho,");
    Serial.println(rho, 4);

    // Print vectors in row format

    // Control variables (u vector)
    Serial.print("u,");
    for (int i = 0; i < MAX_NODES; i++)
    {
        Serial.print(localNode.u[i], 4);
        if (i < MAX_NODES - 1)
            Serial.print(",");
    }
    Serial.println();

    // Average control values (u_av vector)
    Serial.print("u_av,");
    for (int i = 0; i < MAX_NODES; i++)
    {
        Serial.print(localNode.u_av[i], 4);
        if (i < MAX_NODES - 1)
            Serial.print(",");
    }
    Serial.println();

    // Dual variables (lambda vector)
    Serial.print("lambda,");
    for (int i = 0; i < MAX_NODES; i++)
    {
        Serial.print(localNode.lambda[i], 4);
        if (i < MAX_NODES - 1)
            Serial.print(",");
    }
    Serial.println();

    // Cost coefficients
    Serial.print("cost_coef,");
    for (int i = 0; i < MAX_NODES; i++)
    {
        Serial.print(localNode.c[i], 4);
        if (i < MAX_NODES - 1)
            Serial.print(",");
    }
    Serial.println();

    // Gain values
    Serial.print("gain,");
    for (int i = 0; i < MAX_NODES; i++)
    {
        Serial.print(localNode.k[i], 4);
        if (i < MAX_NODES - 1)
            Serial.print(",");
    }
    Serial.println();

    // Derived values

    // Current cost
    float currentCost = evaluate_cost(localNode, localNode.u, rho);
    Serial.print("objective_cost,");
    Serial.println(currentCost, 4);

    // This node's actual duty cycle
    float duty = controlToOutputDuty(localNode.u[localNode.index]);
    Serial.print("duty_cycle,");
    Serial.println(duty, 4);

    // Constraint satisfaction
    bool constraintsSatisfied = check_feasibility(localNode, localNode.u);
    Serial.print("constraints_satisfied,");
    Serial.println(constraintsSatisfied ? "1" : "0");

    // Print neighbor information
    Serial.println("\nNeighbor Data:");
    for (int i = 0; i < MAX_ADMM_NEIGHBORS; i++)
    {
        if (admmNeighbors[i].active)
        {
            Serial.print("neighbor_");
            Serial.print(admmNeighbors[i].nodeId);
            Serial.print(",");
            for (int j = 0; j < MAX_NODES; j++)
            {
                Serial.print(admmNeighbors[i].u[j], 4);
                if (j < MAX_NODES - 1)
                    Serial.print(",");
            }
            Serial.println();
        }
    }

    Serial.println("=================================\n");
}