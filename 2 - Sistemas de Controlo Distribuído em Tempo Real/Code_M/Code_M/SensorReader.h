/**
 * SensorReader.h - Interface do subsistema de sensores
 */

#pragma once
#include <Arduino.h>

//============================================================================
// FUNÇÕES DE INICIALIZAÇÃO
//============================================================================

/**
 * Inicializa o subsistema de sensores
 */
void initSensor();

//============================================================================
// FUNÇÕES DE LEITURA
//============================================================================

/**
 * Lê o valor de iluminância atual com filtragem
 * @return Valor de iluminância em lux
 */
float readLux();

/**
 * Estima a iluminância externa subtraindo a contribuição do LED
 * @return Iluminância externa em lux
 */
float getExternalIlluminance();

/**
 * Obtém a tensão atual no divisor de tensão do LDR
 * @return Valor da tensão em volts
 */
float getVoltageAtLDR();

//============================================================================
// FUNÇÕES DE CALIBRAÇÃO E ADAPTAÇÃO
//============================================================================

/**
 * Calibra o modelo de iluminância medindo a contribuição do LED
 * @return Ganho calibrado do sistema (y2-y1)
 */
float calibrateIlluminanceModel();

/**
 * Realiza calibração abrangente do sistema
 * @param referenceValue Valor de referência de iluminância
 * @return Ganho calibrado do LED
 */
float calibrateSystem(float referenceValue);