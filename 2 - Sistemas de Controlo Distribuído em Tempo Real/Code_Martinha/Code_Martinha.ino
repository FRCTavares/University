/**
 * Sistema de Controlo Distribuído em Tempo Real
 *
 * Este sistema implementa um controlador de iluminação com:
 * - Controlo PID de iluminação
 * - Comunicação CAN-BUS
 * - Interface de comandos via Serial (USB)
 * - Armazenamento de dados num circular buffer
 */

//============================================================================
// BIBLIOTECAS
//============================================================================

// --- Hardware e Sistema ---
#include <Arduino.h>
#include <pico/unique_id.h>
#include <SPI.h>

// --- Comunicação ---
#include <mcp2515.h>

// --- Componentes do Sistema ---
#include "PIDController.h"
#include "Configuration.h"
#include "CANCom.h"
#include "StateManager.h"
#include "SensorReader.h"
#include "LEDController.h"
#include "DataLogger.h"
#include "SerialInterface.h"

//============================================================================
// VARIÁVEIS GLOBAIS
//============================================================================

// --- Sistema de Controlo de Iluminação ---
float setpointLux = 10.0;    // Nível de iluminação alvo, em lux
float dutyCycle = 0.0;       // Duty Cycle do LED [0.0 a 1.0]
float refIlluminance = 15.0; // Referência base de iluminação
bool occupancy = false;      // Estado de ocupação da mesa
bool antiWindup = false;     // Proteção contra saturação do integrador
bool feedbackControl = true; // Modo de controlo (true=malha fechada, false=manual)

// --- Temporização ---
unsigned long lastTransmit = 0; // Timestamp da última transmissão CAN

// --- Comunicação CAN-BUS ---
bool periodicCANEnabled = PERIODIC_CAN_ENABLED; // Ativa mensagens periódicas
bool canMonitorEnabled = CAN_MONITOR_ENABLED;   // Ativa monitor de mensagens recebidas
uint8_t nodeID = 0;                             // ID deste nó na rede CAN
unsigned long lastHeartbeat = 0;                // Timestamp do último heartbeat
unsigned long heartbeatInterval = 5000;         // Intervalo entre heartbeats (ms)

// --- Controlador PID ---
PIDController pid(KP, KI, KD, FILTER_N, DT); // Inicialização com parâmetros de Configuration.h

//============================================================================
// INICIALIZAÇÃO
//============================================================================
void setup()
{

  // Configuração da comunicação serial
  Serial.begin(115200);

  // Identificação do pico
  pico_unique_board_id_t board_id;
  pico_get_unique_board_id(&board_id);

  // Configuração ADC e PWM
  analogReadResolution(12);  // Resolução de 12 bits para leitura
  analogWriteFreq(30000);    // Frequência PWM de 30kHz
  analogWriteRange(PWM_MAX); // Escala conforme definido em Configuration.h

  // Driver do LED
  initLEDDriver(LED_PIN);
  testLED(); // Teste visual de funcionamento

  // Sistema de Armazenamento
  initStorage(); // Inicializa buffer circular

  // Sistema de controlo de estados
  initStateManager(); // Máquina de estados do luminário

  // Sensor de iluminação
  initSensor(); // Configura o sensor de luz

  // Interface de usuário
  initSerialInterface(); // Inicializa interface de comandos

  Serial.println("Sistema de Controlo Distribuído com CAN-BUS e Interface de Comandos");

  // Comunicação CAN-BUS
  initCANComm(); // Inicializa controlador CAN

  // Configuração inicial de operação
  setpointLux = 15.0;           // Setpoint inicial
  refIlluminance = setpointLux; // Sincroniza referência com setpoint

  // Cabeçalho para Serial Plotter
  Serial.println("LuxMedido\tSetpoint");
}

//============================================================================
// LOOP PRINCIPAL
//============================================================================
void loop()
{
  // Leitura e processamento de comandos via Serial (USB)
  processSerialCommands();

  // Gestão de streaming de dados
  handleStreaming();

  // Leitura do sensor
  float lux = readLux(); // Obter nível atual de iluminação

  // Adaptação às condições ambientais
  adaptToExternalLight(); // Ajustar setpoint baseado em iluminação externa

  // Obtenção do estado atual do sistema
  LuminaireState currentState = getCurrentState();

  // Controlo e atuação
  if (currentState == STATE_OFF)
  {
    // Sistema desligado - Apagar LED
    setLEDDutyCycle(0.0);
  }
  else if (feedbackControl)
  {
    // Controlo em malha fechada (PID)
    float u = pid.compute(setpointLux, lux);
    setLEDPWMValue((int)u);
  }
  else
  {
    // Controlo manual (malha aberta)
    setLEDDutyCycle(dutyCycle);
  }

  // Armazenamento de dados
  logData(millis(), lux, dutyCycle);

  // Comunicação CAN
  canCommLoop(); // Processamento de mensagens CAN

  // Obtenção do timestamp atual
  unsigned long now = millis();

  // Debugging e visualização
  if (DEBUG_MODE && DEBUG_PLOTTING)
  {
    // Output para Serial Plotter
    Serial.print(lux, 2); // Valor medido
    Serial.print("\t");
    Serial.print(setpointLux, 2); // Valor objetivo
    Serial.print("\t");
    Serial.print(30.0, 2); // Limite superior (fixo)
    Serial.print("\t");
    Serial.println(0.0, 2); // Limite inferior (fixo)
  }

  // Sincronização temporal
  // Espera pelo próximo ciclo de controlo
  delay((int)(pid.getSamplingTime() * 1000));
}