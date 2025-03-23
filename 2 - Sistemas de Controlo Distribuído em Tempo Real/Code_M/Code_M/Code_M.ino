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

#include "Configuration.h"
#include "PIDController.h"
#include "LEDController.h"
#include "SensorReader.h"
#include "DataLogger.h"
#include "CANCom.h"
#include "SerialInterface.h"
#include "StateManager.h"

//============================================================================
// VARIÁVEIS GLOBAIS
//============================================================================

// Define KP here
float KP = KP_BASE; // Initial value based on the defined constant

// Other global variables
float setpointLux = SETPOINT_OFF;
float dutyCycle = 0.0;
float refIlluminance = 0.0;
bool occupancy = false;
bool antiWindup = true;
bool feedbackControl = true;
uint8_t nodeID = 1; // Default node ID
bool periodicCANEnabled = PERIODIC_CAN_ENABLED;
bool canMonitorEnabled = CAN_MONITOR_ENABLED;

// --- Temporização ---
unsigned long lastTransmit = 0; // Timestamp da última transmissão CAN

// --- Comunicação CAN-BUS ---
unsigned long lastHeartbeat = 0;        // Timestamp do último heartbeat
unsigned long heartbeatInterval = 5000; // Intervalo entre heartbeats (ms)

// --- Controlador PID ---
PIDController pid(KP, KI, 1.0, 1.0); // Using default Beta=1.0 and Gamma=1.0

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
  setpointLux = 5.0;            // Setpoint inicial
  refIlluminance = setpointLux; // Sincroniza referência com setpoint

  // Initialize KP with base value
  KP = KP_BASE;

  // Perform comprehensive calibration of the sensor and LED gain
  float calibratedGain = calibrateSystem(1.0); // Use a reference value of 1.0 lux

  // Sanity check the calibrated gain
  if (calibratedGain > 0.1 && calibratedGain < MAX_ILLUMINANCE)
  {
    // Calculate KP based on the LED gain from calibration
    // This normalizes the controller to the specific box characteristics
    KP = KP_BASE * (100.0f / calibratedGain);
    Serial.print("Calibrated KP: ");
    Serial.println(KP);
  }
  else
  {
    // Use default value if calibration seems unreliable
    KP = KP_BASE;
    Serial.println("Using default KP value");
  }

  // Update the PID controller with the calibrated gain
  pid.setGains(KP, KI);
  pid.setSetpointWeight(BETA); // Apply setpoint weighting from configuration

  // Cabeçalho para Serial Plotter
  Serial.println("LuxMedido\tSetpoint");
}

// Add a static counter for sampling
static int sampleCounter = 0;
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

  // Only log data every 10th iteration
  sampleCounter++;
  if (sampleCounter >= 10)
  {
    logData(millis(), lux, dutyCycle);
    sampleCounter = 0;
  }

  // Comunicação CAN
  canCommLoop();          // Processamento de mensagens CAN
  processRemoteStreams(); // Processa e envia streams remotos

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