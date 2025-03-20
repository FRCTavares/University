#include <Arduino.h>
#include <pico/unique_id.h>
#include <SPI.h>
#include <mcp2515.h>

// ===================== CONFIGURAÇÃO =====================

#define MAX_ILLUMINANCE 2000.0f

// --- Global Constants ---
extern const float MAX_POWER_WATTS;

// --- Control System State ---
extern float setpointLux;    // Desired lux (setpoint)
extern float dutyCycle;      // Current duty cycle [0..1]
extern float refIlluminance; // Reference illuminance
extern bool occupancy;       // Occupancy flag
extern bool antiWindup;      // Anti-windup flag for PID controller
extern bool feedbackControl; // Enable/disable feedback control

// Define luminaire states
enum LuminaireState
{
    STATE_OFF = 0,        // No one in office
    STATE_UNOCCUPIED = 1, // No one at desk, low light
    STATE_OCCUPIED = 2    // Desk is busy, full light
};

// Declare in global section
extern LuminaireState luminaireState;


// --- CAN Communication Flags ---
extern bool periodicCANEnabled; // Enable periodic message sending
extern bool canMonitorEnabled;  // Display received messages
extern uint8_t nodeID;          // This node's identifier

// --- CAN Message Types ---
#define CAN_TYPE_CONTROL 0x00      // Control messages (setpoints, modes)
#define CAN_TYPE_SENSOR 0x01       // Sensor data (illuminance, duty cycle)
#define CAN_TYPE_STATUS 0x02       // Status information (power, mode)
#define CAN_TYPE_CONFIG 0x03       // Configuration parameters
#define CAN_TYPE_ERROR 0x04        // Error reports
#define CAN_TYPE_QUERY 0x05        // Data requests
#define CAN_TYPE_RESPONSE 0x06     // Responses to queries
#define CAN_TYPE_HEARTBEAT 0x07    // Node presence signals
#define CAN_CTRL_STATE_CHANGE 0x10 // Or choose another appropriate value that doesn't conflict

// CAN priority levels
#define CAN_PRIO_HIGH 0x00
#define CAN_PRIO_NORMAL 0x01
#define CAN_PRIO_LOW 0x02
#define CAN_PRIO_LOWEST 0x03

// CAN node addresses
#define CAN_ADDR_BROADCAST 0x00 // Broadcast to all nodes

// --- Calibração do Sensor ---
#define VCC 3.3
#define MY_ADC_RESOLUTION 4095.0
#define FIXED_RESISTOR 10000.0

// Exemplo de referência LDR a ~10 lux
const float R10 = 225000.0;
const float LDR_M = -1.0;
float LDR_B = log10(R10) - LDR_M;

// --- Atribuição de PINOS ---
const int LED_PIN = 15;
const int LDR_PIN = A0;

// --- Configuração PWM ---
const int PWM_MAX = 4095;
const int PWM_MIN = 0;

// --- Configuração do Filtro de Medição ---
const int NUM_SAMPLES = 10;          // Número de amostras para cálculo da média
const float OUTLIER_THRESHOLD = 2.0; // Desvios padrão para rejeição de valores atípicos
const float ALPHA = 0.3;             // Coeficiente do filtro EMA (0-1)

// --- Parâmetros do Controlador PID ---
const float KP = 28.0;  // Ganho proporcional
const float KI = 230.0; // Ganho integral
const float KD = 0.0;   // Ganho derivativo
const float N = 10.0;   // Coeficiente de filtro
const float DT = 0.01;  // Tempo de amostragem (segundos)

// --- Modelo de Consumo de Energia ---
const float MAX_POWER_WATTS = 1.0; // Potência máxima a 100% de duty cycle

// ===================== VARIÁVEIS GLOBAIS =====================

// --- Estado do Sistema de Controlo ---
float setpointLux = 10.0;    // Lux desejado (setpoint)
float dutyCycle = 0.0;       // Duty cycle atual [0..1]
float refIlluminance = 15.0; // Iluminância de referência
bool occupancy = false;      // Flag de ocupação
bool antiWindup = false;     // Flag anti-windup para o controlador PID
bool feedbackControl = true; // Ativar/desativar controlo por feedback

// --- Estado de Calibração do Sensor ---
float calibrationOffset = 0.0; // Desvio de calibração
float lastFilteredLux = -1.0;  // Último valor filtrado para EMA

// --- Comunicação e Temporização ---
unsigned long lastTransmit = 0; // Último momento de transmissão CAN
#define LOG_SIZE 1200           // Tamanho do buffer circular

// --- Comunicação CAN ---
bool periodicCANEnabled = false;        // Ativar envio periódico de mensagens
bool canMonitorEnabled = false;         // Mostrar mensagens recebidas
uint8_t nodeID = 0;                     // Será atribuído em initCANComm()
unsigned long lastHeartbeat = 0;        // Tempo do último heartbeat
unsigned long heartbeatInterval = 5000; // Enviar heartbeat a cada 5 segundos

// --- Objeto Controlador ---
PIDController pid(KP, KI, KD, N, DT);

// --- Gestão de Estado do Luminária ---
LuminaireState luminaireState = STATE_UNOCCUPIED;

// --- Variáveis de Streaming ---
bool streamingEnabled = false;    // Flag para ativar streaming
String streamingVar = "";         // Variável para fazer streaming
int streamingIndex = 0;           // Índice para sistemas multi-nó
unsigned long lastStreamTime = 0; // Timestamp da última atualização de stream

// --- Flags de Depuração ---
bool DEBUG_MODE = false;     // Interruptor principal de depuração
bool DEBUG_LED = false;      // Mensagens de depuração do driver LED
bool DEBUG_SENSOR = false;   // Depuração de leituras do sensor
bool DEBUG_PID = false;      // Depuração de controlo PID
bool DEBUG_PLOTTING = false; // Saída para o plotter serial

// Setpoints de iluminância para diferentes estados
const float SETPOINT_OFF = 0.0;        // 0 lux quando desligado
const float SETPOINT_UNOCCUPIED = 5.0; // 5 lux quando desocupado
const float SETPOINT_OCCUPIED = 15.0;  // 15 lux quando ocupado

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

// --- Gestão de Vizinhos ---
struct NeighborInfo
{
  uint8_t nodeId;
  float lastLux;
  float lastDuty;
  LuminaireState state;
  unsigned long lastUpdate;
  bool isActive;
};

const int MAX_NEIGHBORS = 5;
NeighborInfo neighbors[MAX_NEIGHBORS];

// Atualizar informação dos vizinhos ao receber mensagens CAN
void updateNeighborInfo(uint8_t nodeId, uint8_t sensorType, float value)
{
  int emptySlot = -1;

  // Encontrar vizinho existente ou slot vazio
  for (int i = 0; i < MAX_NEIGHBORS; i++)
  {
    if (neighbors[i].isActive && neighbors[i].nodeId == nodeId)
    {
      // Atualizar vizinho existente
      if (sensorType == 0)
        neighbors[i].lastLux = value;
      else if (sensorType == 1)
        neighbors[i].lastDuty = value;
      else if (sensorType == 2)
        neighbors[i].state = (LuminaireState)((int)value);

      neighbors[i].lastUpdate = millis();
      return;
    }

    if (!neighbors[i].isActive && emptySlot < 0)
    {
      emptySlot = i;
    }
  }

  // Adicionar como novo vizinho se houver slot disponível
  if (emptySlot >= 0)
  {
    neighbors[emptySlot].nodeId = nodeId;
    neighbors[emptySlot].isActive = true;
    neighbors[emptySlot].lastUpdate = millis();

    if (sensorType == 0)
      neighbors[emptySlot].lastLux = value;
    else if (sensorType == 1)
      neighbors[emptySlot].lastDuty = value;
    else if (sensorType == 2)
      neighbors[emptySlot].state = (LuminaireState)((int)value);
  }
}

// Calcular contribuição de luz dos vizinhos
float getNeighborsContribution()
{
  float totalContribution = 0.0;
  unsigned long currentTime = millis();
  const unsigned long NEIGHBOR_TIMEOUT = 10000; // 10 segundos de timeout

  for (int i = 0; i < MAX_NEIGHBORS; i++)
  {
    if (neighbors[i].isActive)
    {
      // Marcar como inativo se for demasiado antigo
      if (currentTime - neighbors[i].lastUpdate > NEIGHBOR_TIMEOUT)
      {
        neighbors[i].isActive = false;
        continue;
      }

      // Ignorar vizinhos que estão desligados
      if (neighbors[i].state == STATE_OFF)
        continue;

      // Modelo simples de contribuição de luz - necessitaria de calibração numa implementação real
      float contribution = neighbors[i].lastDuty * 3.0; // Cada vizinho a 100% adiciona ~3 lux
      totalContribution += contribution;
    }
  }

  return totalContribution;
}

// Algoritmo de coordenação para otimizar o uso de energia
void coordinateWithNeighbors()
{
  // Calcular contribuição total de luz dos vizinhos
  float neighborContribution = getNeighborsContribution();

  if (neighborContribution > 0.5)
  { // Apenas ajustar se a contribuição for significativa
    // Ajustar o nosso alvo para considerar a luz dos vizinhos
    float adjustedTarget = max(0.0f, setpointLux - neighborContribution * 0.8);

    // Ajuste dinâmico do PID baseado na cooperação
    pid.setTarget(adjustedTarget);
  }
}

// ===================== FUNÇÕES DO SENSOR =====================

/**
 * Medição LUX avançada com filtragem multi-estágio:
 * 1. Múltiplas amostras para reduzir ruído
 * 2. Rejeição estatística de outliers
 * 3. Filtragem EMA para suavização temporal
 * 4. Aplicação de offset de calibração
 */
float readLux()
{
  float samples[NUM_SAMPLES];
  float sum = 0.0;
  float count = 0.0;

  // 1. Recolher múltiplas amostras para reduzir ruído
  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    int adcValue = analogRead(LDR_PIN);
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;

    // Ignorar leituras inválidas
    if (voltage <= 0.0)
    {
      continue;
    }

    // Calcular resistência do LDR usando a fórmula do divisor de tensão
    float resistance = FIXED_RESISTOR * (VCC / voltage - 1.0);

    // Converter resistência para LUX usando fórmula de calibração
    float logR = log10(resistance);
    float logLux = (logR - LDR_B) / LDR_M;
    float luxValue = pow(10, logLux);

    samples[i] = luxValue;
    sum += luxValue;
    count++;
  }

  if (count == 0)
    return 0.0; // Sem leituras válidas

  // 2. Calcular média e desvio padrão
  float mean = sum / count;
  float variance = 0.0;

  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    if (samples[i] > 0)
    { // Considerar apenas amostras válidas
      variance += sq(samples[i] - mean);
    }
  }
  variance /= count;
  float stdDev = sqrt(variance);

  // 3. Filtrar outliers e recalcular média
  float filteredSum = 0.0;
  float filteredCount = 0.0;

  for (int i = 0; i < NUM_SAMPLES; i++)
  {
    if (samples[i] > 0 && abs(samples[i] - mean) <= OUTLIER_THRESHOLD * stdDev)
    {
      filteredSum += samples[i];
      filteredCount++;
    }
  }

  float filteredMean = (filteredCount > 0) ? filteredSum / filteredCount : mean;

  // 4. Aplicar filtro de média móvel exponencial (EMA) para suavização temporal
  if (lastFilteredLux < 0)
  {
    lastFilteredLux = filteredMean; // Primeira leitura válida
  }
  else
  {
    lastFilteredLux = ALPHA * filteredMean + (1.0 - ALPHA) * lastFilteredLux;
  }

  // 5. Aplicar offset de calibração e verificação de limites de segurança
  float calibratedLux = lastFilteredLux + calibrationOffset;
  if (calibratedLux < 0.0)
    calibratedLux = 0.0;

  return calibratedLux;
}

/**
 * Calibra o sensor LUX usando um valor de referência conhecido
 * Deve ser chamado quando um medidor de luz externo confiável estiver disponível
 */
void calibrateLuxSensor(float knownLux)
{
  float measuredLux = 0.0;
  const int CAL_SAMPLES = 20;

  for (int i = 0; i < CAL_SAMPLES; i++)
  {
    // Usar uma leitura bruta especial para evitar a calibração existente
    int adcValue = analogRead(LDR_PIN);
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
    if (voltage <= 0.0)
      continue;

    float resistance = FIXED_RESISTOR * (VCC / voltage - 1.0);
    float logR = log10(resistance);
    float logLux = (logR - LDR_B) / LDR_M;
    float rawLux = pow(10, logLux);

    measuredLux += rawLux;
    delay(50);
  }
  measuredLux /= CAL_SAMPLES;

  // Calcular o offset necessário
  calibrationOffset = knownLux - measuredLux;

  Serial.print("Sensor calibrado: offset = ");
  Serial.println(calibrationOffset);
}

// Retornar tensão no pino LDR
float getVoltageAtLDR()
{
  int adcValue = analogRead(LDR_PIN);
  return (adcValue / MY_ADC_RESOLUTION) * VCC;
}

// ===================== FUNÇÕES UTILITÁRIAS =====================
// --- Testar funcionalidade do LED ---
// Adicionar este código ao seu main.ino temporariamente para teste

// --- Adaptação à Luz Externa ---
float lastExternalLux = 0.0;
float externalLuxAverage = 0.0;
const float EXT_LUX_ALPHA = 0.05; // Média de movimento lento para estabilidade

// Cálculo aprimorado de iluminância externa
float getExternalIlluminance()
{
  float measuredLux = readLux();

  // Modelo não-linear mais preciso da contribuição do LED
  float ledContribution;
  if (dutyCycle < 0.1)
  {
    ledContribution = dutyCycle * 15.0; // Linear em duty cycles baixos
  }
  else
  {
    ledContribution = dutyCycle * dutyCycle * 35.0; // Não-linear em duty cycles mais altos
  }

  // Calcular estimativa atual de lux externo
  float currentExternalLux = max(0.0f, measuredLux - ledContribution);

  // Aplicar média de movimento lento à iluminância externa
  if (lastExternalLux == 0.0)
  {
    externalLuxAverage = currentExternalLux;
  }
  else
  {
    externalLuxAverage = EXT_LUX_ALPHA * currentExternalLux +
                         (1.0 - EXT_LUX_ALPHA) * externalLuxAverage;
  }

  lastExternalLux = currentExternalLux;
  return externalLuxAverage;
}

// Adaptar controlo às mudanças de luz externa
void adaptToExternalLight()
{
  static unsigned long lastAdaptTime = 0;
  static float previousExternal = -1.0;

  // Verificar apenas a cada 5 segundos para evitar ajustes rápidos
  if (millis() - lastAdaptTime < 5000)
  {
    return;
  }
  lastAdaptTime = millis();

  // Obter iluminância externa atual
  float externalLux = getExternalIlluminance();

  // Ignorar primeira execução ou quando em modo manual
  if (previousExternal < 0 || !feedbackControl)
  {
    previousExternal = externalLux;
    return;
  }

  // Se a luz externa mudou significativamente (>1 lux)
  if (abs(externalLux - previousExternal) > 1.0)
  {
    // Calcular quanto do nosso setpoint é satisfeito pela luz externa
    float externalContribution = min(externalLux, setpointLux);
    float requiredFromLED = max(0.0f, setpointLux - externalContribution);

    // Pré-ajustar duty cycle com base na luz externa (controlo feedforward)
    float estimatedDuty = requiredFromLED / 30.0; // Assumindo 30 lux na potência máxima
    estimatedDuty = constrain(estimatedDuty, 0.0, 1.0);

    // Aplicar um pequeno ajuste para ajudar o PID a convergir mais rapidamente
    float currentDuty = getLEDDutyCycle();
    float newDuty = currentDuty * 0.7 + estimatedDuty * 0.3; // Ajuste gradual

    setLEDDutyCycle(newDuty);

    if (DEBUG_MODE && DEBUG_SENSOR)
    {
      Serial.print("Adaptação à luz externa: ");
      Serial.print(externalLux);
      Serial.print(" lux, necessário do LED: ");
      Serial.print(requiredFromLED);
      Serial.print(" lux, duty ajustado: ");
      Serial.println(newDuty, 3);
    }

    previousExternal = externalLux;
  }
}

// --- Retornar estimativa de consumo de energia ---
float getPowerConsumption()
{
  return dutyCycle * MAX_POWER_WATTS;
}

// --- Retornar tempo decorrido desde o arranque em segundos ---
unsigned long getElapsedTime()
{
  return millis() / 1000;
}

// --- Testar funcionalidade do LED ---
void testLED()
{
  // Teste rápido do LED aumentando e diminuindo a intensidade
  Serial.println("A testar LED...");

  for (int i = 0; i <= 100; i += 10)
  {
    setLEDPercentage(i);
    delay(50);
  }

  for (int i = 100; i >= 0; i -= 10)
  {
    setLEDPercentage(i);
    delay(50);
  }

  // Definir LED como desligado após o teste
  setLEDDutyCycle(0.0);
  Serial.println("Teste do LED concluído.");
}

// ===================== FUNÇÕES DE STREAMING =====================

// --- Iniciar streaming de uma variável ---
void startStream(const String &var, int index)
{
  streamingEnabled = true;
  streamingVar = var;
  streamingIndex = index;
  Serial.println("ack");
}

// --- Parar streaming ---
void stopStream(const String &var, int index)
{
  streamingEnabled = false;
  streamingVar = "";  // Limpar a variável
  Serial.print("Streaming parado para ");
  Serial.print(var);
  Serial.print(" no nó ");
  Serial.println(index);
}

// --- Gerir streaming no loop principal ---
void handleStreaming()
{
  if (!streamingEnabled || (millis() - lastStreamTime < 500))
  {
    return; // Sem streaming ou ainda não é tempo de fazer streaming
  }

  unsigned long currentTime = millis();
  lastStreamTime = currentTime;
  String var = streamingVar;
  int index = streamingIndex;

  if (var.equalsIgnoreCase("y"))
  {
    float lux = readLux();
    Serial.print("s ");  // Adicionar prefixo "s"
    Serial.print(var);
    Serial.print(" ");
    Serial.print(index);
    Serial.print(" ");
    Serial.print(lux, 2);
    Serial.print(" ");
    Serial.println(currentTime);  // Adicionar timestamp
  }
  else if (var.equalsIgnoreCase("u"))
  {
    Serial.print("s ");  // Adicionar prefixo "s"
    Serial.print(var);
    Serial.print(" ");
    Serial.print(index);
    Serial.print(" ");
    Serial.print(dutyCycle, 4);
    Serial.print(" ");
    Serial.println(currentTime);  // Adicionar timestamp
  }
  else if (var.equalsIgnoreCase("p"))
  {
    float power = getPowerConsumption();
    Serial.print("s ");  // Adicionar prefixo "s"
    Serial.print(var);
    Serial.print(" ");
    Serial.print(index);
    Serial.print(" ");
    Serial.print(power, 2);
    Serial.print(" ");
    Serial.println(currentTime);  // Adicionar timestamp
  }
}

// Retornar buffer do último minuto como uma string separada por vírgulas
String getLastMinuteBuffer(const String &var, int index)
{
  String result = "";
  int count = getLogCount();
  if (count == 0)
    return result;

  LogEntry *logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;

  // Número máximo de amostras a retornar (para evitar overflow no buffer serial)
  const int MAX_SAMPLES = 60;
  int sampleCount = min(count, MAX_SAMPLES);

  // Calcular passo para obter amostras distribuídas uniformemente
  int step = count > MAX_SAMPLES ? count / MAX_SAMPLES : 1;

  for (int i = 0; i < count; i += step)
  {
    int realIndex = (startIndex + i) % LOG_SIZE;

    if (var.equalsIgnoreCase("y"))
    {
      // Para valores de iluminância
      result += String(logBuffer[realIndex].lux, 1);
    }
    else if (var.equalsIgnoreCase("u"))
    {
      // Para valores de duty cycle
      result += String(logBuffer[realIndex].duty, 3);
    }

    if (i + step < count)
    {
      result += ",";
    }
  }

  return result;
}

// Variáveis estáticas para o módulo
static int ledPin = -1;
static int pwmMax = 4095;
static int pwmMin = 0;

// --- Constantes de Configuração PWM ---
const unsigned int PWM_FREQUENCY = 30000;


void initLEDDriver(int pin) {
    ledPin = pin;
    pinMode(ledPin, OUTPUT);
    
    // Configurar PWM com definições ótimas
    analogWriteRange(pwmMax);
    analogWriteFreq(PWM_FREQUENCY);
    
    // Começar com o LED desligado
    analogWrite(ledPin, pwmMin);
    dutyCycle = 0.0; // Utilizar a variável global de duty cycle
    
    // Mensagem de depuração apenas se a depuração estiver ativada
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("Driver LED inicializado no pino ");
        Serial.println(pin);
    }
}

void setLEDDutyCycle(float newDutyCycle) {
    // Validar e restringir entrada
    if (isnan(newDutyCycle) || isinf(newDutyCycle)) {
        return; // Proteger contra entradas inválidas
    }
    
    // Restringir para intervalo válido
    newDutyCycle = constrain(newDutyCycle, 0.0f, 1.0f);
    
    // Aplicar duty cycle
    int pwmValue = (int)(newDutyCycle * pwmMax);
    analogWrite(ledPin, pwmValue);
    
    // Atualizar o duty cycle global
    dutyCycle = newDutyCycle;
    
    // Mensagem de depuração apenas se a depuração estiver ativada
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("Duty cycle do LED definido para: ");
        Serial.println(newDutyCycle, 3);
    }
}

void setLEDPercentage(float percentage) {
    percentage = constrain(percentage, 0.0f, 100.0f);
    float newDutyCycle = percentage / 100.0f;
    setLEDDutyCycle(newDutyCycle);
    
    // Mensagem de depuração apenas se a depuração estiver ativada
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("Percentagem do LED definida para: ");
        Serial.println(percentage, 1);
    }
}

void setLEDPWMValue(int pwmValue) {
    pwmValue = constrain(pwmValue, pwmMin, pwmMax);
    analogWrite(ledPin, pwmValue);
    
    // Atualizar duty cycle global
    dutyCycle = (float)pwmValue / pwmMax;
    
    // Mensagem de depuração apenas se a depuração estiver ativada
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("Valor PWM do LED definido para: ");
        Serial.println(pwmValue);
    }
}

void setLEDPower(float powerWatts) {
    powerWatts = constrain(powerWatts, 0.0f, MAX_POWER_WATTS);
    float newDutyCycle = powerWatts / MAX_POWER_WATTS;
    setLEDDutyCycle(newDutyCycle);
    
    // Mensagem de depuração apenas se a depuração estiver ativada
    if (DEBUG_MODE && DEBUG_LED) {
        Serial.print("Potência do LED definida para: ");
        Serial.println(powerWatts, 3);
    }
}

float getLEDDutyCycle() {
    return dutyCycle;
}

float getLEDPercentage() {
    return dutyCycle * 100.0f;
}

int getLEDPWMValue() {
    return (int)(dutyCycle * pwmMax);
}

float getLEDPower() {
    return dutyCycle * MAX_POWER_WATTS;
}


class PIDController
{
public:
    PIDController(float kp, float ki, float kd, float n, float samplingTime);
    float compute(float setpoint, float measurement);
    void reset();
    void setGains(float kp, float ki, float kd);
    float getSamplingTime() const; // Added getter for sampling time
    // Set internal target for coordination purposes
    void setTarget(float newTarget);

private:
    float Kp;    // Proportional gain
    float Ki;    // Integral gain
    float Kd;    // Derivative gain
    float N;     // Filter coefficient for derivative
    float h;     // Sampling time
    float Iterm; // Integral term
    float Dterm; // Derivative term with filtering
    float e_old; // Previous error

    // Make sure these are declared in the header
    float internalTarget; // Target for coordination
    bool useInternalTarget;
};

PIDController::PIDController(float kp, float ki, float kd, float n, float samplingTime)
    : Kp(kp), Ki(ki), Kd(kd), N(n), h(samplingTime), Iterm(0), Dterm(0), e_old(0),
      internalTarget(0), useInternalTarget(false) {}

float PIDController::compute(float setpoint, float measurement)
{
    // Usar alvo interno se definido pela lógica de coordenação
    float actualSetpoint = useInternalTarget ? internalTarget : setpoint;

    float e = actualSetpoint - measurement;

    // Saída de depuração
    if (DEBUG_MODE && DEBUG_PID)
    {
        Serial.print("PID: SP=");
        Serial.print(actualSetpoint);
        Serial.print(" PV=");
        Serial.print(measurement);
        Serial.print(" e=");
        Serial.println(e);
    }

    // Termo proporcional
    float Pterm = Kp * e;

    // Termo derivativo com filtragem
    float derivative = (e - e_old) / h;
    float alpha = N * h;
    Dterm = (alpha * derivative + Dterm) / (1 + alpha);
    float D_out = Kd * Dterm;

    // Calcular ação de controlo não saturada
    float u_unsat = Pterm + Iterm + D_out;

    // Anti-windup: Apenas integrar se o controlo não estiver saturado
    const int PWM_MAX = 4095;
    const int PWM_MIN = 0;
    if ((u_unsat < PWM_MAX || e < 0) && (u_unsat > PWM_MIN || e > 0))
    {
        Iterm += Ki * e * h;
    }

    e_old = e;
    return Pterm + Iterm + D_out;
}

void PIDController::reset()
{
    Iterm = 0;
    Dterm = 0;
    e_old = 0;
}

void PIDController::setGains(float kp, float ki, float kd)
{
    Kp = kp;
    Ki = ki;
    Kd = kd;
}

float PIDController::getSamplingTime() const
{
    return h;
}

void PIDController::setTarget(float newTarget)
{
    // Isto permite que algoritmos de coordenação ajustem temporariamente o alvo
    // sem alterar o setpoint definido pelo utilizador
    internalTarget = newTarget;
    useInternalTarget = true;
}


LogEntry logBuffer[LOG_SIZE];
int logIndex = 0;
bool bufferFull = false;

void initStorage() {
  logIndex = 0;
  bufferFull = false;
}

void logData(unsigned long timestamp, float lux, float duty) {
  logBuffer[logIndex].timestamp = timestamp;
  logBuffer[logIndex].lux = lux;
  logBuffer[logIndex].duty = duty;
  
  logIndex++;
  if (logIndex >= LOG_SIZE) {
    logIndex = 0;
    bufferFull = true;
  }
}

void dumpBufferToSerial() {
  Serial.println("timestamp_ms,rawLux,duty");
  int count = bufferFull ? LOG_SIZE : logIndex;
  int startIndex = bufferFull ? logIndex : 0;
  
  for (int i = 0; i < count; i++) {
    int realIndex = (startIndex + i) % LOG_SIZE;
    unsigned long t = logBuffer[realIndex].timestamp;
    float lx = logBuffer[realIndex].lux;
    float d = logBuffer[realIndex].duty;
    
    Serial.print(t);
    Serial.print(",");
    Serial.print(lx, 2);
    Serial.print(",");
    Serial.println(d, 4);
  }
  
  Serial.println("End of dump.\n");
}

LogEntry* getLogBuffer() {
  return logBuffer;
}

int getLogCount() {
  return bufferFull ? LOG_SIZE : logIndex;
}

bool isBufferFull() {
  return bufferFull;
}

int getCurrentIndex() {
  return logIndex;
}


// Para Cálculo de Energia
const float Pmax = 1.0;  // Potência máxima do LED em Watts
extern float setpointLux; // Declarado em main.ino

void computeAndPrintMetrics() {
  float E = computeEnergyFromBuffer();
  float VE = computeVisibilityErrorFromBuffer();
  float F = computeFlickerFromBuffer();

  Serial.println("----- Métricas do Buffer Circular -----");
  Serial.print("Energia (J aprox.): ");
  Serial.println(E, 4);
  Serial.print("Erro de Visibilidade (lux): ");
  Serial.println(VE, 2);
  Serial.print("Cintilação: ");
  Serial.println(F, 4);
  Serial.println("----------------------------------------\n");
}


float computeEnergyFromBuffer() {
  int count = getLogCount();
  if (count < 2) return 0.0;

  LogEntry* logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;
  unsigned long prevTime = 0;
  float prevDuty = 0.0;
  bool first = true;
  float totalE = 0.0;

  for (int i = 0; i < count; i++) {
    int realIndex = (startIndex + i) % LOG_SIZE;
    unsigned long t = logBuffer[realIndex].timestamp;
    float d = logBuffer[realIndex].duty;
    
    if (!first) {
      float dt = (t - prevTime) / 1000.0;
      totalE += (Pmax * prevDuty * dt);
    } else {
      first = false;
    }
    
    prevTime = t;
    prevDuty = d;
  }
  
  return totalE;
}

float computeVisibilityErrorFromBuffer() {
  int count = getLogCount();
  if (count == 0) return 0.0;

  LogEntry* logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;
  float totalErr = 0.0;
  int sampleCount = 0;

  for (int i = 0; i < count; i++) {
    int realIndex = (startIndex + i) % LOG_SIZE;
    float measuredLux = logBuffer[realIndex].lux;
    
    if (measuredLux < setpointLux) {
      totalErr += (setpointLux - measuredLux);
    }
    
    sampleCount++;
  }
  
  if (sampleCount == 0) return 0.0;
  return (totalErr / sampleCount);
}

float computeFlickerFromBuffer() {
  int count = getLogCount();
  if (count < 3) return 0.0;

  LogEntry* logBuffer = getLogBuffer();
  int startIndex = isBufferFull() ? getCurrentIndex() : 0;
  float flickerSum = 0.0;
  int flickerCount = 0;

  bool first = true, second = false;
  float d0, d1;

  for (int i = 0; i < count; i++) {
    int realIndex = (startIndex + i) % LOG_SIZE;
    float d2 = logBuffer[realIndex].duty;
    
    if (first) {
      d0 = d2;
      first = false;
      second = false;
    }
    else if (!second) {
      d1 = d2;
      second = true;
    }
    else {
      float diff1 = d1 - d0;
      float diff2 = d2 - d1;
      
      if (diff1 * diff2 < 0.0) {
        flickerSum += (fabs(diff1) + fabs(diff2));
        flickerCount++;
      }
      
      d0 = d1;
      d1 = d2;
    }
  }
  
  if (flickerCount == 0) return 0.0;
  return (flickerSum / flickerCount);
}





// Auxiliar para dividir uma linha de comando em tokens por espaço
static void parseTokens(const String &cmd, String tokens[], int maxTokens, int &numFound)
{
  numFound = 0;
  int startIdx = 0;
  while (numFound < maxTokens)
  {
    int spaceIdx = cmd.indexOf(' ', startIdx);
    if (spaceIdx == -1)
    {
      if (startIdx < (int)cmd.length())
      {
        tokens[numFound++] = cmd.substring(startIdx);
      }
      break;
    }
    tokens[numFound++] = cmd.substring(startIdx, spaceIdx);
    startIdx = spaceIdx + 1;
  }
}

// Função auxiliar para verificar se o comando deve ser executado localmente ou reencaminhado
static bool shouldForwardCommand(uint8_t targetNode)
{
  // Se o destino for transmissão (0) ou corresponder a este nó, processar localmente
  if (targetNode == 0 || targetNode == nodeID)
  {
    return false;
  }
  // Caso contrário, reencaminhar para o nó de destino
  return true;
}

// Processa uma única linha de comando da porta Serial
static void processCommandLine(const String &cmdLine)
{
  String trimmed = cmdLine;
  trimmed.trim();
  if (trimmed.length() == 0)
    return;

  // Tokenizar
  const int MAX_TOKENS = 6;
  String tokens[MAX_TOKENS];
  int numTokens = 0;
  parseTokens(trimmed, tokens, MAX_TOKENS, numTokens);
  if (numTokens == 0)
    return;

  String c0 = tokens[0];
  //c0.toLowerCase();

  // ------------------- COMANDOS DE CONTROLO -------------------

  // "u <i> <val>" => definir duty cycle
  if (c0 == "u")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    float val = tokens[2].toFloat();

    if (val < 0.0f || val > 1.0f)
    {
      Serial.println("err");
      return;
    }

    // Verificar se precisamos de reencaminhar este comando
    if (shouldForwardCommand(targetNode))
    {
      // Reencaminhar para nó específico - tipo de controlo 4 = duty cycle
      if (sendControlCommand(targetNode, 4, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha no reencaminhamento CAN");
      }
      return;
    }

    // Tratar o caso de difusão (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 4, val))
      {
        // Também aplicar localmente, pois a difusão inclui este nó
        setLEDDutyCycle(val);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha na difusão CAN");
      }
      return;
    }

    // Aplicar localmente
    setLEDDutyCycle(val);
    Serial.println("ack");
    return;
  }
  // "p <i> <percentagem>" => definir LED por percentagem
  else if (c0 == "p")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    float val = tokens[2].toFloat();

    if (val < 0.0f || val > 100.0f)
    {
      Serial.println("err");
      return;
    }

    // Verificar se precisamos de reencaminhar este comando
    if (shouldForwardCommand(targetNode))
    {
      // Reencaminhar para nó específico - tipo de controlo 5 = percentagem
      if (sendControlCommand(targetNode, 5, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha no reencaminhamento CAN");
      }
      return;
    }

    // Tratar o caso de difusão (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 5, val))
      {
        // Também aplicar localmente, pois a difusão inclui este nó
        setLEDDutyCycle(val);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha na difusão CAN");
      }
      return;
    }

    // Aplicar localmente

    setLEDPercentage(val);
    Serial.println("ack");
    return;
  }
  // "w <i> <watts>" => definir LED por potência em watts
  else if (c0 == "w")
  {

    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    float val = tokens[2].toFloat();

    if (val < 0.0f || val > MAX_POWER_WATTS)
    {
      Serial.println("err");
      return;
    }

    // Verificar se precisamos de reencaminhar este comando
    if (shouldForwardCommand(targetNode))
    {
      // Reencaminhar para nó específico - tipo de controlo 6 = potência em watts
      if (sendControlCommand(targetNode, 6, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha no reencaminhamento CAN");
      }
      return;
    }

    // Tratar o caso de difusão (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 6, val))
      {
        // Também aplicar localmente, pois a difusão inclui este nó
        setLEDPower(val);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha na difusão CAN");
      }
      return;
    }

    setLEDPower(val);
    Serial.println("ack");
    return;
  }
  // "o <i> <val>" => definir estado de ocupação
  else if (c0 == "o")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    int val = tokens[2].toInt();

    if (val != 0 && val != 1)
    {
      Serial.println("err");
      return;
    }

    // Verificar se precisamos de reencaminhar este comando
    if (shouldForwardCommand(targetNode))
    {
      // Reencaminhar para nó específico - tipo de controlo 7 = ocupação
      if (sendControlCommand(targetNode, 7, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha no reencaminhamento CAN");
      }
      return;
    }

    // Tratar o caso de difusão (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 7, val))
      {
        // Também aplicar localmente, pois a difusão inclui este nó
        occupancy = (val == 1);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha na difusão CAN");
      }
      return;
    }

    occupancy = (val == 1);
    Serial.println("ack");
    return;
  }
  // "a <i> <val>" => ativar/desativar anti-windup
  else if (c0 == "a")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    int val = tokens[2].toInt();

    if (val != 0 && val != 1)
    {
      Serial.println("err");
      return;
    }

    // Verificar se precisamos de reencaminhar este comando
    if (shouldForwardCommand(targetNode))
    {
      // Reencaminhar para nó específico - tipo de controlo 8 = anti-windup
      if (sendControlCommand(targetNode, 8, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha no reencaminhamento CAN");
      }
      return;
    }

    // Tratar o caso de difusão (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 8, val))
      {
        // Também aplicar localmente, pois a difusão inclui este nó
        antiWindup = (val == 1);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha na difusão CAN");
      }
      return;
    }

    antiWindup = (val == 1);
    Serial.println("ack");
    return;
  }
  // "f <i> <val>" => ativar/desativar controlo por feedback
  else if (c0 == "f")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    int val = tokens[2].toInt();

    if (val != 0 && val != 1)
    {
      Serial.println("err");
      return;
    }

    // Verificar se precisamos de reencaminhar este comando
    if (shouldForwardCommand(targetNode))
    {
      // Reencaminhar para nó específico - tipo de controlo 9 = controlo por feedback
      if (sendControlCommand(targetNode, 9, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha no reencaminhamento CAN");
      }
      return;
    }

    // Tratar o caso de difusão (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 9, val))
      {
        // Também aplicar localmente, pois a difusão inclui este nó
        feedbackControl = (val == 1);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha na difusão CAN");
      }
      return;
    }

    feedbackControl = (val == 1);
    Serial.println("ack");
    return;
  }
  // "r <i> <val>" => definir referência de iluminância
  else if (c0 == "r")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    float val = tokens[2].toFloat();

    if (val < 0.0f || val > MAX_ILLUMINANCE)
    {
      Serial.println("err");
      return;
    }

    // Verificar se precisamos de reencaminhar este comando
    if (shouldForwardCommand(targetNode))
    {
      // Reencaminhar para nó específico - tipo de controlo 10 = referência de iluminância
      if (sendControlCommand(targetNode, 10, val))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha no reencaminhamento CAN");
      }
      return;
    }

    // Tratar o caso de difusão (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 10, val))
      {
        // Também aplicar localmente, pois a difusão inclui este nó
        refIlluminance = val;
        setpointLux = val;
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha na difusão CAN");
      }
      return;
    }

    refIlluminance = val;
    setpointLux = val;
    Serial.println("ack");
    return;
  }
  // "s <x> <i>" => iniciar fluxo da variável em tempo real <x> para a secretária <i>
  else if (c0 == "s")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    String var = tokens[1];
    uint8_t targetNode = tokens[2].toInt();

    // Verificar se precisamos de reencaminhar este comando
    if (shouldForwardCommand(targetNode))
    {
      // Início de fluxo = tipo de controlo 11
      // Codificaremos o tipo de variável no campo de valor:
      // 'y'=0, 'u'=1, 'p'=2, etc.
      float varCode = 0; // Predefinido para 'y' (lux)

      if (var.equalsIgnoreCase("u"))
        varCode = 1;
      else if (var.equalsIgnoreCase("p"))
        varCode = 2;
      else if (var.equalsIgnoreCase("o"))
        varCode = 3;
      else if (var.equalsIgnoreCase("a"))
        varCode = 4;
      else if (var.equalsIgnoreCase("f"))
        varCode = 5;
      else if (var.equalsIgnoreCase("r"))
        varCode = 6;
      else if (var.equalsIgnoreCase("y"))
        varCode = 0;
      else if (var.equalsIgnoreCase("v"))
        varCode = 7;
      else if (var.equalsIgnoreCase("d"))
        varCode = 8;
      else if (var.equalsIgnoreCase("t"))
        varCode = 9;
      else if (var.equalsIgnoreCase("V"))
        varCode = 10;
      else if (var.equalsIgnoreCase("F"))
        varCode = 11;
      else if (var.equalsIgnoreCase("E"))
        varCode = 12;

      if (sendControlCommand(targetNode, 11, varCode))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha no reencaminhamento CAN");
      }
      return;
    }
    // Tratar localmente
    startStream(var, targetNode);
    return;
  }
  // "S <x> <i>" => parar fluxo
  else if (c0 == "S")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    String var = tokens[1];
    uint8_t targetNode = tokens[2].toInt();

    // Verificar se precisamos de reencaminhar este comando
    if (shouldForwardCommand(targetNode))
    {
      // Parar fluxo = tipo de controlo 12
      // Codificaremos o tipo de variável no campo de valor
      float varCode = 0; // Predefinido para 'y' (lux)

      if (var.equalsIgnoreCase("u"))
        varCode = 1;
      else if (var.equalsIgnoreCase("p"))
        varCode = 2;
      else if (var.equalsIgnoreCase("o"))
        varCode = 3;
      else if (var.equalsIgnoreCase("a"))
        varCode = 4;
      else if (var.equalsIgnoreCase("f"))
        varCode = 5;
      else if (var.equalsIgnoreCase("r"))
        varCode = 6;
      else if (var.equalsIgnoreCase("y"))
        varCode = 0;
      else if (var.equalsIgnoreCase("v"))
        varCode = 7;
      else if (var.equalsIgnoreCase("d"))
        varCode = 8;
      else if (var.equalsIgnoreCase("t"))
        varCode = 9;
      else if (var.equalsIgnoreCase("V"))
        varCode = 10;
      else if (var.equalsIgnoreCase("F"))
        varCode = 11;
      else if (var.equalsIgnoreCase("E"))
        varCode = 12;

      if (sendControlCommand(targetNode, 12, varCode))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: Falha no reencaminhamento CAN");
      }
      return;
    }

    // Tratar localmente
    stopStream(var, targetNode);
    Serial.println("ack");
    return;
  }

  // ------------------- COMANDOS DE MÉTRICAS -------------------

  // Se o primeiro token for 'g' => "g <sub> <i>"
  else if (c0 == "g")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }
    String subCommand = tokens[1];
    String originalCase = tokens[1];
    subCommand.toLowerCase();

    uint8_t targetNode = tokens[2].toInt();
    String idx = tokens[2];

    // Verificar se precisamos de reencaminhar este comando
    if (shouldForwardCommand(targetNode))
    {
      // Mapear o comando get para um tipo de mensagem de consulta CAN (usando código 20-30)
      uint8_t queryType = 20; // Tipo de consulta predefinido

      if (originalCase == "V")
        queryType = 20; // Erro de visibilidade
      else if (originalCase == "F")
        queryType = 21; // Cintilação
      else if (originalCase == "E")
        queryType = 22; // Energia
      else if (subCommand == "u")
        queryType = 23; // Duty cycle
      else if (subCommand == "o")
        queryType = 24; // Ocupação
      else if (subCommand == "a")
        queryType = 25; // Anti-windup
      else if (subCommand == "f")
        queryType = 26; // Controlo por feedback
      else if (subCommand == "r")
        queryType = 27; // Referência de iluminância
      else if (subCommand == "y")
        queryType = 28; // Iluminância atual
      else if (subCommand == "p")
        queryType = 29; // Consumo de energia
      else if (subCommand == "t")
        queryType = 30; // Tempo decorrido
      else if (subCommand == "v")
        queryType = 31; // Tensão no LDR
      else if (subCommand == "d")
        queryType = 32; // Iluminância externa
      else
      {
        Serial.println("err: Consulta de variável remota não suportada");
        return;
      }

      // Enviar a consulta para o nó remoto
      if (sendControlCommand(targetNode, queryType, 0.0f))
      {
        Serial.println("Consulta enviada para o nó " + String(targetNode));

        // Aguardar resposta com tempo limite
        unsigned long timeout = millis() + 500; // 500ms de tempo limite
        bool responseReceived = false;

        while (millis() < timeout && !responseReceived)
        {
          can_frame frame;
          if (readCANMessage(&frame) == MCP2515::ERROR_OK)
          {
            // Analisar mensagem e verificar se é uma resposta do nosso alvo
            uint8_t msgType, destAddr, priority;
            parseCANId(frame.can_id, msgType, destAddr, priority);

            uint8_t senderNodeID = frame.data[0];

            if (msgType == CAN_TYPE_RESPONSE && senderNodeID == targetNode && (destAddr == nodeID || destAddr == CAN_ADDR_BROADCAST))
            {
              // Extrair o valor da resposta e mostrar
              float value = 0.0f;
              value = bytesToFloat(&frame.data[2]);

              // Formatar a resposta com base no tipo de consulta original
              if (originalCase == "V" || originalCase == "F" || originalCase == "E")
              {
                Serial.print(originalCase);
              }
              else
              {
                Serial.print(subCommand);
              }
              Serial.print(" ");
              Serial.print(idx);
              Serial.print(" ");

              // Formatar o valor com a precisão adequada
              if (subCommand == "u" || originalCase == "F" || originalCase == "E")
                Serial.println(value, 4);
              else if (originalCase == "V" || subCommand == "y" || subCommand == "p" || subCommand == "d")
                Serial.println(value, 2);
              else if (subCommand == "v")
                Serial.println(value, 3);
              else if (subCommand == "o" || subCommand == "a" || subCommand == "f" || subCommand == "t")
                Serial.println((int)value);
              else
                Serial.println(value);

              responseReceived = true;
            }
          }
          delay(5); // Pequeno atraso para evitar sobrecarregar o barramento CAN
        }

        if (!responseReceived)
        {
          Serial.println("err: Sem resposta do nó " + String(targetNode));
        }
      }
      else
      {
        Serial.println("err: Falha ao enviar consulta para o nó " + String(targetNode));
      }
      return;
    }

    // Processar localmente
    // Comandos de métricas
    // "g V <i>" => "V <i> <val>" (Métrica de erro de visibilidade)
    if (originalCase == "V")
    {
      float V = computeVisibilityErrorFromBuffer();
      Serial.print("V ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(V, 2);
      return;
    }
    // "g F <i>" => "F <i> <val>" (Métrica de cintilação)
    else if (originalCase == "F")
    {
      float F = computeFlickerFromBuffer();
      Serial.print("F ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(F, 4);
      return;
    }
    // "g E <i>" => "E <i> <val>" (Métrica de energia)
    else if (originalCase == "E")
    {
      float E = computeEnergyFromBuffer();
      Serial.print("E ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(E, 4);
      return;
    }

    // "g u <i>" => "u <i> <val>"
    if (subCommand == "u")
    {
      Serial.print("u ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(dutyCycle, 4);
      return;
    }
    // "g o <i>" => "o <i> <val>"
    else if (subCommand == "o")
    {
      int occVal = occupancy ? 1 : 0;
      Serial.print("o ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(occVal);
      return;
    }
    // "g a <i>" => "a <i> <val>"
    else if (subCommand == "a")
    {
      int awVal = antiWindup ? 1 : 0;
      Serial.print("a ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(awVal);
      return;
    }
    // "g f <i>" => "f <i> <val>"
    else if (subCommand == "f")
    {
      int fbVal = feedbackControl ? 1 : 0;
      Serial.print("f ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(fbVal);
      return;
    }
    // "g r <i>" => "r <i> <val>"
    else if (subCommand == "r")
    {
      Serial.print("r ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(refIlluminance, 4);
      return;
    }
    // "g y <i>" => "y <i> <val>"
    else if (subCommand == "y")
    {
      float lux = readLux();
      Serial.print("y ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(lux, 2);
      return;
    }
    // "g v <i>" => medir nível de tensão no LDR => "v <i> <val>"
    else if (subCommand == "v")
    {
      float vLdr = getVoltageAtLDR();
      Serial.print("v ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(vLdr, 3);
      return;
    }
    // "g d <i>" => iluminância externa => "d <i> <val>"
    else if (subCommand == "d")
    {
      float dVal = getExternalIlluminance();
      Serial.print("d ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(dVal, 2);
      return;
    }
    // "g p <i>" => potência instantânea => "p <i> <val>"
    else if (subCommand == "p")
    {
      float pVal = getPowerConsumption();
      Serial.print("p ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(pVal, 2);
      return;
    }
    // "g t <i>" => tempo decorrido => "t <i> <val>"
    else if (subCommand == "t")
    {
      unsigned long sec = getElapsedTime();
      Serial.print("t ");
      Serial.print(idx);
      Serial.print(" ");
      Serial.println(sec);
      return;
    }
    // "g b <x> <i>" => "b <x> <i> <val1>,<val2>..."
    else if (subCommand == "b")
    {
      if (numTokens < 4)
      {
        Serial.println("err");
        return;
      }
      String xVar = tokens[2];
      int iDesk = tokens[3].toInt();
      String bufferData = getLastMinuteBuffer(xVar, iDesk);
      Serial.print("b ");
      Serial.print(xVar);
      Serial.print(" ");
      Serial.print(iDesk);
      Serial.print(" ");
      Serial.println(bufferData);
      return;
    }

    else
    {
      Serial.println("err");
      return;
    }
  }
  // ------------------- COMANDOS CAN -------------------

  // Comandos CAN tratados se c0 == "c"
  // "c sd <destNode> <msgType> <value>" => Enviar uma mensagem CAN
  else if (c0 == "c" && tokens[1] == "sd")
  {
    if (numTokens < 5)
    {
      Serial.println("err");
      return;
    }

    uint8_t destNode = tokens[2].toInt();
    uint8_t msgType = tokens[3].toInt();
    float value = tokens[4].toFloat();

    bool success = false;

    if (msgType == 0)
    {                                                   // Mensagem de controlo
      success = sendControlCommand(destNode, 0, value); // Tipo 0 = setpoint
    }
    else if (msgType == 1)
    {                                                  // Leitura de sensor
      success = sendSensorReading(destNode, 0, value); // Tipo 0 = lux
    }
    else
    {
      Serial.println("err: Tipo de mensagem inválido");
      return;
    }

    if (success)
    {
      Serial.println("ack");
    }
    else
    {
      Serial.println("err: Falha no envio");
    }
    return;
  }
  // "c m <0|1>" => Ativar/desativar impressão de mensagens CAN recebidas
  else if (c0 == "c" && tokens[1] == "m")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    canMonitorEnabled = (tokens[2].toInt() == 1);

    Serial.print("Monitorização CAN ");
    Serial.println(canMonitorEnabled ? "ativada" : "desativada");
    Serial.println("ack");
    return;
  }
  // "c st" => Mostrar estatísticas de comunicação CAN
  else if (c0 == "c" && tokens[1] == "st")
  {
    uint32_t sent, received, errors;
    float avgLatency;
    getCANStats(sent, received, errors, avgLatency);

    Serial.println("CAN Statistics:");
    Serial.print("  Node ID: ");
    Serial.println(nodeID);
    Serial.print("  Messages sent: ");
    Serial.println(sent);
    Serial.print("  Messages received: ");
    Serial.println(received);
    Serial.print("  Errors: ");
    Serial.println(errors);
    Serial.print("  Avg. latency: ");
    Serial.print(avgLatency);
    Serial.println(" us");
    Serial.println("ack");
    return;
  }
  // "c r" => Reset CAN statistics
  else if (c0 == "c" && tokens[1] == "r")
  {
    resetCANStats();
    Serial.println("CAN statistics reset");
    Serial.println("ack");
    return;
  }
  // "c sc" => Scan for active nodes on the network
  else if (c0 == "c" && tokens[1] == "sc")
  {
    Serial.println("Scanning for active CAN nodes...");

    // We'll track which nodes respond
    bool nodeFound[64] = {false};
    int foundCount = 0;

    // Send ping messages to all possible node addresses
    for (uint8_t node = 1; node < 64; node++)
    {
      // Send a special ping message
      if (sendControlCommand(node, 3, 0))
      {
        // Give some time for node to respond
        delay(50);

        // Process any responses that came in
        for (int i = 0; i < 5; i++)
        {
          can_frame frame;
          if (readCANMessage(&frame) == MCP2515::ERROR_OK)
          {
            uint8_t msgType, srcAddr, priority;
            parseCANId(frame.can_id, msgType, srcAddr, priority);

            if (!nodeFound[srcAddr])
            {
              nodeFound[srcAddr] = true;
              foundCount++;
            }
          }
          delay(10);
        }
      }
    }

    // Now send a broadcast message to catch any we missed
    sendControlCommand(CAN_ADDR_BROADCAST, 3, 0);
    delay(200);

    // Process any additional responses
    for (int i = 0; i < 20; i++)
    {
      can_frame frame;
      if (readCANMessage(&frame) == MCP2515::ERROR_OK)
      {
        uint8_t msgType, srcAddr, priority;
        parseCANId(frame.can_id, msgType, srcAddr, priority);

        if (!nodeFound[srcAddr])
        {
          nodeFound[srcAddr] = true;
          foundCount++;
        }
      }
      delay(10);
    }

    // Display results
    Serial.print("Found ");
    Serial.print(foundCount);
    Serial.println(" active nodes:");

    for (uint8_t node = 1; node < 64; node++)
    {
      if (nodeFound[node])
      {
        Serial.print("  Node ");
        Serial.println(node);
      }
    }

    Serial.println("Network scan complete");
    Serial.println("ack");
    return;
  }
  // "c l <destNode> <count>" => Measure round-trip latency
  else if (c0 == "c" && tokens[1] == "l")
  {
    if (numTokens < 4)
    {
      Serial.println("err");
      return;
    }

    uint8_t destNode = tokens[2].toInt();
    int count = tokens[3].toInt();

    Serial.print("Measuring round-trip latency to node ");
    Serial.print(destNode);
    Serial.print(" (");
    Serial.print(count);
    Serial.println(" samples)");

    unsigned long totalLatency = 0;
    int successCount = 0;

    for (int i = 0; i < count; i++)
    {
      unsigned long startTime = micros();

      // Send echo request (using control message type 2)
      if (sendControlCommand(destNode, 2, startTime))
      {
        // Wait for response with timeout
        unsigned long timeout = millis() + 500; // 500ms timeout
        bool responseReceived = false;

        while (millis() < timeout && !responseReceived)
        {
          can_frame frame;
          if (readCANMessage(&frame) == MCP2515::ERROR_OK)
          {
            // Parse message and check if it's an echo response
            uint8_t msgType, srcAddr, priority;
            parseCANId(frame.can_id, msgType, srcAddr, priority);

            if (msgType == CAN_TYPE_RESPONSE && srcAddr == destNode)
            {
              unsigned long endTime = micros();
              unsigned long latency = endTime - startTime;
              totalLatency += latency;
              successCount++;
              responseReceived = true;

              Serial.print("Sample ");
              Serial.print(i + 1);
              Serial.print(": ");
              Serial.print(latency);
              Serial.println(" us");
            }
          }
        }

        if (!responseReceived)
        {
          Serial.print("Sample ");
          Serial.print(i + 1);
          Serial.println(": Timeout");
        }
      }
      else
      {
        Serial.print("Sample ");
        Serial.print(i + 1);
        Serial.println(": Send failed");
      }

      delay(100); // Wait between samples
    }

    Serial.println("Latency measurement complete");
    if (successCount > 0)
    {
      float avgLatency = (float)totalLatency / successCount;
      Serial.print("Average round-trip latency: ");
      Serial.print(avgLatency, 2);
      Serial.println(" us");
    }
    else
    {
      Serial.println("No successful measurements");
    }
    Serial.println("ack");
    return;
  }
  // "state <i> <state>" => set luminaire state (off/unoccupied/occupied)
  else if (c0 == "st")
  {
    if (numTokens < 3)
    {
      Serial.println("err");
      return;
    }

    uint8_t targetNode = tokens[1].toInt();
    String stateStr = tokens[2];
    stateStr.toLowerCase();

    // Convert state string to value for CAN transmission
    int stateVal = 0;
    if (stateStr == "off")
      stateVal = 0;
    else if (stateStr == "unoccupied")
      stateVal = 1;
    else if (stateStr == "occupied")
      stateVal = 2;
    else
    {
      Serial.println("err");
      return;
    }

    // Check if we need to forward this command
    if (shouldForwardCommand(targetNode))
    {
      // Forward to specific node - control type 13 = luminaire state
      if (sendControlCommand(targetNode, 13, stateVal))
      {
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN forwarding failed");
      }
      return;
    }

    // Handle broadcast case (targetNode = 0)
    if (targetNode == 0)
    {
      if (sendControlCommand(CAN_ADDR_BROADCAST, 13, stateVal))
      {
        // Also apply locally since broadcast includes this node
        if (stateStr == "off")
          changeState(STATE_OFF);
        else if (stateStr == "unoccupied")
          changeState(STATE_UNOCCUPIED);
        else if (stateStr == "occupied")
          changeState(STATE_OCCUPIED);
        Serial.println("ack");
      }
      else
      {
        Serial.println("err: CAN broadcast failed");
      }
      return;
    }

    // Apply locally
    if (stateStr == "off")
      changeState(STATE_OFF);
    else if (stateStr == "unoccupied")
      changeState(STATE_UNOCCUPIED);
    else if (stateStr == "occupied")
      changeState(STATE_OCCUPIED);
    Serial.println("ack");
    return;
  }
  Serial.println("ack");
  return;
}

// Processar comandos seriais recebidos
void processSerialCommands()
{
  if (Serial.available() > 0)
  {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0)
    {
      processCommandLine(input);
    }
  }
}



// Utilizar a mesma configuração de pinos do exemplo funcional
const int CAN_CS_PIN = 17;
const int CAN_MOSI_PIN = 19;
const int CAN_MISO_PIN = 16;
const int CAN_SCK_PIN = 18;

// Criar a instância MCP2515 com a configuração de pinos correspondente
MCP2515 can0(spi0, CAN_CS_PIN, CAN_MOSI_PIN, CAN_MISO_PIN, CAN_SCK_PIN, 10000000);


typedef void (*CANMessageCallback)(const can_frame &msg);

// Variável estática para armazenar o callback registado
static CANMessageCallback messageCallback = nullptr;

// Rastreamento de estatísticas
static unsigned long lastLatencyMeasure = 0;
static unsigned long totalLatency = 0;
static uint32_t latencySamples = 0;
static uint32_t msgSent = 0;
static uint32_t msgReceived = 0;
static uint32_t msgErrors = 0;

void initCANComm()
{
  // Inicializar SPI
  SPI.begin();

  // Reinicialização simples - como no exemplo funcional
  can0.reset();

  // Utilizar 1000KBPS como no exemplo (não 125KBPS)
  can0.setBitrate(CAN_1000KBPS);

  // Definir modo normal
  can0.setNormalMode();

  Serial.println("CANComm: CAN inicializado em modo normal");

  // Gerar um ID de nó único a partir dos últimos 6 bits do ID único da placa
  pico_unique_board_id_t board_id;
  pico_get_unique_board_id(&board_id);
  nodeID = board_id.id[7] & 0x3F; // Utilizar os últimos 6 bits para o ID do nó (1-63)
  if (nodeID == 0)
    nodeID = 1; // Evitar endereço de difusão

  Serial.print("CANComm: ID de nó atribuído: ");
  Serial.println(nodeID);
}

// Construir um ID de mensagem CAN a partir dos componentes
uint32_t buildCANId(uint8_t msgType, uint8_t destAddr, uint8_t priority)
{
  return ((uint32_t)msgType << 8) | ((uint32_t)destAddr << 2) | priority;
}

// Extrair componentes de um ID CAN
void parseCANId(uint32_t canId, uint8_t &msgType, uint8_t &destAddr, uint8_t &priority)
{
  msgType = (canId >> 8) & 0x07;
  destAddr = (canId >> 2) & 0x3F;
  priority = canId & 0x03;
}

// Converter float para bytes para transmissão CAN (little-endian)
void floatToBytes(float value, uint8_t *bytes)
{
  memcpy(bytes, &value, 4);
}

// Converter bytes para float (little-endian)
float bytesToFloat(const uint8_t *bytes)
{
  float value;
  memcpy(&value, bytes, 4);
  return value;
}

// Enviar uma leitura de sensor para outro nó ou difusão
bool sendSensorReading(uint8_t destAddr, uint8_t sensorType, float value)
{
  can_frame frame;

  // Construir o ID CAN
  frame.can_id = buildCANId(CAN_TYPE_SENSOR, destAddr, CAN_PRIO_NORMAL);

  // Definir comprimento dos dados
  frame.can_dlc = 8;

  // Carga útil: nó de origem, tipo de sensor, valor como float, marca temporal
  frame.data[0] = nodeID;     // Nó de origem
  frame.data[1] = sensorType; // Tipo de sensor (0 = lux, 1 = duty, etc)

  // Valor float (4 bytes)
  floatToBytes(value, &frame.data[2]);

  // Marca temporal - contador de milissegundos de 16 bits
  uint16_t timestamp = (uint16_t)(millis() & 0xFFFF);
  frame.data[6] = timestamp & 0xFF;
  frame.data[7] = (timestamp >> 8) & 0xFF;

  // Enviar a mensagem
  MCP2515::ERROR result = sendCANMessage(frame);

  if (result == MCP2515::ERROR_OK)
  {
    msgSent++;
    return true;
  }
  else
  {
    msgErrors++;
    return false;
  }
}

// Enviar um comando de controlo para outro nó
bool sendControlCommand(uint8_t destAddr, uint8_t controlType, float value)
{
  can_frame frame;

  // Construir o ID CAN
  frame.can_id = buildCANId(CAN_TYPE_CONTROL, destAddr, CAN_PRIO_HIGH);

  // Definir comprimento dos dados
  frame.can_dlc = 8;

  // Carga útil: nó de origem, tipo de controlo, valor como float, número de sequência
  frame.data[0] = nodeID;      // Nó de origem
  frame.data[1] = controlType; // Tipo de controlo

  // Valor float (4 bytes)
  floatToBytes(value, &frame.data[2]);

  // Número de sequência (para detetar mensagens perdidas)
  static uint16_t seqNum = 0;
  frame.data[6] = seqNum & 0xFF;
  frame.data[7] = (seqNum >> 8) & 0xFF;
  seqNum++;

  // Enviar a mensagem
  MCP2515::ERROR result = sendCANMessage(frame);

  if (result == MCP2515::ERROR_OK)
  {
    msgSent++;
    return true;
  }
  else
  {
    msgErrors++;
    return false;
  }
}

// Enviar uma resposta a uma mensagem de consulta
bool sendQueryResponse(uint8_t destNode, float value)
{
  can_frame frame;
  frame.can_id = buildCANId(CAN_TYPE_RESPONSE, destNode, CAN_PRIO_NORMAL);
  frame.can_dlc = 8; // Utilizar 8 bytes como todas as outras mensagens

  // Incluir o ID do nó emissor e um tipo de resposta
  frame.data[0] = nodeID; // ID do nó que está a responder
  frame.data[1] = 2;      // Tipo 2 = resposta de consulta

  // Colocar o valor float nos bytes 2-5
  floatToBytes(value, &frame.data[2]);

  // Bytes 6-7 podem ser utilizados para sequência ou deixados como zero
  frame.data[6] = 0;
  frame.data[7] = 0;

  // Enviar a mensagem
  MCP2515::ERROR result = sendCANMessage(frame);

  Serial.print("DEBUG: Enviada resposta de consulta para o nó ");
  Serial.print(destNode);
  Serial.print(", valor: ");
  Serial.println(value);

  if (result == MCP2515::ERROR_OK)
  {
    msgSent++;
    return true;
  }
  else
  {
    msgErrors++;
    return false;
  }
}

// Enviar uma mensagem de heartbeat para indicar que o nó está ativo
bool sendHeartbeat()
{
  can_frame frame;

  // Construir o ID CAN
  frame.can_id = buildCANId(CAN_TYPE_HEARTBEAT, CAN_ADDR_BROADCAST, CAN_PRIO_LOW);

  // Definir comprimento dos dados
  frame.can_dlc = 6;

  // Carga útil: nó de origem, flags de estado, tempo de atividade
  frame.data[0] = nodeID;

  // Flags de estado: bit0=feedback, bit1=ocupação
  uint8_t statusFlags = 0;
  if (feedbackControl)
    statusFlags |= 0x01;
  if (occupancy)
    statusFlags |= 0x02;
  frame.data[1] = statusFlags;

  // Tempo de atividade do nó em segundos
  uint32_t uptime = getElapsedTime();
  frame.data[2] = uptime & 0xFF;
  frame.data[3] = (uptime >> 8) & 0xFF;
  frame.data[4] = (uptime >> 16) & 0xFF;
  frame.data[5] = (uptime >> 24) & 0xFF;

  // Enviar a mensagem
  MCP2515::ERROR result = sendCANMessage(frame);

  if (result == MCP2515::ERROR_OK)
  {
    msgSent++;
    return true;
  }
  else
  {
    msgErrors++;
    return false;
  }
}

void processIncomingMessage(const can_frame &msg)
{
  // Analisar o ID CAN
  uint8_t msgType, destAddr, priority;
  parseCANId(msg.can_id, msgType, destAddr, priority);

  if (canMonitorEnabled)
  {
    Serial.print("DEBUG: Recebida mensagem CAN, tipo ");
    Serial.print(msgType);
    Serial.print(", origem ");
    Serial.print(msg.data[0]);
  }

  // Verificar se esta mensagem é para nós (ou difusão)
  if (destAddr != nodeID && destAddr != CAN_ADDR_BROADCAST)
  {
    return; // Não é para nós
  }

  // Mensagem é para nós, processar com base no tipo
  uint8_t sourceNode = msg.data[0];

  switch (msgType)
  {
  case CAN_TYPE_SENSOR:
  {
    uint8_t sensorType = msg.data[1];
    float value = bytesToFloat(&msg.data[2]);
    uint16_t timestamp = ((uint16_t)msg.data[7] << 8) | msg.data[6];

    if (canMonitorEnabled)
    {
      Serial.print("CAN: Nó ");
      Serial.print(sourceNode);
      Serial.print(" sensor ");
      Serial.print(sensorType);
      Serial.print(" = ");
      Serial.print(value);
      Serial.print(" (ts: ");
      Serial.print(timestamp);
      Serial.println(")");
    }
    break;
  }
  case CAN_TYPE_HEARTBEAT:
  {
    uint8_t statusFlags = msg.data[1];
    uint32_t uptime = ((uint32_t)msg.data[5] << 24) |
                      ((uint32_t)msg.data[4] << 16) |
                      ((uint32_t)msg.data[3] << 8) |
                      msg.data[2];

    if (canMonitorEnabled)
    {
      Serial.print("CAN: Nó ");
      Serial.print(sourceNode);
      Serial.print(" heartbeat, tempo de atividade ");
      Serial.print(uptime);
      Serial.print("s, flags: ");
      Serial.println(statusFlags, BIN);
    }
    break;
  }
  case CAN_TYPE_CONTROL:
  {
    uint8_t controlType = msg.data[1];
    float value = bytesToFloat(&msg.data[2]);
    uint16_t sequence = ((uint16_t)msg.data[7] << 8) | msg.data[6];

    if (canMonitorEnabled)
    {
      Serial.print("CAN: Nó ");
      Serial.print(sourceNode);
      Serial.print(" controlo ");
      Serial.print(controlType);
      Serial.print(" = ");
      Serial.print(value);
      Serial.print(" (seq: ");
      Serial.print(sequence);
      Serial.println(")");
    }

    // Lidar com comandos de controlo
    if (controlType == 0)
    { // Setpoint
      setpointLux = value;
      // Enviar confirmação?
    }
    else if (controlType == 2)
    { // Pedido de eco
      // Enviar resposta de eco - utilizar o valor recebido
      can_frame response;
      response.can_id = buildCANId(CAN_TYPE_RESPONSE, sourceNode, CAN_PRIO_HIGH);
      response.can_dlc = 8;
      response.data[0] = nodeID;
      response.data[1] = 0;                   // Tipo de resposta 0 = eco
      floatToBytes(value, &response.data[2]); // Devolver o mesmo valor
      response.data[6] = msg.data[6];         // Copiar números de sequência
      response.data[7] = msg.data[7];
      sendCANMessage(response);
    }
    else if (controlType == 3)
    { // Ping/descoberta
      // Enviar uma resposta para nos identificarmos
      can_frame response;
      response.can_id = buildCANId(CAN_TYPE_RESPONSE, sourceNode, CAN_PRIO_NORMAL);
      response.can_dlc = 8;
      response.data[0] = nodeID;
      response.data[1] = 1; // Tipo de resposta 1 = descoberta
      floatToBytes(0, &response.data[2]);
      response.data[6] = 0;
      response.data[7] = 0;
      sendCANMessage(response);
    }
    else if (controlType == 4) { // Definir duty cycle
      setLEDDutyCycle(value);
      if (canMonitorEnabled) {
        Serial.print("CAN: A definir duty cycle para ");
        Serial.println(value);
      }
    }
    else if (controlType == 5) { // Definir percentagem do LED
      setLEDPercentage(value);
      if (canMonitorEnabled) {
        Serial.print("CAN: A definir percentagem do LED para ");
        Serial.println(value);
      }
    }
    else if (controlType == 6) { // Definir potência do LED em watts
      setLEDPower(value);
      if (canMonitorEnabled) {
        Serial.print("CAN: A definir potência do LED para ");
        Serial.println(value);
      }
    }
    else if (controlType == 7) { // Definir ocupação
      occupancy = (value != 0.0f);
      if (canMonitorEnabled) {
        Serial.print("CAN: A definir ocupação para ");
        Serial.println(occupancy ? "verdadeiro" : "falso");
      }
    }
    else if (controlType == 8) { // Definir anti-windup
      antiWindup = (value != 0.0f);
      if (canMonitorEnabled) {
        Serial.print("CAN: A definir anti-windup para ");
        Serial.println(antiWindup ? "verdadeiro" : "falso");
      }
    }
    else if (controlType == 9) { // Definir controlo por feedback
      feedbackControl = (value != 0.0f);
      if (canMonitorEnabled) {
        Serial.print("CAN: A definir controlo por feedback para ");
        Serial.println(feedbackControl ? "verdadeiro" : "falso");
      }
    }
    else if (controlType == 10) { // Iluminância de referência
      refIlluminance = value;
      setpointLux = value;
      
      if (canMonitorEnabled) {
        Serial.print("CAN: A definir iluminância de referência para ");
        Serial.println(value);
      }
    }
    else if (controlType == 11)
    { // Início de stream
      // Extrair tipo de variável do valor
      int varCode = (int)value;
      String var = "y"; // Predefinição

      if (varCode == 1)
        var = "u";
      else if (varCode == 2)
        var = "p";
      else if (varCode == 3)
        var = "o";
      else if (varCode == 4)
        var = "a";
      else if (varCode == 5)
        var = "f";
      else if (varCode == 6)
        var = "r";
      else if (varCode == 7)
        var = "v";
      else if (varCode == 8)
        var = "d";
      else if (varCode == 9)
        var = "t";
      else if (varCode == 10)
        var = "V";
      else if (varCode == 11)
        var = "F";
      else if (varCode == 12)
        var = "E";

      startStream(var, sourceNode);
    }
    else if (controlType == 12)
    { // Paragem de stream
      // Extrair tipo de variável do valor
      int varCode = (int)value;
      String var = "y"; // Predefinição

      if (varCode == 1)
        var = "u";
      else if (varCode == 2)
        var = "p";
      else if (varCode == 3)
        var = "o";
      else if (varCode == 4)
        var = "a";
      else if (varCode == 5)
        var = "f";
      else if (varCode == 6)
        var = "r";
      else if (varCode == 7)
        var = "v";
      else if (varCode == 8)
        var = "d";
      else if (varCode == 9)
        var = "t";
      else if (varCode == 10)
        var = "V";
      else if (varCode == 11)
        var = "F";
      else if (varCode == 12)
        var = "E";

      stopStream(var, sourceNode);
    }
    else if (controlType == 13)
    { // Estado do luminária
      int stateVal = (int)value;
      if (stateVal == 0)
        changeState(STATE_OFF);
      else if (stateVal == 1)
        changeState(STATE_UNOCCUPIED);
      else if (stateVal == 2)
        changeState(STATE_OCCUPIED);
    }
    else if (controlType >= 20 && controlType <= 32)
    {
      // Esta é uma mensagem de consulta, enviar de volta uma resposta
      float responseValue = 0.0f;

      // Obter o valor solicitado
      switch (controlType)
      {
      case 20:
        responseValue = computeVisibilityErrorFromBuffer();
        break;
      case 21:
        responseValue = computeFlickerFromBuffer();
        break;
      case 22:
        responseValue = computeEnergyFromBuffer();
        break;
      case 23:
        responseValue = dutyCycle;
        break;
      case 24:
        responseValue = occupancy ? 1.0f : 0.0f;
        break;
      case 25:
        responseValue = antiWindup ? 1.0f : 0.0f;
        break;
      case 26:
        responseValue = feedbackControl ? 1.0f : 0.0f;
        break;
      case 27:
        responseValue = refIlluminance;
        break;
      case 28:
        Serial.println("Consulta para iluminância atual");
        responseValue = readLux();
        break;
      case 29:
        responseValue = getPowerConsumption();
        break;
      case 30:
        responseValue = getElapsedTime();
        break;
      case 31:
        responseValue = getVoltageAtLDR();
        break;
      case 32:
        responseValue = getExternalIlluminance();
        break;
      default:
        return; // Tipo de consulta desconhecido
      }
      // Enviar uma mensagem de resposta com o valor
      sendQueryResponse(sourceNode, responseValue);
    }
    break;
  }
  }
}

void canCommLoop()
{
  // Verificar mensagens recebidas
  can_frame msg;
  if (can0.readMessage(&msg) == MCP2515::ERROR_OK)
  {
    // Registar estatísticas
    msgReceived++;

    // Processar a mensagem
    processIncomingMessage(msg);

    // Se um callback foi registado, chamá-lo
    if (messageCallback)
    {
      messageCallback(msg);
    }
  }
}

MCP2515::ERROR sendCANMessage(const can_frame &frame)
{
  // Registar tempo de envio para medições de latência
  lastLatencyMeasure = micros();

  // Enviar a mensagem
  MCP2515::ERROR err = can0.sendMessage(&frame);

  // Atualizar latência se bem-sucedido (assume que o hardware enviou a mensagem)
  if (err == MCP2515::ERROR_OK)
  {
    unsigned long latency = micros() - lastLatencyMeasure;
    totalLatency += latency;
    latencySamples++;
  }

  return err;
}

MCP2515::ERROR readCANMessage(struct can_frame *frame)
{
  return can0.readMessage(frame);
}

void setCANMessageCallback(CANMessageCallback callback)
{
  messageCallback = callback;
}

// Obter estatísticas de comunicação
void getCANStats(uint32_t &sent, uint32_t &received, uint32_t &errors, float &avgLatency)
{
  sent = msgSent;
  received = msgReceived;
  errors = msgErrors;
  avgLatency = latencySamples > 0 ? (float)totalLatency / latencySamples : 0.0f;
}

// Reiniciar estatísticas de comunicação
void resetCANStats()
{
  msgSent = 0;
  msgReceived = 0;
  msgErrors = 0;
  totalLatency = 0;
  latencySamples = 0;
}

// ===================== PROGRAMA PRINCIPAL =====================

// No seu loop, adicione uma mensagem CAN periódica:
unsigned long lastCANSend = 0;

void setup()
{
  Serial.begin(115200);

  // ID da placa para depuração
  pico_unique_board_id_t board_id;
  pico_get_unique_board_id(&board_id);

  // Configurar ADC e PWM
  analogReadResolution(12);
  analogWriteFreq(30000);
  analogWriteRange(PWM_MAX);

  // Inicializar driver LED com o pino LED
  initLEDDriver(LED_PIN);

  // Executar teste do LED
  testLED();

  // Inicializar armazenamento de buffer circular para logging
  initStorage();

  Serial.println("Sistema de Controlo Distribuído com CAN-BUS e Interface de Comandos");

  // Inicializar comunicação CAN
  initCANComm();

  // Sincronizar setpoint inicial e referência
  setpointLux = 15.0;
  refIlluminance = setpointLux;

  // Imprimir cabeçalho para o Serial Plotter (duas colunas numéricas: Lux Medido e Setpoint)
  Serial.println("LuxMedido\tSetpoint");
}

void loop()
{
  // (A) Processar comandos seriais recebidos
  processSerialCommands();

  // (B) Gerir qualquer streaming ativo
  handleStreaming();

  // (C) Ler dados do sensor
  float lux = readLux();

  // (D) Adaptar às condições de luz externa
  adaptToExternalLight();

  // (E) Coordenar com vizinhos para otimização de energia
  if (luminaireState != STATE_OFF)
  {
    coordinateWithNeighbors();
  }

  // (F) Cálculo e aplicação da ação de controlo
  if (luminaireState == STATE_OFF)
  {
    // Desligar a luz quando no estado OFF
    setLEDDutyCycle(0.0);
  }
  else if (feedbackControl)
  {
    // Usar controlo PID no modo de feedback
    float u = pid.compute(setpointLux, lux);
    setLEDPWMValue((int)u);
  }
  else
  {
    // Controlo direto do duty cycle no modo manual
    setLEDDutyCycle(dutyCycle);
  }

  // (G) Registar a amostra atual no buffer circular
  logData(millis(), lux, dutyCycle);

  // (H) Processar mensagens CAN (não bloqueante)
  canCommLoop();

  // (I) Tarefas CAN periódicas
  unsigned long now = millis();

  // Enviar dados do sensor se o modo periódico estiver ativado
  if (periodicCANEnabled && (now - lastCANSend >= 1000))
  {
    lastCANSend = now;

    // Enviar leitura de iluminância (broadcast)
    sendSensorReading(CAN_ADDR_BROADCAST, 0, lux);

    // Enviar duty cycle (broadcast)
    sendSensorReading(CAN_ADDR_BROADCAST, 1, dutyCycle);

    // Enviar informação do estado (broadcast)
    sendSensorReading(CAN_ADDR_BROADCAST, 2, (float)luminaireState);

    // Enviar estimativa de luz externa (broadcast)
    sendSensorReading(CAN_ADDR_BROADCAST, 3, getExternalIlluminance());
  }

  // Plotagem de depuração se ativada
  if (DEBUG_MODE && DEBUG_PLOTTING)
  {
    Serial.print(lux, 2);
    Serial.print("\t");
    Serial.print(setpointLux, 2);
    Serial.print("\t");
    Serial.print(30.0, 2); // Limite superior
    Serial.print("\t");
    Serial.println(0.0, 2); // Limite inferior
  }

  // Aguardar pelo próximo ciclo de controlo
  delay((int)(pid.getSamplingTime() * 1000));
}