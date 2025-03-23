/**
 * DataLogger.cpp - Implementação do sistema de logging e métricas
 *
 * Este ficheiro implementa as funcionalidades de armazenamento e análise de dados:
 * - Gestão de um buffer circular para armazenamento de medições
 * - Cálculo de métricas de desempenho (energia, erro de visibilidade, cintilação)
 * - Streaming de variáveis em tempo real
 * - Exportação de dados históricos
 * - Análise estatística do comportamento do sistema
 */

//============================================================================
// FICHEIROS INCLUÍDOS
//============================================================================

#include "DataLogger.h"
#include "Configuration.h"
#include <Arduino.h>

//============================================================================
// VARIÁVEIS GLOBAIS
//============================================================================

LogEntry logBuffer[LOG_SIZE];
int logIndex = 0;
bool bufferFull = false;

// Para Cálculo de Energia
const float Pmax = 1.0;   // Potência máxima do LED em Watts
extern float setpointLux; // Declarado em main.ino
extern float dutyCycle;   // Current duty cycle [0..1]

// Streaming settings
static bool streamEnabled = false;
static String streamVariable = "";
static int streamIndex = 0;
static unsigned long lastStreamTime = 0;

// External function declarations
extern float readLux();
extern float getPowerConsumption();

//============================================================================
// FUNÇÕES DE STREAMING DE DADOS
//============================================================================

/**
 * Inicia o streaming de uma variável específica
 * @param var Nome da variável a ser monitorada
 * @param index Índice do nó para identificação
 */
void startStream(const String &var, int index)
{
    streamEnabled = true;
    streamVariable = var;
    streamIndex = index;
    Serial.println("ack");
}

/**
 * Interrompe o streaming de uma variável
 * @param var Nome da variável que estava sendo monitorada
 * @param index Índice do nó associado
 */
void stopStream(const String &var, int index)
{
    streamEnabled = false;
    streamVariable = ""; // Limpar a variável
    Serial.print("Streaming parado para ");
    Serial.print(var);
    Serial.print(" no nó ");
    Serial.println(index);
}

/**
 * Processa o envio periódico de dados em streaming
 * Deve ser chamado ciclicamente no loop principal
 */
void handleStreaming()
{
    if (!streamEnabled || (millis() - lastStreamTime < STREAM_INTERVAL))
    {
        return; // Sem streaming ou ainda não é tempo de fazer streaming
    }

    unsigned long currentTime = millis();
    lastStreamTime = currentTime;
    String var = streamVariable;
    int index = streamIndex;

    if (var.equalsIgnoreCase("y"))
    {
        float lux = readLux();
        Serial.print("s "); // Adicionar prefixo "s"
        Serial.print(var);
        Serial.print(" ");
        Serial.print(index);
        Serial.print(" ");
        Serial.print(lux, 2);
        Serial.print(" ");
        Serial.println(currentTime); // Adicionar timestamp
    }
    else if (var.equalsIgnoreCase("u"))
    {
        Serial.print("s "); // Adicionar prefixo "s"
        Serial.print(var);
        Serial.print(" ");
        Serial.print(index);
        Serial.print(" ");
        Serial.print(dutyCycle, 4);
        Serial.print(" ");
        Serial.println(currentTime); // Adicionar timestamp
    }
    else if (var.equalsIgnoreCase("p"))
    {
        float power = getPowerConsumption();
        Serial.print("s "); // Adicionar prefixo "s"
        Serial.print(var);
        Serial.print(" ");
        Serial.print(index);
        Serial.print(" ");
        Serial.print(power, 2);
        Serial.print(" ");
        Serial.println(currentTime); // Adicionar timestamp
    }
}

/**
 * Obtém dados históricos do buffer para visualização
 * @param var Nome da variável a ser extraída
 * @param index Índice do nó (não utilizado nesta implementação)
 * @return String formatada com valores separados por vírgula
 */
String getLastMinuteBuffer(const String &var, int index)
{
    String result = "";
    int count = getLogCount();
    if (count == 0)
        return result;

    LogEntry *buffer = getLogBuffer();
    int startIndex = isBufferFull() ? getCurrentIndex() : 0;

    // Número máximo de amostras a retornar (para evitar overflow no buffer)
    const int MAX_SAMPLES = 60;
    int sampleCount = min(count, MAX_SAMPLES);

    // Calcular passo para obter amostras distribuídas uniformemente
    int step = count > MAX_SAMPLES ? count / MAX_SAMPLES : 1;

    for (int i = 0; i < count; i += step)
    {
        int realIndex = (startIndex + i) % LOG_SIZE;

        if (var.equalsIgnoreCase("y"))
        {
            // Para valores de iluminação
            result += String(buffer[realIndex].lux, 1);
        }
        else if (var.equalsIgnoreCase("u"))
        {
            // Para valores de duty cycle
            result += String(buffer[realIndex].duty, 3);
        }

        if (i + step < count)
        {
            result += ",";
        }
    }

    return result;
}

//============================================================================
// FUNÇÕES DO CIRCULAR BUFFER
//============================================================================

/**
 * Inicializa o sistema de armazenamento
 */
void initStorage()
{
    logIndex = 0;
    bufferFull = false;
}

/**
 * Armazena uma nova entrada de log no buffer circular
 * @param timestamp Timestamp da medição em millisegundos
 * @param lux Valor de iluminação medido
 * @param duty Duty cycle atual do LED [0..1]
 */
void logData(unsigned long timestamp, float lux, float duty)
{
    // Adicionar variável externa
    extern float setpointLux;
    
    // Variáveis estáticas para cálculo de jitter e flicker
    static unsigned long lastSampleMicros = 0;
    static float lastDuty = 0.0f;
    
    // Obter o tempo atual em microssegundos para cálculo de jitter
    unsigned long currentMicros = micros();
    
    // Calcular jitter (diferença em relação ao período nominal de 10ms = 10000µs)
    float jitterValue = 0.0f;
    if (lastSampleMicros > 0) {
        jitterValue = (float)((currentMicros - lastSampleMicros) - 10000.0f);
    }
    
    // Calcular flicker (variação absoluta do duty cycle)
    float flickerValue = fabs(duty - lastDuty);
    
    // Armazenar valores no buffer
    logBuffer[logIndex].timestamp = timestamp;
    logBuffer[logIndex].lux = lux;
    logBuffer[logIndex].duty = duty;
    logBuffer[logIndex].setpoint = setpointLux; // Armazenar setpoint atual também
    logBuffer[logIndex].flicker = flickerValue;
    logBuffer[logIndex].jitter = jitterValue;
    
    // Atualizar variáveis para próxima amostra
    lastSampleMicros = currentMicros;
    lastDuty = duty;
    
    // Avançar índice do buffer circular
    logIndex++;
    if (logIndex >= LOG_SIZE)
    {
        logIndex = 0;
        bufferFull = true;
    }
}

/**
 * Exporta todo o conteúdo do buffer para a interface Serial
 * Formato: timestamp_ms,rawLux,duty
 */
void dumpBufferToSerial()
{
    Serial.println("timestamp_ms,rawLux,duty");
    int count = bufferFull ? LOG_SIZE : logIndex;
    int startIndex = bufferFull ? logIndex : 0;

    for (int i = 0; i < count; i++)
    {
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

    Serial.println("Fim da Exportação de Dados dos Buffer\n");
}

/**
 * Exporta amostras do buffer para a interface Serial
 * Formato: Time, MeasuredLux, DutyCycle, SetpointLux
 */
void dumpSampledBufferToSerial()
{
    int count = bufferFull ? LOG_SIZE : logIndex;
    if (count <= 0)
    {
        Serial.println("Buffer vazio.");
        return;
    }

    int startIndex = bufferFull ? logIndex : 0;
    int step = max(1, count / 1000); // Sample at least every 'step' entries to get ~1000 points
    int numSamples = min(1000, count / step);

    // Print timestamps
    Serial.print("Time: ");
    for (int i = 0; i < numSamples; i++)
    {
        int idx = (startIndex + i * step) % LOG_SIZE;
        Serial.print(logBuffer[idx].timestamp);
        if (i < numSamples - 1)
            Serial.print(",");
    }
    Serial.println();

    // Print measured lux values
    Serial.print("MeasuredLux: ");
    for (int i = 0; i < numSamples; i++)
    {
        int idx = (startIndex + i * step) % LOG_SIZE;
        Serial.print(logBuffer[idx].lux, 2);
        if (i < numSamples - 1)
            Serial.print(",");
    }
    Serial.println();

    // Print duty cycle values
    Serial.print("DutyCycle: ");
    for (int i = 0; i < numSamples; i++)
    {
        int idx = (startIndex + i * step) % LOG_SIZE;
        Serial.print(logBuffer[idx].duty, 4);
        if (i < numSamples - 1)
            Serial.print(",");
    }
    Serial.println();

    // Print setpoint values
    Serial.print("SetpointLux: ");
    for (int i = 0; i < numSamples; i++)
    {
        int idx = (startIndex + i * step) % LOG_SIZE;
        Serial.print(logBuffer[idx].setpoint, 2);
        if (i < numSamples - 1)
            Serial.print(",");
    }
    Serial.println();
}

/**
 * Obtém ponteiro para o buffer de log
 * @return Ponteiro para o array de entradas de log
 */
LogEntry *getLogBuffer()
{
    return logBuffer;
}

/**
 * Obtém o número de entradas válidas no buffer
 * @return Número de entradas armazenadas
 */
int getLogCount()
{
    return bufferFull ? LOG_SIZE : logIndex;
}

/**
 * Verifica se o buffer está completo
 * @return true se o buffer já deu uma volta completa
 */
bool isBufferFull()
{
    return bufferFull;
}

/**
 * Obtém o índice atual de escrita no buffer
 * @return Índice onde a próxima entrada será escrita
 */
int getCurrentIndex()
{
    return logIndex;
}

//============================================================================
// FUNÇÕES DE ANÁLISE E MÉTRICAS
//============================================================================

/**
 * Calcula e imprime métricas de desempenho do sistema
 * com base nos dados armazenados no buffer circular
 */
void computeAndPrintMetrics()
{
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

/**
 * Calcula o consumo de energia com base nas medições armazenadas
 * @return Energia consumida estimada em Joules
 */
float computeEnergyFromBuffer()
{
    int count = getLogCount();
    if (count < 2)
        return 0.0;

    LogEntry *logBuffer = getLogBuffer();
    int startIndex = isBufferFull() ? getCurrentIndex() : 0;
    unsigned long prevTime = 0;
    float prevDuty = 0.0;
    bool first = true;
    float totalE = 0.0;

    for (int i = 0; i < count; i++)
    {
        int realIndex = (startIndex + i) % LOG_SIZE;
        unsigned long t = logBuffer[realIndex].timestamp;
        float d = logBuffer[realIndex].duty;

        if (!first)
        {
            float dt = (t - prevTime) / 1000.0;
            totalE += (Pmax * prevDuty * dt);
        }
        else
        {
            first = false;
        }

        prevTime = t;
        prevDuty = d;
    }

    return totalE;
}

/**
 * Calcula o erro médio de visibilidade (quando iluminação < setpoint)
 * @return Erro médio de visibilidade em lux
 */
float computeVisibilityErrorFromBuffer()
{
    int count = getLogCount();
    if (count == 0)
        return 0.0;

    LogEntry *logBuffer = getLogBuffer();
    int startIndex = isBufferFull() ? getCurrentIndex() : 0;
    float totalErr = 0.0;
    int sampleCount = 0;

    for (int i = 0; i < count; i++)
    {
        int realIndex = (startIndex + i) % LOG_SIZE;
        float measuredLux = logBuffer[realIndex].lux;

        if (measuredLux < setpointLux)
        {
            totalErr += (setpointLux - measuredLux);
        }

        sampleCount++;
    }

    if (sampleCount == 0)
        return 0.0;
    return (totalErr / sampleCount);
}

/**
 * Calcula a métrica de cintilação (flicker) com base na
 * frequência e magnitude das inversões de direção do duty cycle
 * @return Valor da métrica de cintilação
 */
float computeFlickerFromBuffer()
{
    int count = getLogCount();
    if (count < 3)
        return 0.0;

    LogEntry *logBuffer = getLogBuffer();
    int startIndex = isBufferFull() ? getCurrentIndex() : 0;
    float flickerSum = 0.0;
    int flickerCount = 0;

    bool first = true, second = false;
    float d0, d1;

    for (int i = 0; i < count; i++)
    {
        int realIndex = (startIndex + i) % LOG_SIZE;
        float d2 = logBuffer[realIndex].duty;

        if (first)
        {
            d0 = d2;
            first = false;
            second = false;
        }
        else if (!second)
        {
            d1 = d2;
            second = true;
        }
        else
        {
            float diff1 = d1 - d0;
            float diff2 = d2 - d1;

            if (diff1 * diff2 < 0.0)
            {
                flickerSum += (fabs(diff1) + fabs(diff2));
                flickerCount++;
            }

            d0 = d1;
            d1 = d2;
        }
    }

    if (flickerCount == 0)
        return 0.0;
    return (flickerSum / flickerCount);
}
