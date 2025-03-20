#include "DataLogger.h"
#include "Configuration.h"
#include <Arduino.h>

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

// ===================== FUNÇÕES DE STREAMING =====================

void startStream(const String &var, int index)
{
    streamEnabled = true;
    streamVariable = var;
    streamIndex = index;
    Serial.println("ack");
}

void stopStream(const String &var, int index)
{
    streamEnabled = false;
    streamVariable = ""; // Limpar a variável
    Serial.print("Streaming parado para ");
    Serial.print(var);
    Serial.print(" no nó ");
    Serial.println(index);
}

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

String getLastMinuteBuffer(const String &var, int index)
{
    String result = "";
    int count = getLogCount();
    if (count == 0)
        return result;

    LogEntry *buffer = getLogBuffer();
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

// ===================== FUNÇÕES DO CIRCULAR BUFFER =====================

void initStorage()
{
    logIndex = 0;
    bufferFull = false;
}

void logData(unsigned long timestamp, float lux, float duty)
{
    logBuffer[logIndex].timestamp = timestamp;
    logBuffer[logIndex].lux = lux;
    logBuffer[logIndex].duty = duty;

    logIndex++;
    if (logIndex >= LOG_SIZE)
    {
        logIndex = 0;
        bufferFull = true;
    }
}

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

    Serial.println("End of dump.\n");
}

LogEntry *getLogBuffer()
{
    return logBuffer;
}

int getLogCount()
{
    return bufferFull ? LOG_SIZE : logIndex;
}

bool isBufferFull()
{
    return bufferFull;
}

int getCurrentIndex()
{
    return logIndex;
}

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
