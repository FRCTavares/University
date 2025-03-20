/**
 * SensorReader.cpp - Implementação do subsistema de leitura de sensores
 *
 * Este ficheiro implementa as funcionalidades de leitura e processamento de sensores:
 * - Leitura do sensor LDR (Light Dependent Resistor) para medição de iluminação
 * - Filtragem de ruído através de múltiplas amostras
 * - Deteção e eliminação de outliers
 * - Calibração do sensor
 * - Conversão de valores ADC para unidades físicas (lux)
 * - Suavização temporal através de média móvel exponencial
 */

//============================================================================
// FICHEIROS INCLUÍDOS
//============================================================================

#include "SensorReader.h"
#include "Configuration.h"
#include <Arduino.h>
#include <math.h>

//============================================================================
// VARIÁVEIS GLOBAIS
//============================================================================

// --- Estado de calibração do sensor ---
float calibrationOffset = 0.0;     // Offset de calibração para compensar desvios
float lastFilteredLux = -1.0;      // Último valor filtrado de iluminação
float LDR_B = log10(LDR_R10) - LDR_M;  // Constante B da equação logarítmica do LDR

// Adaptação à luz externa
extern float lastExternalLux;      // Última medição de luz externa (definida em LEDController.cpp)
extern float externalLuxAverage;   // Média móvel da luz externa (definida em LEDController.cpp)
unsigned long lastAdaptTime = 0;   // Timestamp da última adaptação

//============================================================================
// FUNÇÕES DE INICIALIZAÇÃO E CALIBRAÇÃO
//============================================================================

/**
 * Inicializa o subsistema de sensores
 * Configura pinos e reinicia os valores de calibração
 */
void initSensor()
{
    pinMode(LDR_PIN, INPUT);
    lastFilteredLux = -1.0;
    calibrationOffset = 0.0;
}

/**
 * Lê e processa o valor de iluminação do sensor LDR
 * Aplica várias técnicas de processamento de sinal para obter uma leitura precisa:
 * 1. Recolha de múltiplas amostras para redução de ruído
 * 2. Cálculo estatístico (média e desvio padrão)
 * 3. Filtragem de outliers
 * 4. Suavização temporal com média móvel exponencial
 * 5. Calibração e verificação de limites
 * 
 * @return Valor de iluminação em lux
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
 * Calibra o sensor de luz com um valor de referência conhecido
 * Calcula e aplica um offset para ajustar as leituras do sensor
 * 
 * @param knownLux Valor de referência em lux para calibração
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

/**
 * Obtém a tensão atual no pino do LDR
 * Útil para diagnóstico do circuito e verificações de hardware
 * 
 * @return Tensão em Volts no pino do LDR
 */
float getVoltageAtLDR()
{
    int adcValue = analogRead(LDR_PIN);
    return (adcValue / MY_ADC_RESOLUTION) * VCC;
}