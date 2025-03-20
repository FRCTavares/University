// (LDR sensor implementation)
#include "SensorReader.h"
#include "Configuration.h"
#include <Arduino.h>
#include <math.h>

// --- Sensor calibration state ---
float calibrationOffset = 0.0;
float lastFilteredLux = -1.0;
float LDR_B = log10(LDR_R10) - LDR_M;

// External light adaptation
extern float lastExternalLux;
extern float externalLuxAverage;
unsigned long lastAdaptTime = 0;

void initSensor()
{
    pinMode(LDR_PIN, INPUT);
    lastFilteredLux = -1.0;
    calibrationOffset = 0.0;
}

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

float getVoltageAtLDR()
{
    int adcValue = analogRead(LDR_PIN);
    return (adcValue / MY_ADC_RESOLUTION) * VCC;
}
