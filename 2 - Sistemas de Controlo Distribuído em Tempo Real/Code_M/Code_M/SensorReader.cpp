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
#include "LEDController.h"
#include <Arduino.h>
#include <math.h>

//============================================================================
// VARIÁVEIS GLOBAIS
//============================================================================

// --- Estado de calibração do sensor ---
float calibrationOffset = 0.0;        // Offset de calibração para compensar desvios
float lastFilteredLux = -1.0;         // Último valor filtrado de iluminação
float LDR_B = log10(LDR_R10) - LDR_M; // Constante B da equação logarítmica do LDR
float rawLux = 0.0;                   // Valor bruto de lux (sem filtragem)
float baselineIlluminance = 0.0;      // Iluminação de fundo com LED desligado
float ledGain = 100.0;                // Ganho calibrado do LED (lux/duty)
bool filterEnabled = true;            // Ativar/desativar filtragem do sensor

// Adaptação à luz externa
extern float lastExternalLux;    // Última medição de luz externa (definida em LEDController.cpp)
extern float externalLuxAverage; // Média móvel da luz externa (definida em LEDController.cpp)
unsigned long lastAdaptTime = 0; // Timestamp da última adaptação

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
    filterEnabled = true;
}

/**
 * Lê e processa o valor de iluminação do sensor LDR com filtragem multi-estágio:
 * 1. Múltiplas amostras para redução de ruído
 * 2. Rejeição estatística de outliers
 * 3. Filtragem EMA para suavização temporal
 * 4. Aplicação de offset de calibração
 *
 * @return Valor de iluminação processado em lux
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

    // Armazenar o valor bruto (média de todas as amostras sem filtragem adicional)
    rawLux = sum / count;

    // Se a filtragem estiver desativada, retornar o valor bruto imediatamente
    if (!filterEnabled)
    {
        return rawLux;
    }

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

    // 6. Verificação de valores irrealistas
    if (calibratedLux > MAX_ILLUMINANCE)
    {
        Serial.print("AVISO: Iluminância irrealista detectada: ");
        Serial.print(calibratedLux);
        Serial.println(" lux - limitando a MAX_ILLUMINANCE");
        calibratedLux = MAX_ILLUMINANCE;
    }

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
    const int CAL_SAMPLES = 10;

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
 * Calibra o modelo de iluminância medindo a contribuição do LED
 * @return Ganho calibrado do sistema (y2-y1)
 */
float calibrateIlluminanceModel()
{
    const int SAMPLES = 5;              // Número de medições para média
    const int STABILIZE_TIME = 500;     // Tempo de espera entre estados estáveis em ms
    const int LED_RESPONSE_TIME = 2000; // Tempo de espera para o LDR responder às mudanças do LED

    Serial.println("Calibrando modelo de iluminância...");

    // Desligar LED e medir y1
    setLEDDutyCycle(0.0);
    delay(STABILIZE_TIME);

    // Fazer múltiplas medições e calcular média
    float y1 = 0.0;
    for (int i = 0; i < SAMPLES; i++)
    {
        y1 += readLux();
        delay(100);
    }
    y1 /= SAMPLES;

    // Armazenar em variável global para uso em getExternalIlluminance()
    baselineIlluminance = y1;

    Serial.print("Iluminância de fundo (LED desligado): ");
    Serial.print(y1);
    Serial.println(" lux");

    // Ligar LED ao máximo e esperar pela resposta do LDR
    setLEDDutyCycle(1.0);
    Serial.println("Aguardando estabilização do LED e LDR...");

    // Permitir tempo para o LED atingir brilho total e o LDR responder
    delay(LED_RESPONSE_TIME);

    // Fazer múltiplas medições e calcular média
    float y2 = 0.0;
    for (int i = 0; i < SAMPLES; i++)
    {
        y2 += readLux();
        delay(100);
    }
    y2 /= SAMPLES;

    // Verificação de valores irrealistas
    if (y2 > MAX_ILLUMINANCE)
    {
        Serial.print("AVISO: Iluminância muito alta detectada: ");
        Serial.print(y2);
        Serial.println(" lux - limitando a MAX_ILLUMINANCE");
        y2 = MAX_ILLUMINANCE;
    }

    Serial.print("Iluminância total (LED ligado): ");
    Serial.print(y2);
    Serial.println(" lux");

    // Calcular ganho: G = y2 - y1
    float gain = y2 - y1;

    // Garantir valor de ganho razoável
    if (gain < 0.1f || gain > MAX_ILLUMINANCE)
    {
        Serial.println("AVISO: Ganho calculado irrealista, usando valor padrão");
        gain = 100.0f; // Usar um valor padrão razoável
    }

    // Armazenar o ganho calibrado na variável global
    ledGain = gain;

    Serial.print("Ganho LED calibrado (G): ");
    Serial.println(gain);

    return gain;
}

/**
 * Realiza calibração abrangente do sistema:
 * 1. Calibra a precisão do sensor LDR
 * 2. Mede a contribuição do LED para cálculo de iluminância externa
 *
 * @param referenceValue Valor de iluminância de referência (tipicamente muito baixo, como 1.0)
 * @return Valor de ganho do LED calibrado (G)
 */
float calibrateSystem(float referenceValue)
{
    const int SAMPLES = 5;              // Número de medições para média
    const int STABILIZE_TIME = 500;     // Tempo de espera entre medições em ms
    const int LED_RESPONSE_TIME = 2000; // Tempo de espera para o LDR responder às mudanças do LED

    Serial.println("Iniciando calibração abrangente...");

    //---------------------------------------------------------------------
    // 1. Primeiro calibrar o sensor LDR para leituras absolutas precisas
    //---------------------------------------------------------------------
    float measuredLux = 0.0;
    const int CAL_SAMPLES = 10;

    Serial.println("Calibrando sensor LDR...");

    for (int i = 0; i < CAL_SAMPLES; i++)
    {
        // Usar leitura bruta especial para evitar calibração existente
        int adcValue = analogRead(LDR_PIN);
        float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
        if (voltage <= 0.0)
            continue;

        float resistance = FIXED_RESISTOR * (VCC / voltage - 1.0);
        float logR = log10(resistance);
        float logLux = (logR - LDR_B) / LDR_M;
        float rawLux = pow(10, logLux);

        measuredLux += rawLux;
        delay(50); // Pequeno atraso entre leituras
    }
    measuredLux /= CAL_SAMPLES;

    // Calcular o offset necessário
    calibrationOffset = referenceValue - measuredLux;

    Serial.print("Sensor calibrado: offset = ");
    Serial.println(calibrationOffset);

    //---------------------------------------------------------------------
    // 2. Agora calibrar o modelo de iluminância com contribuição do LED
    //---------------------------------------------------------------------
    Serial.println("Calibrando modelo de iluminância...");

    // Desligar LED e medir y1
    setLEDDutyCycle(0.0);
    delay(STABILIZE_TIME);

    // Fazer múltiplas medições e calcular média
    float y1 = 0.0;
    for (int i = 0; i < SAMPLES; i++)
    {
        y1 += readLux(); // Usando leituras calibradas agora
        delay(STABILIZE_TIME);
    }
    y1 /= SAMPLES;

    // Armazenar iluminância de base para cálculo de luz externa
    baselineIlluminance = y1;

    Serial.print("Iluminância de fundo (LED desligado): ");
    Serial.print(y1);
    Serial.println(" lux");

    // Ligar LED ao máximo e esperar pela resposta do LDR
    setLEDDutyCycle(1.0);
    Serial.println("Aguardando estabilização do LED e LDR...");

    // Permitir tempo para o LED atingir brilho total e o LDR responder
    delay(LED_RESPONSE_TIME);

    // Fazer múltiplas medições e calcular média
    float y2 = 0.0;
    for (int i = 0; i < SAMPLES; i++)
    {
        y2 += readLux();
        delay(STABILIZE_TIME);
    }
    y2 /= SAMPLES;

    Serial.print("Iluminância total (LED ligado): ");
    Serial.print(y2);
    Serial.println(" lux");

    // Calcular ganho: G = y2 - y1
    float gain = y2 - y1;

    // Verificar valores irrealistas
    if (gain < 0.1f || gain > MAX_ILLUMINANCE)
    {
        Serial.println("AVISO: Ganho calculado irrealista, usando valor padrão");
        gain = 100.0f; // Usar um valor padrão razoável
    }

    // Armazenar o ganho calibrado na variável global
    ledGain = gain;

    Serial.print("Ganho LED calibrado (G): ");
    Serial.println(gain);

    // Resetar LED para estado desligado após calibração
    setLEDDutyCycle(0.0);

    Serial.println("Calibração abrangente concluída!");
    return gain;
}

/**
 * Obtém a tensão atual no pino do LDR
 * Útil para diagnóstico do circuito e verificações de hardware
 * @return Tensão em Volts no pino do LDR
 */
float getVoltageAtLDR()
{
    int adcValue = analogRead(LDR_PIN);
    return (adcValue / MY_ADC_RESOLUTION) * VCC;
}

/**
 * Estima a iluminância externa subtraindo a contribuição do LED
 * Usa valor de ganho calibrado G para determinar a contribuição do LED
 * @return Iluminância externa estimada em lux
 */
float getExternalIlluminance()
{
    // Obter duty cycle atual do LED
    extern float dutyCycle;

    // Ler iluminância atual
    float measuredLux = readLux();

    // Remover offset de baseline
    float baselineOffset = baselineIlluminance;

    // Modelo linear usando ganho calibrado: ext = measured - (duty * ledGain)
    float ledContribution = dutyCycle * ledGain;

    // Calcular estimativa atual de lux externo
    float currentExternalLux = max(0.0f, measuredLux - ledContribution);

    // Aplicar suavização para reduzir ruído na estimativa
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