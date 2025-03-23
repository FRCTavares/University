/**
 * LEDController.cpp - Implementação do controlador de iluminação LED
 *
 * Este ficheiro implementa as funcionalidades de controlo da iluminação:
 * - Gestão do ciclo de trabalho (duty cycle) do LED
 * - Adaptação às condições de iluminação externa
 * - Estimativa de consumo energético
 * - Calibração da resposta do LED
 * - Interface para diferentes formas de controlo (percentagem, PWM, potência)
 */

//============================================================================
// FICHEIROS INCLUÍDOS
//============================================================================

#include "LEDController.h"
#include "Configuration.h"
#include "SensorReader.h"
#include "PIDController.h"
#include "StateManager.h"
#include "DataLogger.h"
#include <Arduino.h>

//============================================================================
// VARIÁVEIS GLOBAIS E CONSTANTES
//============================================================================

// Variáveis estáticas para o módulo
static int ledPin = LED_PIN; // Pino do LED definido em Configuration.h
static int pwmMax = PWM_MAX; // Valor máximo de PWM definido em Configuration.h
static int pwmMin = PWM_MIN; // Valor mínimo de PWM definido em Configuration.h

// --- Adaptação à Luz Externa ---
float lastExternalLux = 0.0;    // Última medição de luz externa estimada
float externalLuxAverage = 0.0; // Média móvel da luz externa

//============================================================================
// FUNÇÕES DE ADAPTAÇÃO AMBIENTAL
//============================================================================

/**
 * Adapta o controlo do LED às mudanças na iluminação externa
 * Implementa uma estratégia de feedforward para ajudar o controlo PID
 */
void adaptToExternalLight()
{
    static unsigned long lastAdaptTime = 0;
    static float previousExternal = -1.0;

    // Verificar apenas a cada 5 segundos para evitar ajustes rápidos
    if (millis() - lastAdaptTime < EXTERNAL_ADAPTATION_INTERVAL)
    {
        return;
    }
    lastAdaptTime = millis();

    // Obter iluminação externa atual
    float externalLux = getExternalIlluminance();

    // Ignorar primeira execução ou quando em modo manual
    if (previousExternal < 0 || !feedbackControl)
    {
        previousExternal = externalLux;
        return;
    }

    // Se a luz externa mudou significativamente (>1 lux)
    if (abs(externalLux - previousExternal) > EXTERNAL_LUX_THRESHOLD)
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

/**
 * Calcula o consumo de energia atual do LED
 * @return Consumo de energia em Watts
 */
float getPowerConsumption()
{
    return dutyCycle * MAX_POWER_WATTS;
}

/**
 * Obtém o tempo decorrido desde o início da execução
 * @return Tempo decorrido em segundos
 */
unsigned long getElapsedTime()
{
    return millis() / 1000;
}

//============================================================================
// INICIALIZAÇÃO E TESTE DO LED
//============================================================================

/**
 * Executa um teste visual rápido do LED
 * Aumenta e diminui a intensidade para verificar o funcionamento
 */
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

/**
 * Inicializa o driver do LED configurando o pino e parâmetros iniciais
 * @param pin Número do pino Arduino para o LED
 */
void initLEDDriver(int pin)
{
    ledPin = pin;
    pinMode(ledPin, OUTPUT);

    // Configurar PWM com definições ótimas

    // Começar com o LED desligado
    analogWrite(ledPin, pwmMin);
    dutyCycle = 0.0; // Utilizar a variável global de duty cycle

    // Mensagem de debug apenas se a debug estiver ativada
    if (DEBUG_MODE && DEBUG_LED)
    {
        Serial.print("Driver LED inicializado no pino ");
        Serial.println(pin);
    }
}

//============================================================================
// FUNÇÕES DE CONTROLO DO LED
//============================================================================

/**
 * Define o duty cycle do LED diretamente
 * @param newDutyCycle Novo duty cycle entre 0.0 e 1.0
 */
void setLEDDutyCycle(float newDutyCycle)
{
    // Validar e restringir entrada
    if (isnan(newDutyCycle) || isinf(newDutyCycle))
    {
        return; // Proteger contra entradas inválidas
    }

    // Restringir para intervalo válido
    newDutyCycle = constrain(newDutyCycle, 0.0f, 1.0f);

    // Aplicar duty cycle
    int pwmValue = (int)(newDutyCycle * pwmMax);
    analogWrite(ledPin, pwmValue);

    // Atualizar o duty cycle global
    dutyCycle = newDutyCycle;

    // Mensagem de debug apenas se a debug estiver ativada
    if (DEBUG_MODE && DEBUG_LED)
    {
        Serial.print("Duty cycle do LED definido para: ");
        Serial.println(newDutyCycle, 3);
    }
}

/**
 * Define a intensidade do LED como percentagem
 * @param percentage Percentagem de intensidade (0-100%)
 */
void setLEDPercentage(float percentage)
{
    percentage = constrain(percentage, 0.0f, 100.0f);
    float newDutyCycle = percentage / 100.0f;
    setLEDDutyCycle(newDutyCycle);

    // Mensagem de debug apenas se a debug estiver ativada
    if (DEBUG_MODE && DEBUG_LED)
    {
        Serial.print("Percentagem do LED definida para: ");
        Serial.println(percentage, 1);
    }
}

/**
 * Define o valor PWM direto para o LED
 * @param pwmValue Valor PWM entre pwmMin e pwmMax
 */
void setLEDPWMValue(int pwmValue)
{
    pwmValue = constrain(pwmValue, pwmMin, pwmMax);
    analogWrite(ledPin, pwmValue);

    // Atualizar duty cycle global
    dutyCycle = (float)pwmValue / pwmMax;

    // Mensagem de debug apenas se a debug estiver ativada
    if (DEBUG_MODE && DEBUG_LED)
    {
        Serial.print("Valor PWM do LED definido para: ");
        Serial.println(pwmValue);
    }
}

/**
 * Define a potência do LED em Watts
 * @param powerWatts Potência desejada em Watts (0-MAX_POWER_WATTS)
 */
void setLEDPower(float powerWatts)
{
    powerWatts = constrain(powerWatts, 0.0f, MAX_POWER_WATTS);
    float newDutyCycle = powerWatts / MAX_POWER_WATTS;
    setLEDDutyCycle(newDutyCycle);

    // Mensagem de debug apenas se a debug estiver ativada
    if (DEBUG_MODE && DEBUG_LED)
    {
        Serial.print("Potência do LED definida para: ");
        Serial.println(powerWatts, 3);
    }
}

//============================================================================
// FUNÇÕES DE LEITURA DE ESTADO DO LED
//============================================================================

/**
 * Obtém o duty cycle atual do LED
 * @return Duty cycle entre 0.0 e 1.0
 */
float getLEDDutyCycle()
{
    return dutyCycle;
}

/**
 * Obtém a percentagem atual de intensidade do LED
 * @return Percentagem entre 0 e 100
 */
float getLEDPercentage()
{
    return dutyCycle * 100.0f;
}

/**
 * Obtém o valor PWM atual do LED
 * @return Valor PWM entre pwmMin e pwmMax
 */
int getLEDPWMValue()
{
    return (int)(dutyCycle * pwmMax);
}

/**
 * Obtém a potência atual do LED em Watts
 * @return Potência em Watts
 */
float getLEDPower()
{
    return dutyCycle * MAX_POWER_WATTS;
}
