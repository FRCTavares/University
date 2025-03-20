#include "LEDController.h"
#include "Configuration.h"
#include "SensorReader.h"
#include "PIDController.h"
#include "StateManager.h"
#include "DataLogger.h"
#include <Arduino.h>


// Variáveis estáticas para o módulo
static int ledPin = LED_PIN; // Use value from Configuration.h
static int pwmMax = PWM_MAX; // Use value from Configuration.h
static int pwmMin = PWM_MIN; // Use value from Configuration.h

// --- Adaptação à Luz Externa ---
float lastExternalLux = 0.0;
float externalLuxAverage = 0.0;

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

    // Obter iluminância externa atual
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

float getPowerConsumption()
{
    return dutyCycle * MAX_POWER_WATTS;
}

unsigned long getElapsedTime()
{
    return millis() / 1000;
}

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

void initLEDDriver(int pin)
{
    ledPin = pin;
    pinMode(ledPin, OUTPUT);

    // Configurar PWM com definições ótimas

    // Começar com o LED desligado
    analogWrite(ledPin, pwmMin);
    dutyCycle = 0.0; // Utilizar a variável global de duty cycle

    // Mensagem de depuração apenas se a depuração estiver ativada
    if (DEBUG_MODE && DEBUG_LED)
    {
        Serial.print("Driver LED inicializado no pino ");
        Serial.println(pin);
    }
}

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

    // Mensagem de depuração apenas se a depuração estiver ativada
    if (DEBUG_MODE && DEBUG_LED)
    {
        Serial.print("Duty cycle do LED definido para: ");
        Serial.println(newDutyCycle, 3);
    }
}

void setLEDPercentage(float percentage)
{
    percentage = constrain(percentage, 0.0f, 100.0f);
    float newDutyCycle = percentage / 100.0f;
    setLEDDutyCycle(newDutyCycle);

    // Mensagem de depuração apenas se a depuração estiver ativada
    if (DEBUG_MODE && DEBUG_LED)
    {
        Serial.print("Percentagem do LED definida para: ");
        Serial.println(percentage, 1);
    }
}

void setLEDPWMValue(int pwmValue)
{
    pwmValue = constrain(pwmValue, pwmMin, pwmMax);
    analogWrite(ledPin, pwmValue);

    // Atualizar duty cycle global
    dutyCycle = (float)pwmValue / pwmMax;

    // Mensagem de depuração apenas se a depuração estiver ativada
    if (DEBUG_MODE && DEBUG_LED)
    {
        Serial.print("Valor PWM do LED definido para: ");
        Serial.println(pwmValue);
    }
}

void setLEDPower(float powerWatts)
{
    powerWatts = constrain(powerWatts, 0.0f, MAX_POWER_WATTS);
    float newDutyCycle = powerWatts / MAX_POWER_WATTS;
    setLEDDutyCycle(newDutyCycle);

    // Mensagem de depuração apenas se a depuração estiver ativada
    if (DEBUG_MODE && DEBUG_LED)
    {
        Serial.print("Potência do LED definida para: ");
        Serial.println(powerWatts, 3);
    }
}

float getLEDDutyCycle()
{
    return dutyCycle;
}

float getLEDPercentage()
{
    return dutyCycle * 100.0f;
}

int getLEDPWMValue()
{
    return (int)(dutyCycle * pwmMax);
}

float getLEDPower()
{
    return dutyCycle * MAX_POWER_WATTS;
}
