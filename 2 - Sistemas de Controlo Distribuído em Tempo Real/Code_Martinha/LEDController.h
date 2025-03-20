/**
 * LEDController.h - Interface para controlo do LED
 *
 * Este ficheiro define as funções para controlo e monitorização do LED:
 * - Inicialização do controlador LED com configuração PWM
 * - Controlo do LED por ciclo de trabalho, percentagem ou potência
 * - Consulta de valores atuais (ciclo de trabalho, percentagem, potência)
 * - Estimativa de consumo energético e iluminação externa
 * - Funções de teste para verificação do funcionamento do hardware
 */

 #pragma once

 //============================================================================
 // FUNÇÕES DE INICIALIZAÇÃO
 //============================================================================
 
 /**
  * Inicializa o controlador do LED
  * @param pin Pino PWM utilizado para controlar a luminária
  */
 void initLEDDriver(int pin);
 
 //============================================================================
 // FUNÇÕES DE CONTROLO
 //============================================================================
 
 /**
  * Define o ciclo de trabalho PWM do LED
  * @param dutyCycle Valor do ciclo de trabalho [0.0 a 1.0]
  */
 void setLEDDutyCycle(float dutyCycle);
 
 /**
  * Define a percentagem de potência do LED
  * @param percentage Percentagem de potência [0.0 a 100.0]
  */
 void setLEDPercentage(float percentage);
 
 /**
  * Define diretamente o valor PWM do LED
  * @param pwmValue Valor PWM [0 a MAX_PWM]
  */
 void setLEDPWMValue(int pwmValue);
 
 /**
  * Define a potência de saída do LED em watts
  * @param powerWatts Potência desejada em watts [0.0 a MAX_POWER_WATTS]
  */
 void setLEDPower(float powerWatts);
 
 //============================================================================
 // FUNÇÕES DE CONSULTA
 //============================================================================
 
 /**
  * Obtém o ciclo de trabalho atual do LED
  * @return Valor do ciclo de trabalho [0.0 a 1.0]
  */
 float getLEDDutyCycle();
 
 /**
  * Obtém a percentagem atual de potência do LED
  * @return Percentagem de potência [0.0 a 100.0]
  */
 float getLEDPercentage();
 
 /**
  * Obtém o valor PWM atual do LED
  * @return Valor PWM atual [0 a MAX_PWM]
  */
 int getLEDPWMValue();
 
 /**
  * Obtém a potência atual do LED em watts
  * @return Potência estimada em watts
  */
 float getLEDPower();
 
 /**
  * Obtém uma estimativa da iluminação externa (ambiental)
  * @return iluminação externa estimada em lux
  */
 float getExternalIlluminance();
 
 /**
  * Obtém o consumo de energia acumulado do LED
  * @return Consumo de energia em joules ou unidades relativas
  */
 float getPowerConsumption();
 
 /**
  * Obtém o tempo decorrido desde o início da operação do sistema
  * @return Tempo decorrido em milissegundos
  */
 unsigned long getElapsedTime();
 
 //============================================================================
 // FUNÇÕES DE TESTE
 //============================================================================
 
 /**
  * Executa um teste básico do LED
  * Verifica o funcionamento do hardware realizando uma sequência de teste
  */
 void testLED();