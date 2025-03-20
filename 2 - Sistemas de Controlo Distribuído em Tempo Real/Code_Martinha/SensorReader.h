/**
 * SensorReader.h - Interface para a leitura do sensor de iluminação
 *
 * Este ficheiro define as funções para a gestão do sensor de iluminação:
 * - Inicialização e configuração do sensor LDR
 * - Leitura de valores de iluminação com filtragem
 * - Calibração para condições específicas
 * - Estimativa de iluminação externa
 * - Adaptação a mudanças de iluminação ambiente
 */

 #pragma once

 //============================================================================
 // FUNÇÕES DE INICIALIZAÇÃO
 //============================================================================
 
 /**
  * Inicializa o sensor de iluminação
  * Configura os pinos e os parâmetros iniciais do sensor
  */
 void initSensor();
 
 //============================================================================
 // FUNÇÕES DE LEITURA
 //============================================================================
 
 /**
  * Lê o valor atual de iluminação em lux com filtragem
  * @return Valor de iluminação em lux (unidade SI de iluminação)
  */
 float readLux();
 
 /**
  * Obtém a tensão atual no pino do LDR
  * @return Valor da tensão em Volts
  */
 float getVoltageAtLDR();
 
 /**
  * Obtém uma estimativa da iluminação externa (luz ambiente)
  * @return Valor estimado da iluminação externa em lux
  */
 float getExternalIlluminance();
 
 //============================================================================
 // FUNÇÕES DE CALIBRAÇÃO E ADAPTAÇÃO
 //============================================================================
 
 /**
  * Calibra o sensor de iluminação com um valor de referência conhecido
  * @param knownLux Valor de referência conhecido em lux
  */
 void calibrateLuxSensor(float knownLux);
 
 /**
  * Adapta os parâmetros do sensor às mudanças na iluminação externa
  * Deve ser chamada periodicamente para compensar variações ambientais
  */
 void adaptToExternalLight();