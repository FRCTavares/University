/**
 * Configuration.h - Configurações globais do sistema
 *
 * Este ficheiro define todas as constantes e configurações utilizadas pelo sistema:
 * - Constantes globais e limites do sistema
 * - Definições de hardware (pinos, características)
 * - Parâmetros de calibração para os sensores
 * - Configurações do controlador PID
 * - Constantes do protocolo de comunicação CAN
 * - Setpoints para diferentes estados operacionais
 * - Configurações de adaptação e monitorização
 */

 #pragma once
 #include <Arduino.h>
 #include <pico/unique_id.h>
 
 //============================================================================
 // CONSTANTES DO SISTEMA
 //============================================================================
 
 /**
  * iluminação máxima suportada pelo sistema em lux
  */
 #define MAX_ILLUMINANCE 2000.0f
 
 //============================================================================
 // DEFINIÇÕES DE PINOS
 //============================================================================
 
 #define LED_PIN 15   // Pino de controlo do LED (saída PWM)
 #define LDR_PIN A0   // Pino do sensor LDR (entrada analógica)
 
 //============================================================================
 // CONFIGURAÇÃO PWM
 //============================================================================
 
 #define PWM_FREQUENCY 30000  // Frequência do sinal PWM em Hz
 #define PWM_MAX 4095         // Valor máximo do ciclo PWM
 #define PWM_MIN 0            // Valor mínimo do ciclo PWM
 
 /**
  * Potência máxima do LED em watts
  */
 extern const float MAX_POWER_WATTS;
 
 //============================================================================
 // CALIBRAÇÃO DO SENSOR
 //============================================================================
 
 #define VCC 3.3                // Tensão de referência em volts
 #define MY_ADC_RESOLUTION 4095.0 // Resolução do conversor ADC (12 bits)
 #define FIXED_RESISTOR 10000.0 // Resistência fixa do divisor em ohms
 #define LDR_R10 225000.0       // Resistência do LDR a 10 lux
 #define LDR_M -1.0             // Expoente da relação lux-resistência
 
 //============================================================================
 // CONFIGURAÇÃO DE FILTROS
 //============================================================================
 
 #define NUM_SAMPLES 10        // Número de amostras para média móvel
 #define OUTLIER_THRESHOLD 2.0 // Limiar para rejeição de outliers (desvios padrão)
 #define ALPHA 0.3             // Coeficiente do filtro passa-baixo exponencial
 
 //============================================================================
 // PARÂMETROS DO CONTROLADOR PID
 //============================================================================
 
 #define KP 28.0f        // Ganho proporcional
 #define KI 230.0f       // Ganho integral
 #define DT 0.1f         // Tempo de amostragem em segundos
 
 //============================================================================
 // DECLARAÇÕES DE VARIÁVEIS EXTERNAS
 //============================================================================
 
 extern float setpointLux;       // Setpoint de iluminação desejada em lux
 extern float dutyCycle;         // Ciclo de trabalho atual [0..1]
 extern float refIlluminance;    // iluminação de referência para métricas
 extern bool occupancy;          // Indicador de ocupação da secretária
 extern bool antiWindup;         // Flag para ativar/desativar anti-windup no PID
 extern bool feedbackControl;    // Ativar/desativar controlo em malha fechada
 extern uint8_t nodeID;          // Identificador único deste nó
 extern bool periodicCANEnabled; // Ativar envio periódico de mensagens CAN
 extern bool canMonitorEnabled;  // Mostrar mensagens CAN recebidas
 
 //============================================================================
 // MODELO DE POTÊNCIA
 //============================================================================
 
 #define MAX_POWER_WATTS 1.0    // Potência máxima do LED em watts
 
 //============================================================================
 // CONFIGURAÇÃO DE REGISTOS
 //============================================================================
 
 #define LOG_SIZE 1200          // Tamanho do buffer circular para registos
 
 //============================================================================
 // CONFIGURAÇÃO CAN
 //============================================================================
 
 #define CAN_CS_PIN 17         // Pino de seleção do chip (CS)
 #define CAN_MOSI_PIN 19       // Pino Master-Out-Slave-In (MOSI)
 #define CAN_MISO_PIN 16       // Pino Master-In-Slave-Out (MISO)
 #define CAN_SCK_PIN 18        // Pino de relógio (SCK)
 
 //============================================================================
 // TIPOS DE MENSAGENS CAN
 //============================================================================
 
 #define CAN_TYPE_CONTROL 0x00    // Comandos de controlo
 #define CAN_TYPE_SENSOR 0x01     // Leituras de sensores
 #define CAN_TYPE_STATUS 0x02     // Estado do nó
 #define CAN_TYPE_CONFIG 0x03     // Configuração
 #define CAN_TYPE_ERROR 0x04      // Notificação de erro
 #define CAN_TYPE_QUERY 0x05      // Pedido de informação
 #define CAN_TYPE_RESPONSE 0x06   // Resposta a pedido
 #define CAN_TYPE_HEARTBEAT 0x07  // Sinal de presença na rede
 #define CAN_CTRL_STATE_CHANGE 0x10 // Mudança de estado do LED
 
 //============================================================================
 // NÍVEIS DE PRIORIDADE CAN
 //============================================================================
 
 #define CAN_PRIO_HIGH 0x00    // Prioridade alta (maior)
 #define CAN_PRIO_NORMAL 0x01  // Prioridade normal
 #define CAN_PRIO_LOW 0x02     // Prioridade baixa
 #define CAN_PRIO_LOWEST 0x03  // Prioridade mínima
 
 //============================================================================
 // ENDEREÇOS DE NÓS CAN
 //============================================================================
 
 #define CAN_ADDR_BROADCAST 0x00 // Endereço de difusão (todos os nós)
 
 //============================================================================
 // CONFIGURAÇÕES DE COMUNICAÇÃO CAN
 //============================================================================
 
 #define PERIODIC_CAN_ENABLED false // Ativar envio periódico de mensagens
 #define CAN_MONITOR_ENABLED false  // Mostrar mensagens recebidas
 
 //============================================================================
 // SETPOINTS PARA DIFERENTES ESTADOS
 //============================================================================
 
 #define SETPOINT_OFF 0.0        // iluminação quando desligado (lux)
 #define SETPOINT_UNOCCUPIED 5.0 // iluminação em estado não-ocupado (lux)
 #define SETPOINT_OCCUPIED 15.0  // iluminação em estado ocupado (lux)
 
 //============================================================================
 // PARÂMETROS DE ADAPTAÇÃO A LUZ EXTERNA
 //============================================================================
 
 #define EXT_LUX_ALPHA 0.05f        // Coeficiente de suavização para luz externa
 #define EXTERNAL_ADAPTATION_INTERVAL 5000 // Verificar a cada 5 segundos
 #define EXTERNAL_LUX_THRESHOLD 1.0f      // Limiar de mudança significativa (lux)
 
 //============================================================================
 // CONFIGURAÇÕES DE STREAMING
 //============================================================================
 
 #define STREAM_INTERVAL 500    // Intervalo entre amostras de streaming (ms)
 
 //============================================================================
 // FLAGS DE Debug
 //============================================================================
 
 #define DEBUG_MODE false        // Interruptor mestre de Debug
 #define DEBUG_LED false         // Mensagens de Debug do driver LED
 #define DEBUG_SENSOR false      // Debug das leituras do sensor
 #define DEBUG_PID false         // Debug do controlo PID
 #define DEBUG_PLOTTING false    // Saída para o plotter serial
