/**
 * CANCom.h - Interface de comunicação CAN-BUS
 *
 * Este ficheiro define a interface para o sistema de comunicação CAN-BUS:
 * - Funções para inicialização e configuração do controlador MCP2515
 * - Envio e receção de mensagens através do barramento CAN
 * - Formatação de mensagens para diferentes tipos de comando
 * - Registo de callbacks para processamento de mensagens
 * - Estatísticas e diagnóstico da rede CAN
 */

 #pragma once
 #include <SPI.h>
 #include <mcp2515.h>
 
 //============================================================================
 // TIPOS E DEFINIÇÕES
 //============================================================================
 
 /**
  * Tipo de função callback para processamento de mensagens CAN
  * @param msg Mensagem CAN recebida para processamento
  */
 typedef void (*CANMessageCallback)(const can_frame &msg);
 

 //============================================================================
 // FUNÇÕES DE INICIALIZAÇÃO
 //============================================================================
 
 /**
  * Inicializa a comunicação CAN
  * Configura o controlador MCP2515 e atribui um ID único ao nó
  */
 void initCANComm();
 
 //============================================================================
 // FUNÇÕES DE MANIPULAÇÃO DE MENSAGENS
 //============================================================================
 
 /**
  * Envia uma mensagem CAN pelo barramento
  * @param frame Estrutura contendo a mensagem a enviar
  * @return Código de erro da operação
  */
 MCP2515::ERROR sendCANMessage(const can_frame &frame);
 
 /**
  * Lê uma mensagem CAN do barramento, se disponível
  * @param frame Ponteiro para estrutura onde a mensagem será armazenada
  * @return Código de erro da operação
  */
 MCP2515::ERROR readCANMessage(struct can_frame *frame);
 
 /**
  * Loop principal de processamento CAN, deve ser chamado ciclicamente
  * Processa mensagens recebidas e invoca callbacks registados
  */
 void canCommLoop();
 
 //============================================================================
 // FUNÇÕES AUXILIARES
 //============================================================================
 
 /**
  * Constrói um ID CAN a partir dos seus componentes
  * @param msgType Tipo de mensagem (0-7)
  * @param destAddr Endereço de destino (0-63)
  * @param priority Prioridade (0-3)
  * @return ID CAN formatado
  */
 uint32_t buildCANId(uint8_t msgType, uint8_t destAddr, uint8_t priority);
 
 /**
  * Extrai os componentes de um ID CAN
  * @param canId ID CAN completo
  * @param msgType Tipo de mensagem extraído
  * @param destAddr Endereço de destino extraído
  * @param priority Prioridade extraída
  */
 void parseCANId(uint32_t canId, uint8_t &msgType, uint8_t &destAddr, uint8_t &priority);
 
 /**
  * Converte um valor float em representação de bytes
  * @param value Valor float a converter
  * @param bytes Array de 4 bytes para armazenar o resultado
  */
 void floatToBytes(float value, uint8_t *bytes);
 
 /**
  * Converte bytes de volta para valor float
  * @param bytes Array de 4 bytes a converter
  * @return Valor float resultante
  */
 float bytesToFloat(const uint8_t *bytes);
 
 //============================================================================
 // FUNÇÕES DE ENVIO DE MENSAGENS ESPECÍFICAS
 //============================================================================
 
 /**
  * Envia uma leitura de sensor pela rede CAN
  * @param destAddr Endereço do nó destino
  * @param sensorType Tipo de sensor (0=lux, 1=duty, etc)
  * @param value Valor lido
  * @return true se enviado com sucesso
  */
 bool sendSensorReading(uint8_t destAddr, uint8_t sensorType, float value);
 
 /**
  * Envia um comando de controlo para outro nó
  * @param destAddr Endereço do nó destino
  * @param controlType Tipo de comando de controlo
  * @param value Valor do parâmetro de controlo
  * @return true se enviado com sucesso
  */
 bool sendControlCommand(uint8_t destAddr, uint8_t controlType, float value);
 
 /**
  * Envia uma resposta a uma consulta
  * @param destNode Endereço do nó destino
  * @param value Valor da resposta
  * @return true se enviado com sucesso
  */
 bool sendQueryResponse(uint8_t destNode, float value);
 
 /**
  * Envia um heartbeat para informar presença na rede
  * @return true se enviado com sucesso
  */
 bool sendHeartbeat();
 
 //============================================================================
 // REGISTO DE CALLBACKS
 //============================================================================
 
 /**
  * Regista uma função callback para processamento de mensagens
  * @param callback Função a ser chamada quando uma mensagem é recebida
  */
 void setCANMessageCallback(CANMessageCallback callback);
 
 //============================================================================
 // ESTATÍSTICAS E DIAGNÓSTICO
 //============================================================================
 
 /**
  * Obtém estatísticas da comunicação CAN
  * @param sent Número de mensagens enviadas
  * @param received Número de mensagens recebidas
  * @param errors Número de erros ocorridos
  * @param avgLatency Latência média em microssegundos
  */
 void getCANStats(uint32_t &sent, uint32_t &received, uint32_t &errors, float &avgLatency);
 
 /**
  * Reinicia os contadores de estatísticas da comunicação CAN
  */
 void resetCANStats();
 
 //============================================================================
 // VARIÁVEIS GLOBAIS
 //============================================================================
 
 /**
  * ID único deste nó na rede CAN
  * É gerado automaticamente na inicialização com base no hardware
  */
 extern uint8_t nodeID;
 
 /**
  * Flag para ativar/desativar monitorização CAN na porta série
  */
 extern bool canMonitorEnabled;