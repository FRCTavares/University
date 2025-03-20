/**
 * SerialInterface.h - Interface de comandos via porta série
 *
 * Este ficheiro define a interface para comunicação com o sistema via porta série:
 * - Processamento de comandos de controlo e configuração
 * - Parsing de instruções de utilizador
 * - Reencaminhamento de comandos via rede CAN-BUS
 * - Interface de monitorização para obtenção de dados
 * - Gestão de streams de dados em tempo real
 */

 #pragma once

 //============================================================================
 // FUNÇÕES DE INICIALIZAÇÃO
 //============================================================================
 
 /**
  * Inicializa a interface de comunicação série
  * Configura a comunicação serial e prepara o sistema para receber comandos
  */
 void initSerialInterface();
 
 //============================================================================
 // FUNÇÕES DE PROCESSAMENTO DE COMANDOS
 //============================================================================
 
 /**
  * Processa comandos recebidos pela porta série
  * Deve ser chamada ciclicamente no loop principal para verificar novos comandos
  */
 void processSerialCommands();
 
 //============================================================================
 // FUNÇÕES DE STREAMING DE DADOS
 //============================================================================
 
 /**
  * Gere o envio de dados em tempo real (streaming)
  * Permite a transmissão contínua de variáveis do sistema
  */
 void handleStreaming();