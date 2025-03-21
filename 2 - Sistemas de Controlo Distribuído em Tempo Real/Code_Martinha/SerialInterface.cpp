/**
 * SerialInterface.cpp - Implementação da interface de comandos via porta série
 *
 * Este ficheiro implementa o interpretador de comandos que permite controlar o sistema:
 * - Processamento de comandos via interface USB/Serial
 * - Parsing e validação de entradas do utilizador
 * - Encaminhamento de comandos para nós locais ou remotos via CAN-BUS
 * - Interface de controlo para ajuste de parâmetros do sistema
 * - Funções de consulta para monitorização do estado do sistema
 * - Diagnóstico e estatísticas da rede CAN
 */

//============================================================================
// FICHEIROS INCLUÍDOS
//============================================================================

#include "SerialInterface.h"
#include "Configuration.h"
#include "LEDController.h"
#include "SensorReader.h"
#include "StateManager.h"
#include "DataLogger.h"
#include "CANCom.h"
#include "PIDController.h"
#include <Arduino.h>

//============================================================================
// VARIÁVEIS GLOBAIS
//============================================================================

// Variáveis externas
extern float setpointLux;      // iluminação desejada (setpoint)
extern float dutyCycle;        // Ciclo de trabalho atual [0..1]
extern bool feedbackControl;   // Ativa/desativa controlo por realimentação
extern bool canMonitorEnabled; // Mostrar mensagens recebidas
extern PIDController pid;

// Buffer para comandos recebidos pela porta série
static String input = "";

//============================================================================
// INICIALIZAÇÃO
//============================================================================

/**
 * Inicializa a interface de comandos série
 * Configura a comunicação serial com a velocidade adequada
 */
void initSerialInterface()
{
    Serial.begin(115200);
    Serial.println(F("=== Sistema de Controlo de Iluminação ==="));
    Serial.println(F("Digite 'h' para ver a lista de comandos disponíveis"));
}


//============================================================================
// FUNÇÕES AUXILIARES DE PARSING
//============================================================================

/**
 * Divide uma linha de comando em tokens separados por espaço
 * Utilizado para interpretar comandos com múltiplos parâmetros
 * 
 * @param cmd String de comando a ser analisada
 * @param tokens Array onde os tokens encontrados serão armazenados
 * @param maxTokens Número máximo de tokens a serem extraídos
 * @param numFound Referência para armazenar o número de tokens encontrados
 */
static void parseTokens(const String &cmd, String tokens[], int maxTokens, int &numFound)
{
    numFound = 0;
    int startIdx = 0;
    while (numFound < maxTokens)
    {
        int spaceIdx = cmd.indexOf(' ', startIdx);
        if (spaceIdx == -1)
        {
            if (startIdx < (int)cmd.length())
            {
                tokens[numFound++] = cmd.substring(startIdx);
            }
            break;
        }
        tokens[numFound++] = cmd.substring(startIdx, spaceIdx);
        startIdx = spaceIdx + 1;
    }
}

/**
 * Determina se um comando deve ser processado localmente ou encaminhado via CAN
 * 
 * @param targetNode ID do nó alvo do comando
 * @return true se o comando deve ser encaminhado, false se deve ser processado localmente
 */
static bool shouldForwardCommand(uint8_t targetNode)
{
    // Se o destino for transmissão (0) ou corresponder a este nó, processar localmente
    if (targetNode == 0 || targetNode == nodeID)
    {
        return false;
    }
    // Caso contrário, reencaminhar para o nó de destino
    return true;
}

//============================================================================
// PROCESSAMENTO DE COMANDOS
//============================================================================

/**
 * Processa uma linha de comando recebida pela interface série
 * Analisa e executa comandos de controlo, medição e comunicação CAN
 * 
 * @param cmdLine Linha de comando a ser processada
 */
static void processCommandLine(const String &cmdLine)
{
    String trimmed = cmdLine;
    trimmed.trim();
    if (trimmed.length() == 0)
        return;

    // Tokenizar o comando para análise
    const int MAX_TOKENS = 6;
    String tokens[MAX_TOKENS];
    int numTokens = 0;
    parseTokens(trimmed, tokens, MAX_TOKENS, numTokens);
    if (numTokens == 0)
        return;

    String c0 = tokens[0];


    // ------------------- COMANDOS DE CONTROLO -------------------

    // "u <i> <val>" => definir duty cycle
    if (c0 == "u")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }

        uint8_t targetNode = tokens[1].toInt();
        float val = tokens[2].toFloat();

        if (val < 0.0f || val > 1.0f)
        {
            Serial.println("err");
            return;
        }

        // Verificar se precisamos de reencaminhar este comando
        if (shouldForwardCommand(targetNode))
        {
            // Reencaminhar para nó específico - tipo de controlo 4 = duty cycle
            if (sendControlCommand(targetNode, 4, val))
            {
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha no reencaminhamento CAN");
            }
            return;
        }

        // Tratar o caso de difusão (targetNode = 0)
        if (targetNode == 0)
        {
            if (sendControlCommand(CAN_ADDR_BROADCAST, 4, val))
            {
                // Também aplicar localmente, pois a difusão inclui este nó
                setLEDDutyCycle(val);
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha na difusão CAN");
            }
            return;
        }

        // Aplicar localmente
        setLEDDutyCycle(val);
        Serial.println("ack");
        return;
    }
    // "p <i> <percentagem>" => definir LED por percentagem
    else if (c0 == "p")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }

        uint8_t targetNode = tokens[1].toInt();
        float val = tokens[2].toFloat();

        if (val < 0.0f || val > 100.0f)
        {
            Serial.println("err");
            return;
        }

        // Verificar se precisamos de reencaminhar este comando
        if (shouldForwardCommand(targetNode))
        {
            // Reencaminhar para nó específico - tipo de controlo 5 = percentagem
            if (sendControlCommand(targetNode, 5, val))
            {
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha no reencaminhamento CAN");
            }
            return;
        }

        // Tratar o caso de difusão (targetNode = 0)
        if (targetNode == 0)
        {
            if (sendControlCommand(CAN_ADDR_BROADCAST, 5, val))
            {
                // Também aplicar localmente, pois a difusão inclui este nó
                setLEDPercentage(val);
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha na difusão CAN");
            }
            return;
        }

        // Aplicar localmente
        setLEDPercentage(val);
        Serial.println("ack");
        return;
    }
    
    else if (c0 == "w")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }

        uint8_t targetNode = tokens[1].toInt();
        float val = tokens[2].toFloat();

        if (val < 0.0f || val > MAX_POWER_WATTS)
        {
            Serial.println("err");
            return;
        }

        // Verificar se precisamos de reencaminhar este comando
        if (shouldForwardCommand(targetNode))
        {
            // Reencaminhar para nó específico - tipo de controlo 6 = potência em watts
            if (sendControlCommand(targetNode, 6, val))
            {
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha no reencaminhamento CAN");
            }
            return;
        }

        // Tratar o caso de difusão (targetNode = 0)
        if (targetNode == 0)
        {
            if (sendControlCommand(CAN_ADDR_BROADCAST, 6, val))
            {
                // Também aplicar localmente, pois a difusão inclui este nó
                setLEDPower(val);
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha na difusão CAN");
            }
            return;
        }

        setLEDPower(val);
        Serial.println("ack");
        return;
    }
    
    else if (c0 == "o")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }

        uint8_t targetNode = tokens[1].toInt();
        int val = tokens[2].toInt();

        if (val != 0 && val != 1)
        {
            Serial.println("err");
            return;
        }

        // Verificar se precisamos de reencaminhar este comando
        if (shouldForwardCommand(targetNode))
        {
            // Reencaminhar para nó específico - tipo de controlo 7 = ocupação
            if (sendControlCommand(targetNode, 7, val))
            {
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha no reencaminhamento CAN");
            }
            return;
        }

        // Tratar o caso de difusão (targetNode = 0)
        if (targetNode == 0)
        {
            if (sendControlCommand(CAN_ADDR_BROADCAST, 7, val))
            {
                // Também aplicar localmente, pois a difusão inclui este nó
                occupancy = (val == 1);
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha na difusão CAN");
            }
            return;
        }

        occupancy = (val == 1);
        Serial.println("ack");
        return;
    }
    // "a <i> <val>" => ativar/desativar anti-windup
    else if (c0 == "a")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }

        uint8_t targetNode = tokens[1].toInt();
        int val = tokens[2].toInt();

        if (val != 0 && val != 1)
        {
            Serial.println("err");
            return;
        }

        // Verificar se precisamos de reencaminhar este comando
        if (shouldForwardCommand(targetNode))
        {
            // Reencaminhar para nó específico - tipo de controlo 8 = anti-windup
            if (sendControlCommand(targetNode, 8, val))
            {
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha no reencaminhamento CAN");
            }
            return;
        }

        // Tratar o caso de difusão (targetNode = 0)
        if (targetNode == 0)
        {
            if (sendControlCommand(CAN_ADDR_BROADCAST, 8, val))
            {
                // Também aplicar localmente, pois a difusão inclui este nó
                antiWindup = (val == 1);
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha na difusão CAN");
            }
            return;
        }

        antiWindup = (val == 1);
        Serial.println("ack");
        return;
    }
    // "f <i> <val>" => ativar/desativar controlo por feedback
    else if (c0 == "f")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }

        uint8_t targetNode = tokens[1].toInt();
        int val = tokens[2].toInt();

        if (val != 0 && val != 1)
        {
            Serial.println("err");
            return;
        }

        // Verificar se precisamos de reencaminhar este comando
        if (shouldForwardCommand(targetNode))
        {
            // Reencaminhar para nó específico - tipo de controlo 9 = controlo por feedback
            if (sendControlCommand(targetNode, 9, val))
            {
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha no reencaminhamento CAN");
            }
            return;
        }

        // Tratar o caso de difusão (targetNode = 0)
        if (targetNode == 0)
        {
            if (sendControlCommand(CAN_ADDR_BROADCAST, 9, val))
            {
                // Também aplicar localmente, pois a difusão inclui este nó
                feedbackControl = (val == 1);
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha na difusão CAN");
            }
            return;
        }

        feedbackControl = (val == 1);
        Serial.println("ack");
        return;
    }
    // "r <i> <val>" => definir referência de iluminação
    else if (c0 == "r")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }

        uint8_t targetNode = tokens[1].toInt();
        float val = tokens[2].toFloat();

        if (val < 0.0f || val > MAX_ILLUMINANCE)
        {
            Serial.println("err");
            return;
        }

        // Verificar se precisamos de reencaminhar este comando
        if (shouldForwardCommand(targetNode))
        {
            // Reencaminhar para nó específico - tipo de controlo 10 = referência de iluminação
            if (sendControlCommand(targetNode, 10, val))
            {
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha no reencaminhamento CAN");
            }
            return;
        }

        // Tratar o caso de difusão (targetNode = 0)
        if (targetNode == 0)
        {
            if (sendControlCommand(CAN_ADDR_BROADCAST, 10, val))
            {
                // Também aplicar localmente, pois a difusão inclui este nó
                refIlluminance = val;
                setpointLux = val;
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha na difusão CAN");
            }
            return;
        }

        refIlluminance = val;
        setpointLux = val;
        Serial.println("ack");
        return;
    }
    
    //============================================================================
    // COMANDO DE AJUDA
    //============================================================================
    
    // "h" => mostrar ajuda
    else if (c0 == "h")
    {
        Serial.println("\n=== Sistema de Controlo de Iluminação ===");
        Serial.println("\nComandos de Controlo:");
        Serial.println("  u <nó> <val>    - Definir duty cycle [0-1]");
        Serial.println("  p <nó> <val>    - Definir percentagem LED [0-100]");
        Serial.println("  w <nó> <val>    - Definir potência em watts");
        Serial.println("  r <nó> <val>    - Definir referência de iluminação (lux)");
        Serial.println("  f <nó> <val>    - Ativar/desativar controlo por feedback (0/1)");
        Serial.println("  o <nó> <val>    - Definir estado de ocupação (0/1)");
        Serial.println("  a <nó> <val>    - Ativar/desativar anti-windup (0/1)");
        
        Serial.println("\nComandos do Controlador PI:");
        Serial.println("  k <val>         - Definir ambos os ganhos Kp=Ki (apenas local)");
        Serial.println("  b <val>        - Definir factor de ponderação Beta [0-1] (apenas local)");
        
        Serial.println("\nComandos de Estado:");
        Serial.println("  st <nó> <estado> - Definir estado (off/unoccupied/occupied)");
        
        Serial.println("\nComandos de Consulta:");
        Serial.println("  g <var> <nó>    - Obter valor da variável");
        Serial.println("    Variáveis: u(duty), y(lux), r(ref), o(ocupação),");
        Serial.println("    a(anti-windup), f(feedback), V(visibilidade),");
        Serial.println("    F(flicker), E(energia), v(tensão), d(ilum. externa)");
        
        Serial.println("\nComandos CAN:");
        Serial.println("  c m <0|1>       - Ativar/desativar monitor CAN");
        Serial.println("  c st            - Mostrar estatísticas CAN");
        Serial.println("  c sc            - Procurar nós ativos na rede");
        
        Serial.println("\nComandos de Streaming:");
        Serial.println("  s <var> <nó>    - Iniciar streaming de variável");
        Serial.println("  S <var> <nó>    - Parar streaming de variável");
        
        Serial.println("\nNota: <nó>=0 envia para todos os nós (broadcast)");
        Serial.println("ack");
        return;
    }
    
    //============================================================================
    // COMANDOS DO CONTROLADOR PI (APENAS LOCAIS)
    //============================================================================
    
    // "k <val>" => definir ganho (apenas local)
    else if (c0 == "k")
    {
        if (numTokens < 2)
        {
            Serial.println("err");
            return;
        }
    
        float val = tokens[1].toFloat();
    
        if (val < 0.0f)
        {
            Serial.println("err: O ganho deve ser positivo");
            return;
        }
    
        // Aplicar localmente (sem CAN) - set BOTH Kp and Ki to the same value
        pid.setGains(val, val);
        
        Serial.print("Ganhos definidos para Kp = Ki = ");
        Serial.println(val, 4);
        Serial.println("ack");
        return;
    }

    // "b <val>" => definir fator de ponderação (apenas local)
    else if (c0 == "b")
    {
        if (numTokens < 2)
        {
            Serial.println("err");
            return;
        }

        float val = tokens[1].toFloat();

        if (val < 0.0f || val > 1.0f)
        {
            Serial.println("err: O factor de ponderação deve estar entre 0 e 1");
            return;
        }

        // Aplicar localmente (sem CAN)
        pid.setSetpointWeight(val);
        
        Serial.print("Factor de ponderação definido para Beta = ");
        Serial.println(val, 4);
        Serial.println("ack");
        return;
    }

    
    
    //============================================================================
    // COMANDOS DE STREAMING
    //============================================================================
    
    // "s <x> <i>" => iniciar fluxo da variável em tempo real <x> para a secretária <i>
    else if (c0 == "s")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }

        String var = tokens[1];
        uint8_t targetNode = tokens[2].toInt();

        // Verificar se precisamos de reencaminhar este comando
        if (shouldForwardCommand(targetNode))
        {
            // Início de fluxo = tipo de controlo 11
            // Codificaremos o tipo de variável no campo de valor:
            // 'y'=0, 'u'=1, 'p'=2, etc.
            float varCode = 0; // Predefinido para 'y' (lux)

            if (var.equalsIgnoreCase("u"))
                varCode = 1;
            else if (var.equalsIgnoreCase("p"))
                varCode = 2;
            else if (var.equalsIgnoreCase("o"))
                varCode = 3;
            else if (var.equalsIgnoreCase("a"))
                varCode = 4;
            else if (var.equalsIgnoreCase("f"))
                varCode = 5;
            else if (var.equalsIgnoreCase("r"))
                varCode = 6;
            else if (var.equalsIgnoreCase("y"))
                varCode = 0;
            else if (var.equalsIgnoreCase("v"))
                varCode = 7;
            else if (var.equalsIgnoreCase("d"))
                varCode = 8;
            else if (var.equalsIgnoreCase("t"))
                varCode = 9;
            else if (var.equalsIgnoreCase("V"))
                varCode = 10;
            else if (var.equalsIgnoreCase("F"))
                varCode = 11;
            else if (var.equalsIgnoreCase("E"))
                varCode = 12;

            if (sendControlCommand(targetNode, 11, varCode))
            {
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha no reencaminhamento CAN");
            }
            return;
        }
        // Tratar localmente
        startStream(var, targetNode);
        return;
    }
    // "S <x> <i>" => parar fluxo
    else if (c0 == "S")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }

        String var = tokens[1];
        uint8_t targetNode = tokens[2].toInt();

        // Verificar se precisamos de reencaminhar este comando
        if (shouldForwardCommand(targetNode))
        {
            // Parar fluxo = tipo de controlo 12
            // Codificaremos o tipo de variável no campo de valor
            float varCode = 0; // Predefinido para 'y' (lux)

            if (var.equalsIgnoreCase("u"))
                varCode = 1;
            else if (var.equalsIgnoreCase("p"))
                varCode = 2;
            else if (var.equalsIgnoreCase("o"))
                varCode = 3;
            else if (var.equalsIgnoreCase("a"))
                varCode = 4;
            else if (var.equalsIgnoreCase("f"))
                varCode = 5;
            else if (var.equalsIgnoreCase("r"))
                varCode = 6;
            else if (var.equalsIgnoreCase("y"))
                varCode = 0;
            else if (var.equalsIgnoreCase("v"))
                varCode = 7;
            else if (var.equalsIgnoreCase("d"))
                varCode = 8;
            else if (var.equalsIgnoreCase("t"))
                varCode = 9;
            else if (var.equalsIgnoreCase("V"))
                varCode = 10;
            else if (var.equalsIgnoreCase("F"))
                varCode = 11;
            else if (var.equalsIgnoreCase("E"))
                varCode = 12;

            if (sendControlCommand(targetNode, 12, varCode))
            {
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha no reencaminhamento CAN");
            }
            return;
        }

        // Tratar localmente
        stopStream(var, targetNode);
        Serial.println("ack");
        return;
    }
    //============================================================================
    // COMANDOS DE MÉTRICAS
    //============================================================================

    // Se o primeiro token for 'g' => "g <sub> <i>"
    else if (c0 == "g")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }
        String subCommand = tokens[1];
        String originalCase = tokens[1];
        subCommand.toLowerCase();

        uint8_t targetNode = tokens[2].toInt();
        String idx = tokens[2];

        // Verificar se precisamos de reencaminhar este comando
        if (shouldForwardCommand(targetNode))
        {
            // Mapear o comando get para um tipo de mensagem de consulta CAN (usando código 20-30)
            uint8_t queryType = 20; // Tipo de consulta predefinido

            if (originalCase == "V")
                queryType = 20; // Erro de visibilidade
            else if (originalCase == "F")
                queryType = 21; // Cintilação
            else if (originalCase == "E")
                queryType = 22; // Energia
            else if (subCommand == "u")
                queryType = 23; // Duty cycle
            else if (subCommand == "o")
                queryType = 24; // Ocupação
            else if (subCommand == "a")
                queryType = 25; // Anti-windup
            else if (subCommand == "f")
                queryType = 26; // Controlo por feedback
            else if (subCommand == "r")
                queryType = 27; // Referência de iluminação
            else if (subCommand == "y")
                queryType = 28; // iluminação atual
            else if (subCommand == "p")
                queryType = 29; // Consumo de energia
            else if (subCommand == "t")
                queryType = 30; // Tempo decorrido
            else if (subCommand == "v")
                queryType = 31; // Tensão no LDR
            else if (subCommand == "d")
                queryType = 32; // iluminação externa
            else
            {
                Serial.println("err: Consulta de variável remota não suportada");
                return;
            }

            // Enviar a consulta para o nó remoto
            if (sendControlCommand(targetNode, queryType, 0.0f))
            {
                Serial.println("Consulta enviada para o nó " + String(targetNode));

                // Aguardar resposta com tempo limite
                unsigned long timeout = millis() + 500; // 500ms de tempo limite
                bool responseReceived = false;

                while (millis() < timeout && !responseReceived)
                {
                    can_frame frame;
                    if (readCANMessage(&frame) == MCP2515::ERROR_OK)
                    {
                        // Analisar mensagem e verificar se é uma resposta do nosso alvo
                        uint8_t msgType, destAddr, priority;
                        parseCANId(frame.can_id, msgType, destAddr, priority);

                        uint8_t senderNodeID = frame.data[0];

                        if (msgType == CAN_TYPE_RESPONSE && senderNodeID == targetNode && (destAddr == nodeID || destAddr == CAN_ADDR_BROADCAST))
                        {
                            // Extrair o valor da resposta e mostrar
                            float value = 0.0f;
                            value = bytesToFloat(&frame.data[2]);

                            // Formatar a resposta com base no tipo de consulta original
                            if (originalCase == "V" || originalCase == "F" || originalCase == "E")
                            {
                                Serial.print(originalCase);
                            }
                            else
                            {
                                Serial.print(subCommand);
                            }
                            Serial.print(" ");
                            Serial.print(idx);
                            Serial.print(" ");

                            // Formatar o valor com a precisão adequada
                            if (subCommand == "u" || originalCase == "F" || originalCase == "E")
                                Serial.println(value, 4);
                            else if (originalCase == "V" || subCommand == "y" || subCommand == "p" || subCommand == "d")
                                Serial.println(value, 2);
                            else if (subCommand == "v")
                                Serial.println(value, 3);
                            else if (subCommand == "o" || subCommand == "a" || subCommand == "f" || subCommand == "t")
                                Serial.println((int)value);
                            else
                                Serial.println(value);

                            responseReceived = true;
                        }
                    }
                    delay(5); // Pequeno atraso para evitar sobrecarregar o barramento CAN
                }

                if (!responseReceived)
                {
                    Serial.println("err: Sem resposta do nó " + String(targetNode));
                }
            }
            else
            {
                Serial.println("err: Falha ao enviar consulta para o nó " + String(targetNode));
            }
            return;
        }


        // Processar localmente
        // Comandos de métricas
        // "g V <i>" => "V <i> <val>" (Métrica de erro de visibilidade)
        if (originalCase == "V")
        {
            float V = computeVisibilityErrorFromBuffer();
            Serial.print("V ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(V, 2);
            return;
        }
        // "g F <i>" => "F <i> <val>" (Métrica de cintilação)
        else if (originalCase == "F")
        {
            float F = computeFlickerFromBuffer();
            Serial.print("F ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(F, 4);
            return;
        }
        // "g E <i>" => "E <i> <val>" (Métrica de energia)
        else if (originalCase == "E")
        {
            float E = computeEnergyFromBuffer();
            Serial.print("E ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(E, 4);
            return;
        }

        // "g u <i>" => "u <i> <val>"
        if (subCommand == "u")
        {
            Serial.print("u ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(dutyCycle, 4);
            return;
        }
        // "g o <i>" => "o <i> <val>"
        else if (subCommand == "o")
        {
            int occVal = occupancy ? 1 : 0;
            Serial.print("o ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(occVal);
            return;
        }
        // "g a <i>" => "a <i> <val>"
        else if (subCommand == "a")
        {
            int awVal = antiWindup ? 1 : 0;
            Serial.print("a ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(awVal);
            return;
        }
        // "g f <i>" => "f <i> <val>"
        else if (subCommand == "f")
        {
            int fbVal = feedbackControl ? 1 : 0;
            Serial.print("f ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(fbVal);
            return;
        }
        // "g r <i>" => "r <i> <val>"
        else if (subCommand == "r")
        {
            Serial.print("r ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(refIlluminance, 4);
            return;
        }
        // "g y <i>" => "y <i> <val>"
        else if (subCommand == "y")
        {
            float lux = readLux();
            Serial.print("y ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(lux, 2);
            return;
        }
        // "g v <i>" => medir nível de tensão no LDR => "v <i> <val>"
        else if (subCommand == "v")
        {
            float vLdr = getVoltageAtLDR();
            Serial.print("v ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(vLdr, 3);
            return;
        }
        // "g d <i>" => iluminação externa => "d <i> <val>"
        else if (subCommand == "d")
        {
            float dVal = getExternalIlluminance();
            Serial.print("d ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(dVal, 2);
            return;
        }
        // "g p <i>" => potência instantânea => "p <i> <val>"
        else if (subCommand == "p")
        {
            float pVal = getPowerConsumption();
            Serial.print("p ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(pVal, 2);
            return;
        }
        // "g t <i>" => tempo decorrido => "t <i> <val>"
        else if (subCommand == "t")
        {
            unsigned long sec = getElapsedTime();
            Serial.print("t ");
            Serial.print(idx);
            Serial.print(" ");
            Serial.println(sec);
            return;
        }
        // "g b <x> <i>" => "b <x> <i> <val1>,<val2>..."
        else if (subCommand == "b")
        {
            if (numTokens < 4)
            {
                Serial.println("err");
                return;
            }
            String xVar = tokens[2];
            int iDesk = tokens[3].toInt();
            String bufferData = getLastMinuteBuffer(xVar, iDesk);
            Serial.print("b ");
            Serial.print(xVar);
            Serial.print(" ");
            Serial.print(iDesk);
            Serial.print(" ");
            Serial.println(bufferData);
            return;
        }

        else
        {
            Serial.println("err");
            return;
        }
    }
    
    //============================================================================
    // COMANDOS CAN
    //============================================================================
    
    // Comandos CAN tratados se c0 == "c"
    // "c sd <destNode> <msgType> <value>" => Enviar uma mensagem CAN
    else if (c0 == "c" && tokens[1] == "sd")
    {
        if (numTokens < 5)
        {
            Serial.println("err");
            return;
        }

        uint8_t destNode = tokens[2].toInt();
        uint8_t msgType = tokens[3].toInt();
        float value = tokens[4].toFloat();

        bool success = false;

        if (msgType == 0)
        {                                                     // Mensagem de controlo
            success = sendControlCommand(destNode, 0, value); // Tipo 0 = setpoint
        }
        else if (msgType == 1)
        {                                                    // Leitura de sensor
            success = sendSensorReading(destNode, 0, value); // Tipo 0 = lux
        }
        else
        {
            Serial.println("err: Tipo de mensagem inválido");
            return;
        }

        if (success)
        {
            Serial.println("ack");
        }
        else
        {
            Serial.println("err: Falha no envio");
        }
        return;
    }
    // "c m <0|1>" => Ativar/desativar impressão de mensagens CAN recebidas
    else if (c0 == "c" && tokens[1] == "m")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }

        canMonitorEnabled = (tokens[2].toInt() == 1);

        Serial.print("Monitorização CAN ");
        Serial.println(canMonitorEnabled ? "ativada" : "desativada");
        Serial.println("ack");
        return;
    }
    // "c st" => Mostrar estatísticas de comunicação CAN
    else if (c0 == "c" && tokens[1] == "st")
    {
        uint32_t sent, received, errors;
        float avgLatency;
        getCANStats(sent, received, errors, avgLatency);

        Serial.println("Estatísticas CAN:");
        Serial.print("  ID do Nó: ");
        Serial.println(nodeID);
        Serial.print("  Mensagens enviadas: ");
        Serial.println(sent);
        Serial.print("  Mensagens recebidas: ");
        Serial.println(received);
        Serial.print("  Erros: ");
        Serial.println(errors);
        Serial.print("  Latência média: ");
        Serial.print(avgLatency);
        Serial.println(" us");
        Serial.println("ack");
        return;
    }
    // "c r" => Reset CAN statistics
    else if (c0 == "c" && tokens[1] == "r")
    {
        resetCANStats();
        Serial.println("Estatísticas CAN reiniciadas");
        Serial.println("ack");
        return;
    }
    // "c sc" => Scan for active nodes on the network
    else if (c0 == "c" && tokens[1] == "sc")
    {
        Serial.println("A procurar nós CAN ativos...");

        // Iremos rastrear quais nós respondem
        bool nodeFound[64] = {false};
        int foundCount = 0;

        // Enviar mensagens de ping para todos os endereços de nó possíveis
        for (uint8_t node = 1; node < 64; node++)
        {
            // Enviar uma mensagem especial de ping
            if (sendControlCommand(node, 3, 0))
            {
                // Dar algum tempo para o nó responder
                delay(50);

                // Processar quaisquer respostas que chegaram
                for (int i = 0; i < 5; i++)
                {
                    can_frame frame;
                    if (readCANMessage(&frame) == MCP2515::ERROR_OK)
                    {
                        uint8_t msgType, srcAddr, priority;
                        parseCANId(frame.can_id, msgType, srcAddr, priority);

                        if (!nodeFound[srcAddr])
                        {
                            nodeFound[srcAddr] = true;
                            foundCount++;
                        }
                    }
                    delay(10);
                }
            }
        }

        // Agora enviar uma mensagem de difusão para capturar qualquer um que perdemos
        sendControlCommand(CAN_ADDR_BROADCAST, 3, 0);
        delay(200);

        // Processar quaisquer respostas adicionais
        for (int i = 0; i < 20; i++)
        {
            can_frame frame;
            if (readCANMessage(&frame) == MCP2515::ERROR_OK)
            {
                uint8_t msgType, srcAddr, priority;
                parseCANId(frame.can_id, msgType, srcAddr, priority);

                if (!nodeFound[srcAddr])
                {
                    nodeFound[srcAddr] = true;
                    foundCount++;
                }
            }
            delay(10);
        }

        // Mostrar resultados
        Serial.print("Encontrados ");
        Serial.print(foundCount);
        Serial.println(" nós ativos:");

        for (uint8_t node = 1; node < 64; node++)
        {
            if (nodeFound[node])
            {
                Serial.print("  Nó ");
                Serial.println(node);
            }
        }

        Serial.println("Análise da rede concluída");
        Serial.println("ack");
        return;
    }
    // "c l <destNode> <count>" => Measure round-trip latency
    else if (c0 == "c" && tokens[1] == "l")
    {
        if (numTokens < 4)
        {
            Serial.println("err");
            return;
        }

        uint8_t destNode = tokens[2].toInt();
        int count = tokens[3].toInt();

        Serial.print("A medir latência de ida e volta para o nó ");
        Serial.print(destNode);
        Serial.print(" (");
        Serial.print(count);
        Serial.println(" amostras)");

        unsigned long totalLatency = 0;
        int successCount = 0;

        for (int i = 0; i < count; i++)
        {
            unsigned long startTime = micros();

            // Enviar pedido de eco (usando tipo de mensagem de controlo 2)
            if (sendControlCommand(destNode, 2, startTime))
            {
                // Aguardar resposta com tempo limite
                unsigned long timeout = millis() + 500; // 500ms de tempo limite
                bool responseReceived = false;

                while (millis() < timeout && !responseReceived)
                {
                    can_frame frame;
                    if (readCANMessage(&frame) == MCP2515::ERROR_OK)
                    {
                        // Analisar mensagem e verificar se é uma resposta de eco
                        uint8_t msgType, srcAddr, priority;
                        parseCANId(frame.can_id, msgType, srcAddr, priority);

                        if (msgType == CAN_TYPE_RESPONSE && srcAddr == destNode)
                        {
                            unsigned long endTime = micros();
                            unsigned long latency = endTime - startTime;
                            totalLatency += latency;
                            successCount++;
                            responseReceived = true;

                            Serial.print("Amostra ");
                            Serial.print(i + 1);
                            Serial.print(": ");
                            Serial.print(latency);
                            Serial.println(" us");
                        }
                    }
                }

                if (!responseReceived)
                {
                    Serial.print("Amostra ");
                    Serial.print(i + 1);
                    Serial.println(": Timeout");
                }
            }
            else
            {
                Serial.print("Amostra ");
                Serial.print(i + 1);
                Serial.println(": Falha no envio");
            }

            delay(100); // Aguardar entre amostras
        }

        Serial.println("Medição de latência concluída");
        if (successCount > 0)
        {
            float avgLatency = (float)totalLatency / successCount;
            Serial.print("Latência média de ida e volta: ");
            Serial.print(avgLatency, 2);
            Serial.println(" us");
        }
        else
        {
            Serial.println("Sem medições bem-sucedidas");
        }
        Serial.println("ack");
        return;
    }
   
    //============================================================================
    // COMANDOS DE ESTADO
    //============================================================================
    
    // "state <i> <state>" => definir estado do LED (off/unoccupied/occupied)
    else if (c0 == "st")
    {
        if (numTokens < 3)
        {
            Serial.println("err");
            return;
        }

        uint8_t targetNode = tokens[1].toInt();
        String stateStr = tokens[2];
        stateStr.toLowerCase();

        // Converter string de estado para valor para transmissão CAN
        int stateVal = 0;
        if (stateStr == "off")
            stateVal = 0;
        else if (stateStr == "unoccupied")
            stateVal = 1;
        else if (stateStr == "occupied")
            stateVal = 2;
        else
        {
            Serial.println("err");
            return;
        }

        // Verificar se precisamos de reencaminhar este comando
        if (shouldForwardCommand(targetNode))
        {
            // Reencaminhar para nó específico - tipo de controlo 13 = estado do LED
            if (sendControlCommand(targetNode, 13, stateVal))
            {
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha no reencaminhamento CAN");
            }
            return;
        }

        // Tratar o caso de difusão (targetNode = 0)
        if (targetNode == 0)
        {
            if (sendControlCommand(CAN_ADDR_BROADCAST, 13, stateVal))
            {
                // Também aplicar localmente, pois a difusão inclui este nó
                if (stateStr == "off")
                    changeState(STATE_OFF);
                else if (stateStr == "unoccupied")
                    changeState(STATE_UNOCCUPIED);
                else if (stateStr == "occupied")
                    changeState(STATE_OCCUPIED);
                Serial.println("ack");
            }
            else
            {
                Serial.println("err: Falha na difusão CAN");
            }
            return;
        }

        // Aplicar localmente
        if (stateStr == "off")
            changeState(STATE_OFF);
        else if (stateStr == "unoccupied")
            changeState(STATE_UNOCCUPIED);
        else if (stateStr == "occupied")
            changeState(STATE_OCCUPIED);
        Serial.println("ack");
        return;
    }
    Serial.println("ack");
    return;
}

/**
 * Lê e processa comandos recebidos pela porta série
 * Deve ser chamada ciclicamente no loop principal
 */
void processSerialCommands()
{
    if (Serial.available() > 0)
    {
        char c = Serial.read();

        if (c == '\n')
        {
            // Processar o comando completo
            processCommandLine(input);
            input = ""; // Limpar o buffer de entrada
        }
        else if (c != '\r') // Ignorar retornos de carro
        {
            // Adicionar caractere ao buffer
            input += c;
        }
    }
}