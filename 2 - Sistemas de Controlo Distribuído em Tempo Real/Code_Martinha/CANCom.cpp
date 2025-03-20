/**
 * CANCom.cpp - Implementação da comunicação CAN-BUS
 *
 * Este ficheiro implementa todas as funcionalidades de comunicação CAN-BUS:
 * - Configuração e inicialização do controlador MCP2515
 * - Formatação e parsing de mensagens CAN
 * - Envio de leituras, comandos e heartbeats
 * - Processamento de mensagens recebidas
 * - Estatísticas de comunicação
 */

//============================================================================
// Ficheiros Incluídos
//============================================================================

#include "CANCom.h"
#include "Configuration.h"
#include <Arduino.h>
#include <SPI.h>
#include <mcp2515.h>
#include <pico/unique_id.h>
#include "StateManager.h"
#include "SensorReader.h"
#include "LEDController.h"
#include "DataLogger.h"

//============================================================================
// DEFINIÇÕES DE HARDWARE E VARIÁVEIS GLOBAIS
//============================================================================

// Instância do controlador CAN-BUS
MCP2515 can0(spi0, CAN_CS_PIN, CAN_MOSI_PIN, CAN_MISO_PIN, CAN_SCK_PIN, 10000000);

// Definição do tipo de callback para processamento de mensagens
typedef void (*CANMessageCallback)(const can_frame &msg);

// Variável para armazenar o callback registado
static CANMessageCallback messageCallback = nullptr;

// Variáveis para guardar estatísticas de comunicação
static uint32_t msgReceived = 0;             // Contador de mensagens recebidas
static uint32_t msgErrors = 0;               // Contador de erros de comunicação
static unsigned long lastLatencyMeasure = 0; // Timestamp para medir latência
static unsigned long totalLatency = 0;       // Soma de todas as latências (para média)
static uint32_t latencySamples = 0;          // Número de amostras de latência
static uint32_t msgSent = 0;                 // Contador de mensagens enviadas

//============================================================================
// INICIALIZAÇÃO E CONFIGURAÇÃO
//============================================================================

/**
 * Inicializa o controlador CAN e atribui um ID único ao nó
 */
void initCANComm()
{
    // Inicialização do barramento SPI para comunicação com o MCP2515
    SPI.begin();

    // Reinicialização do controlador
    can0.reset();

    // Configuração do bitrate para 1Mbps
    can0.setBitrate(CAN_1000KBPS);

    // Definição do modo de operação normal
    can0.setNormalMode();

    Serial.println("CANComm: CAN inicializado em modo normal");

    // Geração de ID de nó único baseado no ID de hardware da placa
    pico_unique_board_id_t board_id;
    pico_get_unique_board_id(&board_id);
    nodeID = board_id.id[7] & 0x3F; // Utilizar os últimos 6 bits para ID (1-63)

    // Evitar ID 0 (reservado para broadcast)
    if (nodeID == 0)
        nodeID = 1;

    Serial.print("CANComm: ID de nó atribuído: ");
    Serial.println(nodeID);
}

//============================================================================
// FUNÇÕES UTILITÁRIAS PARA MANIPULAÇÃO DE IDs E DADOS
//============================================================================

/**
 * Constrói um ID CAN a partir dos seus componentes
 * @param msgType Tipo de mensagem (0-7)
 * @param destAddr Endereço de destino (0-63)
 * @param priority Prioridade (0-3)
 * @return ID CAN formatado
 */
uint32_t buildCANId(uint8_t msgType, uint8_t destAddr, uint8_t priority)
{
    return ((uint32_t)msgType << 8) | ((uint32_t)destAddr << 2) | priority;
}

/**
 * Extrai os componentes de um ID CAN
 * @param canId ID CAN completo
 * @param msgType Tipo de mensagem extraído
 * @param destAddr Endereço de destino extraído
 * @param priority Prioridade extraída
 */
void parseCANId(uint32_t canId, uint8_t &msgType, uint8_t &destAddr, uint8_t &priority)
{
    msgType = (canId >> 8) & 0x07;
    destAddr = (canId >> 2) & 0x3F;
    priority = canId & 0x03;
}

/**
 * Converte um float em representação de bytes
 * @param value Valor float a converter
 * @param bytes Array de 4 bytes para armazenar o resultado
 */
void floatToBytes(float value, uint8_t *bytes)
{
    memcpy(bytes, &value, 4);
}

/**
 * Converte bytes de volta para float
 * @param bytes Array de 4 bytes a converter
 * @return Valor float resultante
 */
float bytesToFloat(const uint8_t *bytes)
{
    float value;
    memcpy(&value, bytes, 4);
    return value;
}

//============================================================================
// FUNÇÕES PARA ENVIO DE MENSAGENS
//============================================================================

/**
 * Envia uma leitura de sensor pela rede CAN
 * @param destAddr Endereço do nó destino
 * @param sensorType Tipo de sensor (0=lux, 1=duty, etc)
 * @param value Valor lido
 * @return true se enviado com sucesso
 */
bool sendSensorReading(uint8_t destAddr, uint8_t sensorType, float value)
{
    can_frame frame;

    // Construção do ID com prioridade normal
    frame.can_id = buildCANId(CAN_TYPE_SENSOR, destAddr, CAN_PRIO_NORMAL);
    frame.can_dlc = 8; // Comprimento fixo de 8 bytes

    // Montagem da carga útil
    frame.data[0] = nodeID;     // ID do nó de origem
    frame.data[1] = sensorType; // Tipo de sensor

    // Conversão do valor float para bytes
    floatToBytes(value, &frame.data[2]);

    // Adição de timestamp para sincronização e deteção de atrasos
    uint16_t timestamp = (uint16_t)(millis() & 0xFFFF);
    frame.data[6] = timestamp & 0xFF;
    frame.data[7] = (timestamp >> 8) & 0xFF;

    // Envio da mensagem e gestão de estatísticas
    MCP2515::ERROR result = sendCANMessage(frame);

    if (result == MCP2515::ERROR_OK)
    {
        msgSent++;
        return true;
    }
    else
    {
        msgErrors++;
        return false;
    }
}

/**
 * Envia um comando de controlo para outro nó
 * @param destAddr Endereço do nó destino
 * @param controlType Tipo de comando de controlo
 * @param value Valor do parâmetro de controlo
 * @return true se enviado com sucesso
 */
bool sendControlCommand(uint8_t destAddr, uint8_t controlType, float value)
{
    can_frame frame;

    // Comandos de controlo têm prioridade alta na rede
    frame.can_id = buildCANId(CAN_TYPE_CONTROL, destAddr, CAN_PRIO_HIGH);
    frame.can_dlc = 8; // Comprimento fixo de 8 bytes

    // Montagem da carga útil
    frame.data[0] = nodeID;      // ID do nó de origem
    frame.data[1] = controlType; // Tipo de comando

    // Conversão do valor float para bytes
    floatToBytes(value, &frame.data[2]);

    // Número de sequência para deteção de perda de pacotes
    static uint16_t seqNum = 0;
    frame.data[6] = seqNum & 0xFF;
    frame.data[7] = (seqNum >> 8) & 0xFF;
    seqNum++;

    // Envio da mensagem e gestão de estatísticas
    MCP2515::ERROR result = sendCANMessage(frame);

    if (result == MCP2515::ERROR_OK)
    {
        msgSent++;
        return true;
    }
    else
    {
        msgErrors++;
        return false;
    }
}

/**
 * Envia uma resposta a uma consulta
 * @param destNode Endereço do nó destino
 * @param value Valor da resposta
 * @return true se enviado com sucesso
 */
bool sendQueryResponse(uint8_t destNode, float value)
{
    can_frame frame;

    // Construção da mensagem com formato consistente
    frame.can_id = buildCANId(CAN_TYPE_RESPONSE, destNode, CAN_PRIO_NORMAL);
    frame.can_dlc = 8;

    // Montagem da carga útil
    frame.data[0] = nodeID; // ID do nó que está a responder
    frame.data[1] = 2;      // Tipo 2 = resposta de consulta

    // Conversão do valor float para bytes
    floatToBytes(value, &frame.data[2]);

    // Campos reservados para uso futuro
    frame.data[6] = 0;
    frame.data[7] = 0;

    // Envio da mensagem
    MCP2515::ERROR result = sendCANMessage(frame);

    // Log de Debug
    Serial.print("DEBUG: Enviada resposta de consulta para o nó ");
    Serial.print(destNode);
    Serial.print(", valor: ");
    Serial.println(value);

    // Atualização de estatísticas
    if (result == MCP2515::ERROR_OK)
    {
        msgSent++;
        return true;
    }
    else
    {
        msgErrors++;
        return false;
    }
}

/**
 * Envia um heartbeat para informar presença na rede
 * @return true se enviado com sucesso
 */
bool sendHeartbeat()
{
    can_frame frame;

    // Heartbeats são mensagens de difusão com baixa prioridade
    frame.can_id = buildCANId(CAN_TYPE_HEARTBEAT, CAN_ADDR_BROADCAST, CAN_PRIO_LOW);
    frame.can_dlc = 6; // Comprimento específico de 6 bytes

    // Montagem da carga útil
    frame.data[0] = nodeID; // ID do nó emissor

    // Empacotamento de flags de estado em bits individuais
    uint8_t statusFlags = 0;
    if (feedbackControl)
        statusFlags |= 0x01; // bit 0 = modo de controlo com feedback
    if (occupancy)
        statusFlags |= 0x02; // bit 1 = estado de ocupação
    frame.data[1] = statusFlags;

    // Tempo de atividade do nó em segundos (32 bits)
    uint32_t uptime = getElapsedTime();
    frame.data[2] = uptime & 0xFF; // byte menos significativo
    frame.data[3] = (uptime >> 8) & 0xFF;
    frame.data[4] = (uptime >> 16) & 0xFF;
    frame.data[5] = (uptime >> 24) & 0xFF; // byte mais significativo

    // Envio da mensagem
    MCP2515::ERROR result = sendCANMessage(frame);

    // Atualização de estatísticas
    if (result == MCP2515::ERROR_OK)
    {
        msgSent++;
        return true;
    }
    else
    {
        msgErrors++;
        return false;
    }
}

//============================================================================
// PROCESSAMENTO DE MENSAGENS RECEBIDAS
//============================================================================

/**
 * Processa uma mensagem CAN recebida
 * @param msg Estrutura com a mensagem CAN recebida
 */
void processIncomingMessage(const can_frame &msg)
{
    // Extração dos componentes do ID CAN
    uint8_t msgType, destAddr, priority;
    parseCANId(msg.can_id, msgType, destAddr, priority);

    // Debug condicional
    if (canMonitorEnabled)
    {
        Serial.print("DEBUG: Recebida mensagem CAN, tipo ");
        Serial.print(msgType);
        Serial.print(", origem ");
        Serial.print(msg.data[0]);
    }

    // Filtro de endereço - só processar se for para este nó ou broadcast
    if (destAddr != nodeID && destAddr != CAN_ADDR_BROADCAST)
    {
        return; // Ignorar mensagens não destinadas a este nó
    }

    // Extração do ID do nó de origem
    uint8_t sourceNode = msg.data[0];

    // Processamento baseado no tipo de mensagem
    switch (msgType)
    {
    //-------------------------------------------------------------------------
    // Mensagem de dados de sensor
    //-------------------------------------------------------------------------
    case CAN_TYPE_SENSOR:
    {
        // Extração dos campos da mensagem
        uint8_t sensorType = msg.data[1];
        float value = bytesToFloat(&msg.data[2]);
        uint16_t timestamp = ((uint16_t)msg.data[7] << 8) | msg.data[6];

        // Log se monitorização ativa
        if (canMonitorEnabled)
        {
            Serial.print("CAN: Nó ");
            Serial.print(sourceNode);
            Serial.print(" sensor ");
            Serial.print(sensorType);
            Serial.print(" = ");
            Serial.print(value);
            Serial.print(" (ts: ");
            Serial.print(timestamp);
            Serial.println(")");
        }
        break;
    }

    //-------------------------------------------------------------------------
    // Mensagem de heartbeat
    //-------------------------------------------------------------------------
    case CAN_TYPE_HEARTBEAT:
    {
        // Extração dos campos da mensagem
        uint8_t statusFlags = msg.data[1];
        uint32_t uptime = ((uint32_t)msg.data[5] << 24) |
                          ((uint32_t)msg.data[4] << 16) |
                          ((uint32_t)msg.data[3] << 8) |
                          msg.data[2];

        // Log se monitorização ativa
        if (canMonitorEnabled)
        {
            Serial.print("CAN: Nó ");
            Serial.print(sourceNode);
            Serial.print(" heartbeat, tempo de atividade ");
            Serial.print(uptime);
            Serial.print("s, flags: ");
            Serial.println(statusFlags, BIN);
        }
        break;
    }

    //-------------------------------------------------------------------------
    // Mensagem de comando de controlo
    //-------------------------------------------------------------------------
    case CAN_TYPE_CONTROL:
    {
        // Extração dos campos comuns da mensagem
        uint8_t controlType = msg.data[1];
        float value = bytesToFloat(&msg.data[2]);
        uint16_t sequence = ((uint16_t)msg.data[7] << 8) | msg.data[6];

        // Log se monitorização ativa
        if (canMonitorEnabled)
        {
            Serial.print("CAN: Nó ");
            Serial.print(sourceNode);
            Serial.print(" controlo ");
            Serial.print(controlType);
            Serial.print(" = ");
            Serial.print(value);
            Serial.print(" (seq: ");
            Serial.print(sequence);
            Serial.println(")");
        }

        // Processamento por tipo específico de comando

        // -----------------------------------------------
        // Comandos de configuração de parâmetros
        // -----------------------------------------------
        if (controlType == 0)
        { // Ajuste de setpoint
            setpointLux = value;
            if (canMonitorEnabled)
            {
                Serial.print("CAN: A definir setpoint para ");
                Serial.println(value);
            }
        }
        else if (controlType == 2)
        { // Pedido de eco (teste de conectividade)
            // Responder com o mesmo valor para medir RTT
            can_frame response;
            response.can_id = buildCANId(CAN_TYPE_RESPONSE, sourceNode, CAN_PRIO_HIGH);
            response.can_dlc = 8;
            response.data[0] = nodeID;
            response.data[1] = 0;                   // Tipo 0 = resposta de eco
            floatToBytes(value, &response.data[2]); // Eco do valor recebido
            response.data[6] = msg.data[6];         // Preservar sequência
            response.data[7] = msg.data[7];
            sendCANMessage(response);
        }
        else if (controlType == 3)
        { // Ping/descoberta de nós na rede
            // Responder para identificação na rede
            can_frame response;
            response.can_id = buildCANId(CAN_TYPE_RESPONSE, sourceNode, CAN_PRIO_NORMAL);
            response.can_dlc = 8;
            response.data[0] = nodeID;
            response.data[1] = 1;               // Tipo 1 = resposta a descoberta
            floatToBytes(0, &response.data[2]); // Valor 0 (não utilizado)
            response.data[6] = 0;
            response.data[7] = 0;
            sendCANMessage(response);
        }
        else if (controlType == 4)
        { // Definir duty cycle
            setLEDDutyCycle(value);
            if (canMonitorEnabled)
            {
                Serial.print("CAN: A definir duty cycle para ");
                Serial.println(value);
            }
        }
        else if (controlType == 5)
        { // Definir percentagem do LED
            setLEDPercentage(value);
            if (canMonitorEnabled)
            {
                Serial.print("CAN: A definir percentagem do LED para ");
                Serial.println(value);
            }
        }
        else if (controlType == 6)
        { // Definir potência do LED em watts
            setLEDPower(value);
            if (canMonitorEnabled)
            {
                Serial.print("CAN: A definir potência do LED para ");
                Serial.println(value);
            }
        }
        else if (controlType == 7)
        { // Definir ocupação
            occupancy = (value != 0.0f);
            if (canMonitorEnabled)
            {
                Serial.print("CAN: A definir ocupação para ");
                Serial.println(occupancy ? "verdadeiro" : "falso");
            }
        }
        else if (controlType == 8)
        { // Definir anti-windup
            antiWindup = (value != 0.0f);
            if (canMonitorEnabled)
            {
                Serial.print("CAN: A definir anti-windup para ");
                Serial.println(antiWindup ? "verdadeiro" : "falso");
            }
        }
        else if (controlType == 9)
        { // Definir controlo por feedback
            feedbackControl = (value != 0.0f);
            if (canMonitorEnabled)
            {
                Serial.print("CAN: A definir controlo por feedback para ");
                Serial.println(feedbackControl ? "verdadeiro" : "falso");
            }
        }
        else if (controlType == 10)
        { // iluminação de referência
            refIlluminance = value;
            setpointLux = value;

            if (canMonitorEnabled)
            {
                Serial.print("CAN: A definir iluminação de referência para ");
                Serial.println(value);
            }
        }
        // -----------------------------------------------
        // Comandos de gestão de streams de dados
        // -----------------------------------------------
        else if (controlType == 11 || controlType == 12)
        { // Gestão de streams de dados
            // Início (11) ou paragem (12) de stream

            // Extração do código da variável a partir do valor float
            int varCode = (int)value;
            String var = "y"; // Variável padrão (iluminação)

            // Mapeamento de códigos para variáveis
            if (varCode == 1)
                var = "u"; // Sinal de controlo
            else if (varCode == 2)
                var = "p"; // Potência
            else if (varCode == 3)
                var = "o"; // Ocupação
            else if (varCode == 4)
                var = "a"; // Anti-windup
            else if (varCode == 5)
                var = "f"; // Feedback
            else if (varCode == 6)
                var = "r"; // Referência
            else if (varCode == 7)
                var = "v"; // Visibilidade
            else if (varCode == 8)
                var = "d"; // Duty cycle
            else if (varCode == 9)
                var = "t"; // Tempo
            else if (varCode == 10)
                var = "V"; // Tensão
            else if (varCode == 11)
                var = "F"; // Flicker
            else if (varCode == 12)
                var = "E"; // Energia

            // Iniciar ou parar stream conforme o tipo de comando
            if (controlType == 11)
                startStream(var, sourceNode);
            else
                stopStream(var, sourceNode);
        }
        else if (controlType == 13)
        { // Estado do LED
            int stateVal = (int)value;
            if (stateVal == 0)
                changeState(STATE_OFF);
            else if (stateVal == 1)
                changeState(STATE_UNOCCUPIED);
            else if (stateVal == 2)
                changeState(STATE_OCCUPIED);
        }
        // -----------------------------------------------
        // Comandos de consulta de valores
        // -----------------------------------------------
        else if (controlType >= 20 && controlType <= 32)
        {
            // Responder a consultas de diferentes variáveis
            float responseValue = 0.0f;

            // Determinar o valor solicitado com base no tipo de consulta
            switch (controlType)
            {
            case 20: // Erro de visibilidade
                responseValue = computeVisibilityErrorFromBuffer();
                break;
            case 21: // Flicker
                responseValue = computeFlickerFromBuffer();
                break;
            case 22: // Energia
                responseValue = computeEnergyFromBuffer();
                break;
            case 23: // Duty cycle
                responseValue = dutyCycle;
                break;
            case 24: // Ocupação
                responseValue = occupancy ? 1.0f : 0.0f;
                break;
            case 25: // Anti-windup
                responseValue = antiWindup ? 1.0f : 0.0f;
                break;
            case 26: // Controlo por feedback
                responseValue = feedbackControl ? 1.0f : 0.0f;
                break;
            case 27: // iluminação de referência
                responseValue = refIlluminance;
                break;
            case 28: // iluminação atual
                Serial.println("Consulta para iluminação atual");
                responseValue = readLux();
                break;
            case 29: // Consumo de potência
                responseValue = getPowerConsumption();
                break;
            case 30: // Tempo decorrido
                responseValue = getElapsedTime();
                break;
            case 31: // Tensão no LDR
                responseValue = getVoltageAtLDR();
                break;
            case 32: // iluminação externa
                responseValue = getExternalIlluminance();
                break;
            default:
                return; // Tipo de consulta desconhecido
            }

            // Enviar resposta com o valor solicitado
            sendQueryResponse(sourceNode, responseValue);
        }
        break;
    }
    } // fim do switch
}

//============================================================================
// LOOP PRINCIPAL DE COMUNICAÇÃO CAN
//============================================================================

/**
 * Loop principal de processamento CAN, a ser chamado ciclicamente
 */
void canCommLoop()
{
    // Verificar se existem mensagens para receber
    can_frame msg;
    if (can0.readMessage(&msg) == MCP2515::ERROR_OK)
    {
        // Atualizar estatísticas
        msgReceived++;

        // Processar a mensagem recebida
        processIncomingMessage(msg);

        // Executar callback externo se registado
        if (messageCallback)
        {
            messageCallback(msg);
        }
    }
}

//============================================================================
// FUNÇÕES DE BAIXO NÍVEL E GESTÃO DE ESTATÍSTICAS
//============================================================================

/**
 * Envia uma mensagem CAN formatada
 * @param frame Estrutura com a mensagem a enviar
 * @return Código de erro da operação
 */
MCP2515::ERROR sendCANMessage(const can_frame &frame)
{
    // Marca tempo para medição de latência
    lastLatencyMeasure = micros();

    // Envio efetivo da mensagem através do controlador MCP2515
    MCP2515::ERROR err = can0.sendMessage(&frame);

    // Atualização de estatísticas de latência se envio bem sucedido
    if (err == MCP2515::ERROR_OK)
    {
        unsigned long latency = micros() - lastLatencyMeasure;
        totalLatency += latency;
        latencySamples++;
    }

    return err;
}

/**
 * Lê uma mensagem CAN do controlador
 * @param frame Estrutura onde a mensagem será armazenada
 * @return Código de erro da operação
 */
MCP2515::ERROR readCANMessage(struct can_frame *frame)
{
    return can0.readMessage(frame);
}

/**
 * Regista um callback para processamento externo de mensagens
 * @param callback Função a ser chamada para cada mensagem recebida
 */
void setCANMessageCallback(CANMessageCallback callback)
{
    messageCallback = callback;
}

/**
 * Obtém estatísticas de comunicação CAN
 * @param sent Número de mensagens enviadas
 * @param received Número de mensagens recebidas
 * @param errors Número de erros ocorridos
 * @param avgLatency Latência média em microsegundos
 */
void getCANStats(uint32_t &sent, uint32_t &received, uint32_t &errors, float &avgLatency)
{
    sent = msgSent;
    received = msgReceived;
    errors = msgErrors;
    avgLatency = latencySamples > 0 ? (float)totalLatency / latencySamples : 0.0f;
}

/**
 * Reinicia os contadores de estatísticas
 */
void resetCANStats()
{
    msgSent = 0;
    msgReceived = 0;
    msgErrors = 0;
    totalLatency = 0;
    latencySamples = 0;
}
