/**
 * DataLogger.h - Interface para registo e análise de dados
 *
 * Este ficheiro define funções e estruturas para registo e análise de dados:
 * - Armazenamento de medições em buffer circular
 * - Cálculo de métricas de desempenho (energia, flicker, erro de visibilidade)
 * - Streaming de dados em tempo real
 * - Exportação de dados para análise externa
 * - Funções de consulta para acesso ao historial de dados
 */

#pragma once
#include <Arduino.h>

//============================================================================
// ESTRUTURAS DE DADOS
//============================================================================

/**
 * Estrutura para entradas no buffer circular
 * Armazena uma medição com timestamp e valores associados
 */
struct LogEntry
{
    unsigned long timestamp; // Tempo da medição em milissegundos
    float lux;               // Iluminação medida em lux
    float duty;              // Ciclo de trabalho do LED [0..1]
    float setpoint;          // Setpoint atual em lux
    float flicker;           // Medida de cintilação instantânea
    float jitter;            // Jitter em microssegundos
};

//============================================================================
// INICIALIZAÇÃO E GESTÃO DO BUFFER
//============================================================================

/**
 * Inicializa o sistema de armazenamento
 * Prepara o buffer circular e as estruturas de dados necessárias
 */
void initStorage();

/**
 * Regista um novo ponto de dados no buffer circular
 *
 * @param timestamp Tempo da medição em milissegundos
 * @param lux Valor da iluminação medida em lux
 * @param duty Ciclo de trabalho do LED [0..1]
 */
void logData(unsigned long timestamp, float lux, float duty);

/**
 * Exporta o conteúdo completo do buffer para a porta série
 * Útil para análise detalhada de dados em ferramentas externas
 */
void dumpBufferToSerial();

/**
 * Exporta 1000 amostras do buffer a cada 10 pontos no formato especificado
 * Imprime timestamps, lux, duty cycles e setpoints em linhas separadas
 */
void dumpSampledBufferToSerial();

/**
 * Obtém acesso direto ao buffer de registos
 *
 * @return Ponteiro para o array de entradas de registo
 */
LogEntry *getLogBuffer();

/**
 * Obtém o número de entradas válidas no buffer
 *
 * @return Número de entradas de dados registadas
 */
int getLogCount();

/**
 * Verifica se o buffer circular está cheio
 *
 * @return true se o buffer estiver cheio, false caso contrário
 */
bool isBufferFull();

/**
 * Obtém o índice atual de escrita no buffer
 *
 * @return Índice da posição atual de escrita
 */
int getCurrentIndex();

//============================================================================
// CÁLCULO DE MÉTRICAS
//============================================================================

/**
 * Calcula e imprime todas as métricas de desempenho
 * Inclui energia consumida, flicker e erro de visibilidade
 */
void computeAndPrintMetrics();

/**
 * Calcula o consumo energético a partir dos dados no buffer
 *
 * @return Valor do consumo energético total (unidades relativas)
 */
float computeEnergyFromBuffer();

/**
 * Calcula o erro de visibilidade a partir dos dados no buffer
 * Métrica de qualidade baseada na estabilidade da iluminação
 *
 * @return Valor do erro de visibilidade (menor é melhor)
 */
float computeVisibilityErrorFromBuffer();

/**
 * Calcula o índice de cintilação (flicker) a partir dos dados no buffer
 * Métrica que quantifica variações rápidas na iluminação
 *
 * @return Valor do índice de cintilação (menor é melhor)
 */
float computeFlickerFromBuffer();

//============================================================================
// FUNÇÕES DE STREAMING
//============================================================================

/**
 * Inicia um stream de dados em tempo real
 *
 * @param var Variável a ser transmitida ("y"=lux, "u"=duty, etc.)
 * @param index Identificador da fonte de dados (índice do nó)
 */
void startStream(const String &var, int index);

/**
 * Interrompe um stream de dados em tempo real
 *
 * @param var Variável cujo stream deve ser interrompido
 * @param index Identificador da fonte de dados (índice do nó)
 */
void stopStream(const String &var, int index);

/**
 * Processa os streams ativos e envia dados quando necessário
 * Deve ser chamada ciclicamente no loop principal
 */
void handleStreaming();

/**
 * Obtém os dados do último minuto de uma variável específica
 *
 * @param var Variável desejada ("y"=lux, "u"=duty, etc.)
 * @param index Identificador da fonte de dados (índice do nó)
 * @return String com valores separados por vírgula
 */
String getLastMinuteBuffer(const String &var, int index);