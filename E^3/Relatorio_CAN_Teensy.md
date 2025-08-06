# Relatório de Testes - Comunicação CAN entre Teensy 4.0

**Data:** 6 de Agosto de 2025  
**Projeto:** Comunicação CAN entre duas placas Teensy 4.0  
**Transcivers utilizados:** SN65HVD232D  

## 1. Objectivo

Estabelecer comunicação CAN básica entre duas placas Teensy 4.0 utilizando Transciverses SN65HVD232D para testar a transmissão e receção de mensagens.

## 2. Configuração de Hardware

### 2.1 Placas utilizadas

- 2x Teensy 4.0
- 2x Transcivers SN65HVD232D
- Resistências de terminação: 120Ω em cada extremidade do barramento

### 2.2 Ligações realizadas

**Transcivers SN65HVD232D (8 pinos):**

- Pin 1 (D) → Teensy Pin 22/Pin 1 (CAN_TX)
- Pin 2 (GND) → Terra
- Pin 3 (VCC) → 3.3V
- Pin 4 (R) → Teensy Pin 23/Pin 0 (CAN_RX)
- Pin 5 e 8 (NC) → Não conectado
- Pin 6 (CANL) → CANL do outro Transcivers + resistência 120Ω
- Pin 7 (CANH) → CANH do outro Transcivers + resistência 120Ω

## 3. Testes Realizados

### 3.1 Teste de Loopback Interno

**Resultado:** ✅ **SUCESSO**

Ambas as placas Teensy 4.0 passaram no teste de loopback interno tanto no CAN1 (pins 22/23) como no CAN2 (pins 0/1), confirmando que os controladores CAN estão funcionais.

### 3.2 Teste de Comunicação Externa

#### 3.2.1 Configurações testadas

- **Velocidades de transmissão:** 125000, 250000, 500000 bps
- **Controladores CAN:** CAN1 (pins 22/23) e CAN2 (pins 0/1)
- **Configurações de software:** Com e sem FIFO, com diferentes configurações

#### 3.2.2 Comportamento observado

**Transmissor:**

- Inicialmente reporta "Message sent successfully!"
- Após alguns segundos, alterna entre "Sent!" e "Failed!"

**Receptor:**

- Nunca recebe mensagens
- Mantém-se sempre em estado "waiting for messages"
- Não apresenta erros de configuração

## 4. Análise dos Problemas

### 4.1 Problemas Persistentes

**Principal suspeita: Configuração dos Transciverses SN65HVD232D**

1. **Alimentação:** 3.3V estável confirmada
2. **Terminação:** Resistências de 120Ω confirmadas em ambas as extremidades
3. **Ligações do barramento:** CANH-CANH e CANL-CANL verificadas

### 4.3 Comportamento Estranho

O facto de a função `can.write()` reportar sucesso mesmo sem Transciverses conectados sugere que:

- A biblioteca FlexCAN_T4 não implementa verificação completa de transmissão CAN
- A função apenas confirma que a mensagem foi colocada na fila interna do controlador
- Não há verificação de acknowledgment do barramento CAN

## 5. Testes Adicionais Realizados

### 5.1 Simplificação do Código

Removemos todas as configurações avançadas (FIFO, mailboxes) para usar apenas as funções básicas `read()` e `write()`.

### 5.2 Teste de Múltiplas Velocidades

Testámos 125k, 250k e 500k bps sem sucesso na comunicação externa.

### 5.3 Verificação de Hardware

- Continuidade das ligações verificada
- Alimentação dos Transciverses confirmada
- Posicionamento correto das resistências de terminação

## 6. Conclusões

### 6.1 Estado Atual

- **Controladores CAN das Teensy:** Funcionais (confirmado por loopback)
- **Software:** Correto e simplificado
- **Comunicação externa:** Não funcional

### 6.2 Possíveis Causas do Problema

1. **Transciverses defeituosos:** Um ou ambos os SN65HVD232D podem estar danificados
2. **Configuração específica do SN65HVD232D:** Pode necessitar de configuração adicional não documentada
3. **Incompatibilidade de tensões:** Apesar de ambos operarem a 3.3V
4. **Problema de temporização:** O Transcivers pode necessitar de delays adicionais

### 6.3 Próximos Passos Recomendados

1. **Teste com osciloscópio:** Verificar se há actividade nos pins CANH e CANL durante transmissão
2. **Teste com Transciverses alternativos:** Experimentar MCP2551 ou similar
3. **Teste de tensão diferencial:** Medir tensão entre CANH e CANL (deve ser ~2V em repouso)
4. **Teste com uma só placa:** Conectar ambos os Transciverses à mesma Teensy para testar comunicação local

## 7. Código Final Utilizado

### 7.1 Transmissor (CAN2)

```cpp
#include <FlexCAN_T4.h>
FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> can2;
CAN_message_t msg;

void setup() {
    Serial.begin(115200);
    delay(1000);
    can2.begin();
    can2.setBaudRate(125000);
    Serial.println("CAN2 sender ready");
}

void loop() {
    msg.id = 0x123;
    msg.len = 1;
    msg.buf[0] = 42;
    
    if (can2.write(msg)) {
        Serial.println("Sent!");
    } else {
        Serial.println("Failed!");
    }
    delay(1000);
}
```

### 7.2 Receptor (CAN2)

```cpp
#include <FlexCAN_T4.h>
FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> can2;
CAN_message_t msg;

void setup() {
    Serial.begin(115200);
    delay(1000);
    can2.begin();
    can2.setBaudRate(125000);
    Serial.println("CAN2 receiver ready");
}

void loop() {
    if (can2.read(msg)) {
        Serial.print("Got: ");
        Serial.println(msg.buf[0]);
    }
    delay(10);
}
```

---

**Nota:** Este relatório documenta uma configuração que deveria funcionar teoricamente, mas que apresenta problemas práticos que necessitam de investigação adicional ao nível do hardware dos Transciverses SN65HVD232D.
