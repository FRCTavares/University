#include <Arduino.h>
#include <math.h>

// Constantes do circuito
#define VCC 3.3                    // Tensão de alimentação (volts)
#define MY_ADC_RESOLUTION 4095.0   // Resolução do ADC (12 bits: 0 a 4095)
#define FIXED_RESISTOR 10000.0     // Resistor fixo do divisor de tensão (10 kΩ)

// Parâmetros calibrados para a equação do LDR (baseados no datasheet PGM5659D)
// Relação: log10(R_LDR) = m · log10(LUX) + b
// Exemplo: para R10 ≈ 225 kΩ e γ ≈ 0.8, temos:
#define LDR_M -0.8               
#define LDR_B 6.15

// Constantes para calibração do ganho (se necessário)
const float G = 1.0; // Fator de ganho (ajustar experimentalmente)
const float d = 0;   // Offset (ajustar experimentalmente)

// Número de pontos de calibração (por exemplo, 11 pontos de 0 a 255)
const int NUM_PONTOS = 11;

// Vetores para armazenar os valores de duty cycle (u) e os lux medidos (L)
float dutyCycles[NUM_PONTOS];
float luxValues[NUM_PONTOS];

// Pino do LED (PWM) e do sensor LDR
#define LED_PIN 15    // Pino PWM a controlar o LED
#define SENSOR_PIN A0 // Pino analógico para leitura do LDR

void setup() {
  Serial.begin(115200);
  
  // Configura o ADC para 12 bits, se suportado
  #if defined(analogReadResolution)
  analogReadResolution(12);
  #endif

  Serial.println("Início da calibração com varredura do PWM");
  Serial.println("DutyCycle,PWM;Lux"); // Cabeçalho CSV para análise
  
  // Aguarda estabilização do sistema
  delay(2000);

  // Varredura: para cada ponto de calibração, varia o duty cycle e regista a leitura
  for (int i = 0; i < NUM_PONTOS; i++) {
    // Calcula o valor do duty cycle (u) de forma linear entre 0 e 255
    float u = (255.0 * i) / (NUM_PONTOS - 1);
    dutyCycles[i] = u;
    
    // Define o PWM do LED (utilizando analogWrite, compatível com a biblioteca Erle Phill Hower)
    analogWrite(LED_PIN, (int)u);
    
    // Aguarda 1 segundo para estabilização do LED e do sensor
    delay(1000);
    
    // Lê o valor analógico do LDR
    int adcValue = analogRead(SENSOR_PIN);
    float voltage = (adcValue / MY_ADC_RESOLUTION) * VCC;
    if (voltage <= 0) {
      voltage = 0.0001; // Evita divisão por zero
    }
    
    // Calcula a resistência do LDR utilizando o divisor de tensão:
    // V_out = VCC * (R_fixo/(R_fixo + R_LDR))  =>  R_LDR = R_fixo * (VCC/V_out - 1)
    float rLDR = FIXED_RESISTOR * (VCC / voltage - 1);
    
    // Converte a resistência do LDR em lux utilizando a relação log-log:
    // log10(LUX) = (log10(rLDR) - b) / m  =>  LUX = 10^((log10(rLDR) - b) / m)
    float lux = pow(10, (log10(rLDR) - LDR_B) / LDR_M);
    
    // Aplica a calibração de ganho (se necessária)
    float luxCorrigido = d + G * lux;
    luxValues[i] = luxCorrigido;
    
    // Imprime os dados em formato CSV: duty cycle (PWM) e lux medido
    Serial.print(u, 2);
    Serial.print(";");
    Serial.println(luxCorrigido, 2);
  }
  
  
}

void loop() {
  // A calibração é realizada uma única vez no setup.
}
