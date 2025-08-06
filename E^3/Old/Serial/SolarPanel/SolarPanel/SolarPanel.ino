const int PIN_ADC = A5;       // your ADC pin
const float RATIO = 0.5f;     // 10 MΩ bottom / (10 MΩ + 10 MΩ)

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);    // 0–4095 over 0–3.3 V
}

void loop() {
  int raw = analogRead(PIN_ADC);                          
  float v_adc   = raw * (3.3f / 4095.0f);                  
  float v_panel = v_adc / RATIO;                          

  Serial.print("V_panel: ");
  Serial.print(v_panel, 3);
  Serial.println(" V");
  delay(100);
}
