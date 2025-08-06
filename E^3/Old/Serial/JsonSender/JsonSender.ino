struct PanelData {
  float v1, v2, v3;          // volts
};

const int PIN_P1 = A7;
const int PIN_P2 = A6;
const int PIN_P3 = A5;
const float RATIO = 0.5f;    // 100 k : 100 k divider ⇒ Vpanel = Vpin / 0.5

char jsonBuf[64];            // enough for {"v1":x,"v2":y,"v3":z}

void setup() {
  Serial.begin(115200);          // USB debug
  Serial4.begin(9600);           // TX4=17 → RX? on the other Teensy

  pinMode(PIN_P1, INPUT_DISABLE);   // kill weak keepers
  pinMode(PIN_P2, INPUT_DISABLE);
  pinMode(PIN_P3, INPUT_DISABLE);

  analogReadResolution(12);      // 0-4095 → 0-3.3 V
}

float readPanel(int pin)
{
  int   raw   = analogRead(pin);
  float v_adc = raw * (3.3f / 4095.0f);
  return v_adc / RATIO;
}

void loop()
{
  PanelData d { readPanel(PIN_P1),
                readPanel(PIN_P2),
                readPanel(PIN_P3) };

  // build the JSON into jsonBuf
  snprintf(jsonBuf, sizeof(jsonBuf),
           "{\"v1\":%.3f,\"v2\":%.3f,\"v3\":%.3f}", d.v1, d.v2, d.v3);

  Serial4.println(jsonBuf);   // one clean line → other Teensy

  delay(1000);
}
