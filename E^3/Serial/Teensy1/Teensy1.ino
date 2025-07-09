void setup() {
  Serial.begin(115200);   // USB debug
  Serial4.begin(9600);    // Hardware UART on pins 16/17
}

void loop() {
  // Send once per second:
  static uint32_t t = millis();
  if (millis() - t >= 1000) {
    t = millis();
    Serial4.println("Ping from A");
    Serial.println("A sent ping");
  }
}
