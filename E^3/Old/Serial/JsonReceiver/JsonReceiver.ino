void setup() {
  Serial.begin(115200);     // USB Serial to PC
  Serial2.begin(9600);    // UART from other Teensy (pins 21/20)
}

void loop() {
  while (Serial2.available()) {
    char c = Serial2.read();
    Serial.write(c); // echo to Serial Monitor
  }
}