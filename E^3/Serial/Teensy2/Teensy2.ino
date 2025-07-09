void setup() {
  Serial.begin(115200);   // USB debug
  Serial2.begin(9600);    // Hardware UART on pins 9/10
}

void loop() {
  // Listen for ping on Serial2:
  if (Serial2.available()) {
    String msg = Serial2.readStringUntil('\n');
    Serial.println("B got: " + msg);
    // Send ACK back:
    Serial2.println("Pong from B");
    Serial.println("B sent pong");
  }
}
