#include <FlexCAN_T4.h>

// Teensy 4.0 CAN1 uses pins: TX=22, RX=23
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;
CAN_message_t msg;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 1500) {}

  can1.begin();
  can1.setBaudRate(500000);   // 500 kbps

  Serial.println("CAN Receiver ready. Waiting for frames...");
}

void loop() {
  while (can1.read(msg)) {   // drain all pending frames
    Serial.print("RX ID 0x");
    Serial.print(msg.id, HEX);
    Serial.print("  DLC=");
    Serial.print(msg.len);
    Serial.print("  Data:");

    for (uint8_t i = 0; i < msg.len; i++) {
      Serial.print(" ");
      if (msg.buf[i] < 16) Serial.print("0");
      Serial.print(msg.buf[i], HEX);
    }
    Serial.println();
  }

  delay(5);
}