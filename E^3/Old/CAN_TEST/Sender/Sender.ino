#include <FlexCAN_T4.h>

// Teensy 4.0 CAN1 uses pins: TX=22, RX=23
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;
CAN_message_t msg;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 1500) {}   // let USB come up

  can1.begin();
  can1.setBaudRate(500000);               // 500 kbps

  Serial.println("CAN Sender ready.");
}

void loop() {
  msg = {};                 // clear struct
  msg.id = 0x123;           // 11-bit standard ID
  msg.len = 1;              // 1 data byte
  msg.buf[0] = 42;          // payload

  if (can1.write(msg)) {
    Serial.println("CAN: sent ID 0x123, data[0]=42");
  } else {
    Serial.println("CAN: send failed (TX full?)");
  }

  delay(1000);
}