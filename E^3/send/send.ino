#include <FlexCAN_T4.h>
// Teensy 4.0 CAN1 pins: TX=22, RX=23
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;
CAN_message_t tx;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 1500) {}
  can1.begin();
  can1.setBaudRate(125000);
  Serial.println("CAN1 Sender @125k ready");
}

void loop() {
  tx = {};
  tx.id = 0x123;          // standard 11-bit ID
  tx.len = 1;
  tx.flags.extended = 0;  // explicit: standard frame
  tx.buf[0] = 0x2A;       // 42
  if (can1.write(tx)) Serial.println("TX 0x123 [2A]");
  else                Serial.println("TX failed");
  delay(500);
}