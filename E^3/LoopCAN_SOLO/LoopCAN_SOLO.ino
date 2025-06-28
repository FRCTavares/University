#include <FlexCAN_T4.h>

FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;

void setup() {
  Serial.begin(115200);

  can1.setPins(1, 0);            // Default TX/RX for CAN1 on Teensy 4.0
  can1.begin();
  can1.setBaudRate(500000);
  can1.setCtrlMode(CAN_LOOPBACK);  // Internal loopback mode — no transceiver needed

  Serial.println("CAN1 loopback test running...");
}

void loop() {
  CAN_message_t msg;
  msg.id = 0x321;
  msg.len = 1;
  msg.buf[0] = 42;
  can1.write(msg);

  if (can1.read(msg)) {
    Serial.print("Received in loopback: ");
    Serial.println(msg.buf[0]);
  }

  delay(500);
}
