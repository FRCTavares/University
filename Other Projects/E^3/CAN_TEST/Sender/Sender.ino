#include <FlexCAN_T4.h>
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;
CAN_message_t msg;

void setup() {
  Serial.begin(115200);
  can1.begin();
  can1.setBaudRate(500000);  // Standard CAN speed

  Serial.println("Sender ready.");
}

void loop() {
  msg.id = 0x100;
  msg.len = 1;
  msg.buf[0] = 42;
  can1.write(msg);

  Serial.println("Sent message");
  delay(1000);
}
