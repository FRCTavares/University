#include <FlexCAN_T4.h>
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;
CAN_message_t msg;

void setup() {
  Serial.begin(115200);
  can1.begin();
  can1.setBaudRate(500000);

  Serial.println("Receiver ready.");
}

void loop() {
  if (can1.read(msg)) {
    Serial.print("Received message: ");
    Serial.print("Byte: ");
    Serial.println(msg.buf[0]);
  }
}
