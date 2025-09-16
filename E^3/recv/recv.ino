#include <FlexCAN_T4.h>
// Teensy 4.0 CAN1 pins: TX=22, RX=23
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;
CAN_message_t rx;

void setup() {
  pinMode(13, OUTPUT);
  Serial.begin(115200);
  while (!Serial && millis() < 1500) {}

  can1.begin();
  can1.setBaudRate(125000);

  // Force accept-all
  can1.enableFIFO();
  can1.setFIFOFilter(REJECT_ALL);
  can1.setFIFOFilter(0, 0, STD);  // accept all standard IDs
  can1.setFIFOFilter(1, 0, EXT);  // accept all extended IDs

  Serial.println("CAN1 Receiver @125k ready (accept all)");
}

void loop() {
  while (can1.read(rx)) {
    digitalWrite(13, !digitalRead(13));   // blink on any RX
    Serial.print("RX ID 0x"); Serial.print(rx.id, HEX);
    Serial.print(" DLC="); Serial.print(rx.len);
    Serial.print(" Data:");
    for (uint8_t i=0;i<rx.len;i++){ Serial.print(" "); if(rx.buf[i]<16) Serial.print("0"); Serial.print(rx.buf[i],HEX); }
    Serial.println();
  }
  delay(2);
}