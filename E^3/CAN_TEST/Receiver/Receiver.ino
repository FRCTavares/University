#include <FlexCAN_T4.h>
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;
CAN_message_t msg;

void setup() {
  Serial.begin(115200); delay(1000);

  can1.begin();
  can1.setBaudRate(500000);

  can1.setMaxMB(16);
  can1.enableFIFO();
  can1.enableFIFOInterrupt();
  can1.setFIFOFilter(0, 0x65, STD); 
  can1.onReceive(canSniff);
  can1.mailboxStatus();

  Serial.println("Receiver ready.");
}

void canSniff(const CAN_message_t &msg) {
  Serial.print("MB "); Serial.print(msg.mb);
  Serial.print("  OVERRUN: "); Serial.print(msg.flags.overrun);
  Serial.print("  LEN: "); Serial.print(msg.len);
  Serial.print(" EXT: "); Serial.print(msg.flags.extended);
  Serial.print(" TS: "); Serial.print(msg.timestamp);
  Serial.print(" ID: "); Serial.print(msg.id, HEX);
  Serial.print(" Buffer: ");
  for ( uint8_t i = 0; i < msg.len; i++ ) {
    Serial.print(msg.buf[i], HEX); Serial.print(" ");
  } Serial.println();
}

void loop() {
  can1.events();
  
  static uint32_t timeout = millis();
  if ( millis() - timeout > 1000 ) {
    Serial.println("waiting.");
    timeout = millis();
  }
}