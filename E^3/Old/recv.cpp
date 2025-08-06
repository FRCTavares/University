#include <FlexCAN_T4.h>
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;
CAN_message_t msg;

void setup()
{
    Serial.begin(115200);
    delay(1000); // Give serial time to initialize

    can1.begin();
    can1.setBaudRate(500000);

    Serial.println("Receiver ready - waiting for messages...");
}

void loop()
{
    if (can1.read(msg))
    {
        Serial.print("Received! ID: 0x");
        Serial.print(msg.id, HEX);
        Serial.print(", Data: ");
        Serial.println(msg.buf[0]);
    }

    delay(10); // Small delay to prevent overwhelming serial
}