#include <FlexCAN_T4.h>
FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> can2;
CAN_message_t msg;

void setup()
{
    Serial.begin(115200);
    delay(1000); // Give serial time to initialize

    can2.begin();
    can2.setBaudRate(125000); // Changed from 125000 to match sender

    Serial.println("CAN2 receiver ready");
}

void loop()
{
    if (can2.read(msg))
    {
        Serial.print("Got: ");
        Serial.println(msg.buf[0]);
    }
    delay(10); // Small delay to prevent overwhelming serial
}