// Test CAN2 loopback - upload to ONE Teensy only
#include <FlexCAN_T4.h>
FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> can2;
CAN_message_t msg;

void setup()
{
    Serial.begin(115200);
    delay(1000);

    can2.begin();
    can2.setBaudRate(125000);
    can2.enableLoopBack(); // Internal loopback

    Serial.println("Testing CAN2 loopback...");
}

void loop()
{
    msg.id = 0x123;
    msg.len = 1;
    msg.buf[0] = 42;

    if (can2.write(msg))
    {
        Serial.println("Sent");
    }

    if (can2.read(msg))
    {
        Serial.print("CAN2 LOOPBACK SUCCESS! Got: ");
        Serial.println(msg.buf[0]);
    }
    else
    {
        Serial.println("No loopback");
    }

    delay(1000);
}