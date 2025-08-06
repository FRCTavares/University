#include <FlexCAN_T4.h>
FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> can2;
CAN_message_t msg;

void setup()
{
    Serial.begin(115200);
    delay(1000); // Give serial time to initialize

    can2.begin();
    can2.setBaudRate(125000);

    Serial.println("CAN2 sender ready");
}

void loop()
{
    msg.id = 0x123;  // CAN ID
    msg.len = 1;     // Data length (1 byte)
    msg.buf[0] = 42; // Data to send

    if (can2.write(msg))
    {
        Serial.println("Sent!");
    }
    else
    {
        Serial.println("Failed!");
    }

    delay(1000);
}
