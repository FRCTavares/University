#include <FlexCAN_T4.h>
FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_16> can1;
CAN_message_t msg;

void setup()
{
    Serial.begin(115200);
    delay(1000); // Give serial time to initialize

    can1.begin();
    can1.setBaudRate(500000);

    Serial.println("Sender ready.");
}

void loop()
{
    msg.id = 0x123;  // CAN ID
    msg.len = 1;     // Data length (1 byte)
    msg.buf[0] = 42; // Data to send

    can1.write(msg);
    Serial.println("Message sent!");

    delay(1000); // Send every second
}
