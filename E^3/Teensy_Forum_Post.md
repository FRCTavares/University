# Teensy Forum Post - CAN Communication Issue

**Title:** CAN Communication between two Teensy 4.0 with SN65HVD232D - Loopback works, external fails

---

**Post Content:**

Hi everyone,

I'm having trouble establishing CAN communication between two Teensy 4.0 boards using SN65HVD232D transceivers. The internal loopback works perfectly, but external communication fails completely.

## Hardware Setup
- 2x Teensy 4.0
- 2x SN65HVD232D CAN transceivers
- 120Ω termination resistors on both ends
- Tested both CAN1 (pins 22/23) and CAN2 (pins 0/1)

## Wiring (SN65HVD232D)
- Pin 1 (D) → Teensy CAN_TX
- Pin 2 (GND) → Ground
- Pin 3 (VCC) → 3.3V
- Pin 4 (R) → Teensy CAN_RX
- Pin 6 (CANL) → Other transceiver CANL + 120Ω resistor
- Pin 7 (CANH) → Other transceiver CANH + 120Ω resistor
- Pin 8 (RS) → Ground (for high-speed mode)

## Code Used (Simplified)
```cpp
// Sender
#include <FlexCAN_T4.h>
FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> can2;
CAN_message_t msg;

void setup() {
    Serial.begin(115200);
    delay(1000);
    can2.begin();
    can2.setBaudRate(125000);
    Serial.println("Sender ready");
}

void loop() {
    msg.id = 0x123;
    msg.len = 1;
    msg.buf[0] = 42;
    
    if (can2.write(msg)) {
        Serial.println("Sent!");
    } else {
        Serial.println("Failed!");
    }
    delay(1000);
}
```

```cpp
// Receiver
#include <FlexCAN_T4.h>
FlexCAN_T4<CAN2, RX_SIZE_256, TX_SIZE_16> can2;
CAN_message_t msg;

void setup() {
    Serial.begin(115200);
    delay(1000);
    can2.begin();
    can2.setBaudRate(125000);
    Serial.println("Receiver ready");
}

void loop() {
    if (can2.read(msg)) {
        Serial.print("Received: ");
        Serial.println(msg.buf[0]);
    }
    delay(10);
}
```

## What Works
✅ Internal loopback test works perfectly on both Teensy boards
✅ Both CAN1 and CAN2 controllers pass loopback tests
✅ Code compiles and runs without errors

## What Doesn't Work
❌ No external CAN communication between boards
❌ Receiver never gets any messages
❌ Sender reports "Sent!" initially, then alternates "Sent!"/"Failed!"

## Strange Behavior
The sender reports "Sent!" even when transceivers are completely disconnected, suggesting `can2.write()` only checks internal queuing, not actual bus transmission.

## What I've Tried
- Different baud rates: 125k, 250k, 500k
- Both CAN1 and CAN2 controllers
- Removed FIFO/mailbox configurations
- Verified all connections multiple times
- Confirmed RS pin connected to ground on both transceivers
- Verified 120Ω termination on both ends
- Stable 3.3V power supply

## Questions
1. Is there something specific about SN65HVD232D that I'm missing?
2. Should `can.write()` return false if no other node acknowledges the message?
3. Any known issues with FlexCAN_T4 and SN65HVD232D combination?
4. Would a different transceiver (like MCP2551) be more reliable?

Any help would be greatly appreciated! The fact that loopback works makes me think it's a transceiver configuration issue rather than Teensy/software problem.

Thanks in advance!

---

**Additional info to include if asked:**
- FlexCAN_T4 library version: [check your version]
- Arduino IDE version: [your version]  
- Teensyduino version: [your version]
- Operating System: Windows
