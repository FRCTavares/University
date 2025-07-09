void setup() {
  Serial4.begin(115200);
}

void loop() {
  Serial4.println("Hello from Teensy 1!");
  delay(1000);
}