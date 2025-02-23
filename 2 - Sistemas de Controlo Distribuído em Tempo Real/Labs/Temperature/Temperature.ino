#define TEMP_SENSOR_PIN 4  // Internal temperature sensor is on ADC channel 4

void setup(){
  Serial.begin(9600);
  // Wait for the serial port to connect (useful for boards with native USB)
  while (!Serial) {
    delay(10);
  }
  delay(5000); // additional delay if needed for setup
}

void loop(){
  // Read the raw ADC value from the internal temperature sensor
  int sensorValue = analogRead(TEMP_SENSOR_PIN);
  
  // Convert the ADC reading (0-4095) to voltage (assuming a 3.3V reference)
  float voltage = sensorValue * (3.3 / 4095.0);
  
  // Convert voltage to temperature in Celsius using the RP2040 calibration formula:
  // Temperature (°C) = 27 - ((voltage - 0.706) / 0.001721)
  float temperature = 27.0 - ((voltage - 0.706) / 0.001721);
  
  // Print the temperature reading
  Serial.print("Core temperature: ");
  Serial.print(temperature, 1);  // one decimal place
  Serial.println(" C");
  
  delay(1000);  // wait 1 second between readings
}
