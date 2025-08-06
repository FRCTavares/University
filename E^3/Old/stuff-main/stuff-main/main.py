import json
import serial
import serial.tools.list_ports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone
from models import (
    PowerTelemetry,
    MagnetometerTelemetry,
    GyroscopeTelemetry,
    AccelerometerTelemetry,
    ThermalTelemetry,
    Base
)

# === CONFIGURATION ===
BAUD_RATE = 9600
DATABASE_URL = "postgresql://youruser:yourpassword@localhost:5432/satellite_telemetry"

# Set up DB
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def parse_and_store(data, session):
    timestamp = datetime.now(timezone.utc)

    accel = data.get("accel", {})
    gyro = data.get("gyro", {})
    mag = data.get("mag", {})
    voltages = data.get("voltages", {})
    temp_c = data.get("temp_c")

    accel_entry = AccelerometerTelemetry(
        time=timestamp,
        acel_x=accel.get("x"),
        acel_y=accel.get("y"),
        acel_z=accel.get("z")
    )

    gyro_entry = GyroscopeTelemetry(
        time=timestamp,
        roll_rate=gyro.get("x"),
        pitch_rate=gyro.get("y"),
        yaw_rate=gyro.get("z")
    )

    mag_entry = MagnetometerTelemetry(
        time=timestamp,
        magno_x=mag.get("x_uT"),
        magno_y=mag.get("y_uT"),
        magno_z=mag.get("z_uT")
    )

    power_entry = PowerTelemetry(
        time=timestamp,
        volt_y=voltages.get("v1"),
        volt_z=voltages.get("v2"),
        volt_pos_y=voltages.get("v3"),
        volt_x=None,
        battery_voltage=None
    )

    thermal_entry = ThermalTelemetry(
        time=timestamp,
        cpu_temp=temp_c
    )

    session.add_all([accel_entry, gyro_entry, mag_entry, power_entry, thermal_entry])
    session.commit()

def choose_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        exit(1)

    print("Available serial ports:")
    for i, port in enumerate(ports):
        print(f"[{i}] {port.device} - {port.description}")

    while True:
        try:
            choice = int(input("Select port number: "))
            if 0 <= choice < len(ports):
                return ports[choice].device
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a number.")

def main():
    selected_port = choose_serial_port()
    ser = serial.Serial(selected_port, BAUD_RATE, timeout=1)
    session = Session()
    try:
        print(f"Listening on {selected_port}...")
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    data = json.loads(line)
                    print("Received data:", data)
                    parse_and_store(data, session)
                    print("Stored data at", datetime.now())
                except json.JSONDecodeError:
                    print("Invalid JSON:", line)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        session.close()
        ser.close()

if __name__ == "__main__":
    main()
