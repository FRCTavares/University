from sqlalchemy import Column, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import TIMESTAMP

Base = declarative_base()


class PowerTelemetry(Base):
    __tablename__ = "power_telemetry"
    time = Column(TIMESTAMP(timezone=True), primary_key=True, index=True)
    volt_y = Column(Float)
    volt_z = Column(Float)
    volt_pos_y = Column(Float)
    volt_x = Column(Float)
    battery_voltage = Column(Float)


class MagnetometerTelemetry(Base):
    __tablename__ = "magnetometer"
    time = Column(TIMESTAMP(timezone=True), primary_key=True, index=True)
    magno_x = Column(Float)
    magno_y = Column(Float)
    magno_z = Column(Float)


class GyroscopeTelemetry(Base):
    __tablename__ = "gyroscope"
    time = Column(TIMESTAMP(timezone=True), primary_key=True, index=True)
    roll_rate = Column(Float)
    pitch_rate = Column(Float)
    yaw_rate = Column(Float)


class AccelerometerTelemetry(Base):
    __tablename__ = "accelerometer"
    time = Column(TIMESTAMP(timezone=True), primary_key=True, index=True)
    acel_x = Column(Float)
    acel_y = Column(Float)
    acel_z = Column(Float)


class ThermalTelemetry(Base):
    __tablename__ = "thermal"
    time = Column(TIMESTAMP(timezone=True), primary_key=True, index=True)
    cpu_temp = Column(Float)
