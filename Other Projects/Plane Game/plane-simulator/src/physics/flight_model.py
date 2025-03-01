# FILE: /plane-simulator/plane-simulator/src/physics/flight_model.py

import math

class FlightModel:
    def __init__(self, max_speed=100.0, max_pitch=30.0, max_roll=30.0):
        self.position = [0.0, 0.0, 0.0]  # x, y, z
        self.velocity = [0.0, 0.0, 0.0]  # vx, vy, vz
        self.yaw = 0.0  # rotation around Y-axis
        self.pitch = 0.0  # rotation around X-axis
        self.roll = 0.0  # rotation around Z-axis
        self.max_speed = max_speed
        self.max_pitch = max_pitch
        self.max_roll = max_roll

    def update(self, delta_time):
        # Update position based on velocity
        self.position[0] += self.velocity[0] * delta_time
        self.position[1] += self.velocity[1] * delta_time
        self.position[2] += self.velocity[2] * delta_time

        # Apply simple physics for pitch and roll
        self.velocity[0] += math.sin(math.radians(self.yaw)) * self.max_speed * delta_time
        self.velocity[1] += math.sin(math.radians(self.pitch)) * self.max_speed * delta_time
        self.velocity[2] += math.cos(math.radians(self.yaw)) * self.max_speed * delta_time

        # Clamp speed
        speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2 + self.velocity[2]**2)
        if speed > self.max_speed:
            self.velocity = [v * (self.max_speed / speed) for v in self.velocity]

    def set_orientation(self, yaw, pitch, roll):
        self.yaw = max(-self.max_roll, min(yaw, self.max_roll))
        self.pitch = max(-self.max_pitch, min(pitch, self.max_pitch))
        self.roll = roll  # Roll can be unrestricted for this model

    def get_position(self):
        return self.position

    def get_velocity(self):
        return self.velocity

    def reset(self):
        self.position = [0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0