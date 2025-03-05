class Entity:
    def __init__(self, position, rotation):
        self.position = position  # Position in 3D space (x, y, z)
        self.rotation = rotation    # Rotation (yaw, pitch, roll)

    def update(self, delta_time):
        """Update the entity's state. To be implemented by subclasses."""
        pass

    def render(self):
        """Render the entity. To be implemented by subclasses."""
        pass