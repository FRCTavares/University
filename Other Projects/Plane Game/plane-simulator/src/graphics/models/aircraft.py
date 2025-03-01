class Aircraft:
    def __init__(self, model_path, position=(0, 0, 0), rotation=(0, 0, 0)):
        self.model_path = model_path
        self.position = position
        self.rotation = rotation
        self.speed = 0.0

    def load_model(self):
        # Load the 3D model from the specified path
        pass

    def update(self, delta_time):
        # Update the aircraft's position and rotation based on speed and input
        pass

    def render(self):
        # Render the aircraft model in the 3D world
        pass

    def set_speed(self, speed):
        self.speed = speed

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position

    def get_rotation(self):
        return self.rotation

    def set_rotation(self, rotation):
        self.rotation = rotation