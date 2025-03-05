class Aircraft(Entity):
    def __init__(self, model, position, rotation):
        super().__init__(position, rotation)
        self.model = model
        self.speed = 0.0
        self.thrust = 0.1
        self.drag = 0.05

    def update(self, delta_time):
        self.position[0] += self.speed * math.cos(math.radians(self.rotation[1])) * delta_time
        self.position[1] += self.speed * math.sin(math.radians(self.rotation[1])) * delta_time
        self.speed -= self.drag * delta_time

    def apply_thrust(self):
        self.speed += self.thrust

    def render(self, renderer):
        renderer.draw_model(self.model, self.position, self.rotation)