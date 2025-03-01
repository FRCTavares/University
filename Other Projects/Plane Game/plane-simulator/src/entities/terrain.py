class Terrain:
    def __init__(self, height_map, width, depth):
        self.height_map = height_map
        self.width = width
        self.depth = depth
        self.vertices = self.generate_vertices()
        self.indices = self.generate_indices()

    def generate_vertices(self):
        vertices = []
        for x in range(self.width):
            for z in range(self.depth):
                height = self.height_map[x][z]
                vertices.append((x, height, z))
        return vertices

    def generate_indices(self):
        indices = []
        for x in range(self.width - 1):
            for z in range(self.depth - 1):
                top_left = x * self.depth + z
                top_right = top_left + 1
                bottom_left = (x + 1) * self.depth + z
                bottom_right = bottom_left + 1

                indices.append((top_left, bottom_left, top_right))
                indices.append((top_right, bottom_left, bottom_right))
        return indices

    def get_vertices(self):
        return self.vertices

    def get_indices(self):
        return self.indices