import numpy as np
from OpenGL.GL import *

class Terrain:
    def __init__(self, width, depth, height_map):
        self.width = width
        self.depth = depth
        self.height_map = height_map
        self.vertices = []
        self.indices = []
        self.generate_terrain()

    def generate_terrain(self):
        for z in range(self.depth):
            for x in range(self.width):
                height = self.height_map[z][x]
                self.vertices.append((x, height, z))

        for z in range(self.depth - 1):
            for x in range(self.width - 1):
                top_left = z * self.width + x
                top_right = top_left + 1
                bottom_left = (z + 1) * self.width + x
                bottom_right = bottom_left + 1

                self.indices.extend([top_left, bottom_left, top_right])
                self.indices.extend([top_right, bottom_left, bottom_right])

    def draw(self):
        glBegin(GL_TRIANGLES)
        for index in self.indices:
            vertex = self.vertices[index]
            glVertex3f(vertex[0], vertex[1], vertex[2])
        glEnd()