import numpy as np
from OpenGL.GL import *

class Skybox:
    def __init__(self, textures):
        self.textures = textures
        self.cube_vertices = [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]
        self.cube_faces = [
            (0, 1, 2, 3),  # Back face
            (1, 5, 6, 2),  # Right face
            (5, 4, 7, 6),  # Front face
            (4, 0, 3, 7),  # Left face
            (3, 2, 6, 7),  # Top face
            (4, 5, 1, 0)   # Bottom face
        ]

    def draw(self):
        glBegin(GL_QUADS)
        for i, face in enumerate(self.cube_faces):
            glBindTexture(GL_TEXTURE_2D, self.textures[i])
            for vertex in face:
                glTexCoord2f(vertex % 2, vertex // 2)
                glVertex3fv(self.cube_vertices[vertex])
        glEnd()