import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

class Renderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.init_opengl()

    def init_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (self.width / self.height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def clear(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def render(self, entities):
        self.clear()
        for entity in entities:
            entity.draw()
        pygame.display.flip()