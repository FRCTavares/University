import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

class Game:
    def __init__(self):
        self.running = True
        self.clock = pygame.time.Clock()
        self.plane_pos = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.speed = 0.1
        # Initialize pygame and OpenGL right away
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Plane Simulator")
        # Set up the perspective and initial position
        gluPerspective(45, (display[0] / display[1]), 0.1, 1000.0)  # Increased far plane
        glEnable(GL_DEPTH_TEST)
        glTranslatef(0.0, 0.0, -5)  # Changed from -20 to -5

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        keys = pygame.key.get_pressed()
        if keys[K_LEFT]:
            self.yaw += 1.0
        if keys[K_RIGHT]:
            self.yaw -= 1.0
        if keys[K_UP]:
            self.pitch += 1.0
        if keys[K_DOWN]:
            self.pitch -= 1.0
        if keys[K_a]:
            self.roll += 1.0
        if keys[K_d]:
            self.roll -= 1.0
        if keys[K_w]:
            self.speed += 0.01
        if keys[K_s]:
            self.speed = max(0.01, self.speed - 0.01)
        if keys[K_ESCAPE]:  # Add escape key to quit
            self.running = False

    def update(self):
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)

        dir_x = math.cos(pitch_rad) * math.sin(yaw_rad)
        dir_y = math.sin(pitch_rad)
        dir_z = math.cos(pitch_rad) * math.cos(yaw_rad)

        self.plane_pos[0] += dir_x * self.speed
        self.plane_pos[1] += dir_y * self.speed
        self.plane_pos[2] += dir_z * self.speed

    def render(self):
        # Set background color (sky blue)
        glClearColor(0.5, 0.8, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Adjust camera position for better visibility
        camera_offset = [0, 2, 5]  # Reduced from 10 to 5
        gluLookAt(
            self.plane_pos[0] - camera_offset[0], 
            self.plane_pos[1] + camera_offset[1], 
            self.plane_pos[2] - camera_offset[2],
            self.plane_pos[0], self.plane_pos[1], self.plane_pos[2],
            0, 1, 0
        )
        
        glPushMatrix()
        glTranslatef(self.plane_pos[0], self.plane_pos[1], self.plane_pos[2])
        glRotatef(self.yaw, 0, 1, 0)
        glRotatef(self.pitch, 1, 0, 0)
        glRotatef(self.roll, 0, 0, 1)
        self.draw_plane()
        glPopMatrix()
        
        # Draw debug info
        self.draw_debug_info()
        
        pygame.display.flip()
        self.clock.tick(60)

    def draw_plane(self):
        # Make the plane more visible with thicker lines and bright color
        glLineWidth(3)
        glColor3f(1.0, 0.0, 0.0)  # Bright red
        
        glBegin(GL_LINES)
        # Main body
        glVertex3f(0, 0, 2)
        glVertex3f(0, 0, -2)
        
        # Wings
        glVertex3f(0, 0, 0)
        glVertex3f(-2, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(2, 0, 0)
        
        # Tail
        glVertex3f(0, 0, -2)
        glVertex3f(1, 1, -3)
        glVertex3f(0, 0, -2)
        glVertex3f(-1, 1, -3)
        glEnd()

    def draw_debug_info(self):
        # Reset matrices for 2D drawing
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 800, 600, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth test temporarily
        glDisable(GL_DEPTH_TEST)
        
        # Render text using pygame
        font = pygame.font.SysFont('arial', 20)
        text_surface = font.render(f"Pos: {self.plane_pos[0]:.1f}, {self.plane_pos[1]:.1f}, {self.plane_pos[2]:.1f}", True, (255, 255, 255))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # Restore states
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def run(self):
        # No need to call initialize() as we do it in __init__
        while self.running:
            self.handle_events()
            self.update()
            self.render()
        pygame.quit()