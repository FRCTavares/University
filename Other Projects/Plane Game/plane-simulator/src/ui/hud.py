import pygame
import sys

class HUD:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 36)

    def draw(self, plane_pos, speed, altitude):
        # Draw speed indicator
        speed_text = self.font.render(f"Speed: {speed:.2f} m/s", True, (255, 255, 255))
        self.screen.blit(speed_text, (10, 10))

        # Draw altitude indicator
        altitude_text = self.font.render(f"Altitude: {altitude:.2f} m", True, (255, 255, 255))
        self.screen.blit(altitude_text, (10, 50))

        # Draw position indicator
        position_text = self.font.render(f"Position: X: {plane_pos[0]:.2f}, Y: {plane_pos[1]:.2f}, Z: {plane_pos[2]:.2f}", True, (255, 255, 255))
        self.screen.blit(position_text, (10, 90))

    def update(self):
        # Update HUD elements if necessary
        pass