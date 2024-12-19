import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
import pygame
import sys
import math
import numpy as np


def kinematics_matrix(theta, phi, V, omega, L):
    """
    Compute the rate of change of the robot's state using the kinematics matrix.
    """
    # Define the kinematics matrix
    kinematics_mat = np.array([
        [np.cos(theta) * np.cos(phi), 0],
        [np.sin(theta) * np.cos(phi), 0],
        [np.sin(phi) / L, 0],
        [0, 1]
    ])
    
    # Input velocity vector
    velocity_vec = np.array([V, omega])
    
    # Compute the state derivatives
    state_derivative = np.dot(kinematics_mat, velocity_vec)
    
    return state_derivative


def simulate_motion(initial_state, V, omega, L, dt, max_phi):
    """
    Simulate the motion of the robot for one time step using Euler integration.
    """
    x, y, theta, phi = initial_state
    derivatives = kinematics_matrix(theta, phi, V, omega, L)
    new_state = initial_state + derivatives * dt

    # Constrain phi to the maximum steering angle
    new_state[3] = max(-max_phi, min(max_phi, new_state[3]))

    return new_state


def draw_sine_waves(screen, color, width, height):
    """
    Draw two sine waves across the screen.
    """
    amplitude1 = 100  # Amplitude of the first wave
    frequency1 = 1   # Frequency of the first wave

    amplitude2 = 100  # Amplitude of the second wave
    frequency2 = 2    # Frequency of the second wave

    # First sine wave
    for x in range(0, width):
        y1 = 100+int(height // 2 + amplitude1 * math.sin(2 * math.pi * frequency1 * x / width))
        pygame.draw.circle(screen, color, (x, y1), 2)

    # Second sine wave
    for x in range(0, width):
        #y2 = -100+int(height // 2 + amplitude2 * math.sin(2 * math.pi * frequency2 * x / width + math.pi / 2))
        y2 = -100+int(height // 2 + amplitude1 * math.sin(2 * math.pi * frequency1 * x / width))
        pygame.draw.circle(screen, color, (x, y2), 2)


# Initialize pygame
pygame.init()

# Screen dimensions
scale = 2
WIDTH, HEIGHT = 800 * scale, 600 * scale

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
BLACK = (0, 0, 0)

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Joystick-Like Car Steering with Sine Waves")

# Car attributes
car_width, car_height = 100, 40
x, y = WIDTH // 2, HEIGHT // 2
theta, phi = 0, 0
V, omega = 0, 0
L = car_width  # Characteristic length
dt = 1/60 #/ 60  # 60 FPS
scale_m=20
max_phi = math.radians(30)  # Maximum steering angle in radians

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(BLACK)  # Clear the screen with black background
    initial_state = np.array([x, y, theta, phi])

    # Draw sine waves
    draw_sine_waves(screen, WHITE, WIDTH, HEIGHT)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get keys pressed
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:  # Turn left
        omega = -1
    elif keys[pygame.K_RIGHT]:  # Turn right
        omega = +1
    else:
        omega = 0  # Reset angular velocity

    if keys[pygame.K_UP]:  # Move forward
        V = 5*scale_m
    elif keys[pygame.K_DOWN]:  # Move backward
        V = -5*scale_m
    else:
        V = 0  # Reset linear velocity

    # Simulate motion
    x, y, theta, phi = simulate_motion(initial_state, V, omega, L, dt, max_phi)

    # Draw the car (rotated)
    car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
    car_surface.fill(RED)
    rotated_car = pygame.transform.rotate(car_surface, -math.degrees(theta))
    car_rect = rotated_car.get_rect(center=(x, y))
    screen.blit(rotated_car, car_rect.topleft)

    # Draw the wheel (rotated)
    wheel_surface = pygame.Surface((car_width / 3, car_height / 5), pygame.SRCALPHA)
    wheel_surface.fill(BLUE)
    rotated_wheel = pygame.transform.rotate(wheel_surface, -math.degrees(theta) - math.degrees(phi))
    wheel_rect = rotated_wheel.get_rect(center=(x + (L / 2 - 10) * math.cos(theta), y + (L / 2 - 10) * math.sin(theta)))
    screen.blit(rotated_wheel, wheel_rect.topleft)

    # Update the display
    pygame.display.flip()

    # Limit frames per second
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()
