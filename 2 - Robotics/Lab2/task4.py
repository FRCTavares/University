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
    Draw two sine waves across the screen and return their points.
    """
    amplitude1 = 100  # Amplitude of the first wave
    frequency1 = 1   # Frequency of the first wave

    amplitude2 = 100  # Amplitude of the second wave
    frequency2 = 2    # Frequency of the second wave

    wave1_points = []
    wave2_points = []

    # First sine wave
    for x in range(0, width):
        y1 = 100 + int(height // 2 + amplitude1 * math.sin(2 * math.pi * frequency1 * x / width))
        pygame.draw.circle(screen, color, (x, y1), 2)
        wave1_points.append((x, y1))

    # Second sine wave
    for x in range(0, width):
        y2 = -100 + int(height // 2 + amplitude1 * math.sin(2 * math.pi * frequency1 * x / width))
        pygame.draw.circle(screen, color, (x, y2), 2)
        wave2_points.append((x, y2))

    return wave1_points, wave2_points


def detect_collision(car_rect, wave_points):
    """
    Detect collision between the car and the sine waves.
    """
    for point in wave_points:
        if car_rect.collidepoint(point):
            collision_point = point
            print(f"Collision at {collision_point}")
            return collision_point
    return None


def simulate_sensors(car_pos, car_angle, sensor_angles, sensor_length, screen, wave_points):
    """
    Simulate sensors and detect if they detect the sine waves.
    """
    detections = []
    for angle_offset in sensor_angles:
        total_angle = car_angle + math.radians(angle_offset)
        end_x = car_pos[0] + sensor_length * math.cos(total_angle)
        end_y = car_pos[1] - sensor_length * math.sin(total_angle)
        pygame.draw.line(screen, (0, 255, 0), car_pos, (end_x, end_y), 1)

        # Check for intersection with wave points
        for point in wave_points:
            dx = point[0] - car_pos[0]
            dy = point[1] - car_pos[1]
            distance = math.hypot(dx, dy)
            angle_point = math.atan2(-dy, dx)
            angle_diff = abs(total_angle - angle_point)
            if distance <= sensor_length and angle_diff < math.radians(2):
                pygame.draw.circle(screen, (255, 255, 0), point, 5)
                detections.append(point)
                break
    return detections


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
pygame.display.set_caption("Car Simulation with Sensors and Collision Detection")

# Car attributes
car_width, car_height = 100, 40
x, y = WIDTH // 2, HEIGHT // 2
theta, phi = 0, 0
V, omega = 0, 0
L = car_width  # Characteristic length
dt = 1 / 60  # 60 FPS
scale_m = 20
max_phi = math.radians(30)  # Maximum steering angle in radians

# Sensor attributes
sensor_angles = [-60, -30, 0, 30, 60]  # Angles relative to the car's direction
sensor_length = 300  # Length of the sensor rays

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(BLACK)  # Clear the screen with black background
    initial_state = np.array([x, y, theta, phi])

    # Draw sine waves and get their points
    wave1_points, wave2_points = draw_sine_waves(screen, WHITE, WIDTH, HEIGHT)

    # Combine all wave points
    all_wave_points = wave1_points + wave2_points

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
        V = 5 * scale_m
    elif keys[pygame.K_DOWN]:  # Move backward
        V = -5 * scale_m
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

    # Simulate sensors
    car_pos = (x, y)
    detections = simulate_sensors(car_pos, theta, sensor_angles, sensor_length, screen, all_wave_points)

    # Detect collision
    collision_point = detect_collision(car_rect, all_wave_points)
    if collision_point:
        pygame.draw.circle(screen, (255, 0, 0), collision_point, 5)

    # Update the display
    pygame.display.flip()

    # Limit frames per second
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()
