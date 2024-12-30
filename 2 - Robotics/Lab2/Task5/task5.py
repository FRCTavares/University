import os
import sys
import math
import numpy as np
import pygame
import matplotlib.pyplot as plt

# Initialize pygame environment variable
os.environ["SDL_AUDIODRIVER"] = "dummy"
pygame.init()
pygame.font.init()  # Initialize the font module

# Screen dimensions
scale = 2
WIDTH, HEIGHT = 800 * scale, 600 * scale

# Car attributes
car_width, car_height = 100, 40
L = car_width  # Characteristic length
dt = 1 / 60  # 60 FPS
scale_m = 20
max_phi = math.radians(30)  # Maximum steering angle in radians

# Sensor attributes
sensor_angles = [-60, -45, -30, -15, 0, 15, 30, 45, 60]  # Added angles
sensor_length = 300  # Length of the sensor rays

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
BLACK = (0, 0, 0)

def normalize_angle(angle):
    """
    Normalize an angle to the range [0, 2π).
    """
    return angle % (2 * math.pi)

def kinematics_matrix(theta, phi, V, omega, L):
    """
    Compute the rate of change of the robot's state using the kinematics matrix.
    """
    kinematics_mat = np.array([
        [np.cos(theta) * np.cos(phi), 0],
        [np.sin(theta) * np.cos(phi), 0],
        [np.sin(phi) / L, 0],
        [0, 1]
    ])
    
    velocity_vec = np.array([V, omega])
    state_derivative = np.dot(kinematics_mat, velocity_vec)
    return state_derivative

def simulate_motion(initial_state, V, omega, L, dt, max_phi):
    """
    Simulate the motion of the robot for one time step using Euler integration.
    """
    x, y, theta, phi = initial_state
    derivatives = kinematics_matrix(theta, phi, V, omega, L)
    new_state = initial_state + derivatives * dt
    
    # Clamp the steering angle
    new_state[3] = max(-max_phi, min(max_phi, new_state[3]))
    return new_state

def draw_road_lanes(screen, color, width, height):
    """
    Draw two sine waves across the screen and return their points.
    """
    amplitude1 = 100  # Amplitude of the first wave
    frequency1 = 1   # Frequency of the first wave

    wave1_points = []
    wave2_points = []
    waves_center = []

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

    # Draw the center line
    for x in range(0, width):
        y_center = 0 + int(height // 2 + amplitude1 * math.sin(2 * math.pi * frequency1 * x / width))
        pygame.draw.circle(screen, color, (x, y_center), 1)
        waves_center.append((x, y_center))
    
    """
    Draw to straight horizontal lines across the screen and return their points.
    As well as its center line.
    """
    '''
    # First straight horizontal line
    for x in range(0, width):
        y1 = 150 + int(height // 2)
        pygame.draw.circle(screen, color, (x, y1), 2)
        wave1_points.append((x, y1))

    # Second straight horizontal line
    for x in range(0, width):
        y2 = -150 + int(height // 2)
        pygame.draw.circle(screen, color, (x, y2), 2)
        wave2_points.append((x, y2))

    # Draw the center line
    for x in range(0, width):
        y_center = 0 + int(height // 2)
        pygame.draw.circle(screen, color, (x, y_center), 1)
        waves_center.append((x, y_center))'''


    return wave1_points, wave2_points

def simulate_cone_sensor(car_pos, car_angle, cone_half_angle_degrees, sensor_length, 
                        screen, all_wave_points, wave1_points, wave2_points):

    cone_half_angle = math.radians(cone_half_angle_degrees)
    
    top_distance = None
    top_angle = None
    bottom_distance = None
    bottom_angle = None

    for point in all_wave_points:
        dx = point[0] - car_pos[0]
        dy = point[1] - car_pos[1]
        dist = math.hypot(dx, dy)
        if dist > sensor_length:
            continue
        
        angle_to_point = normalize_angle(math.atan2(dy, dx))
        angle_diff = min(
            abs(car_angle - angle_to_point),
            2 * math.pi - abs(car_angle - angle_to_point)
        )
        
        if angle_diff <= cone_half_angle:
            # Draw detection
            pygame.draw.line(screen, (0,255,0), car_pos, point, 1)
            # Keep track of closest top/bottom
            if point in wave1_points:  # top wave
                if top_distance is None or dist < top_distance:
                    top_distance = dist
                    top_angle = math.degrees(angle_diff)
            elif point in wave2_points:  # bottom wave
                if bottom_distance is None or dist < bottom_distance:
                    bottom_distance = dist
                    bottom_angle = math.degrees(angle_diff)
    
    return top_distance, top_angle, bottom_distance, bottom_angle

class Car:
    def __init__(self, x, y, theta=0, phi=0, V=0, omega=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.phi = phi
        self.V = V
        self.omega = omega

    def update_state(self):
        initial_state = np.array([self.x, self.y, self.theta, self.phi])
        new_state = simulate_motion(initial_state, self.V, self.omega, L, dt, max_phi)
        self.x, self.y, self.theta, self.phi = new_state
        self.theta = normalize_angle(self.theta)

    def draw(self, screen):
        # Draw the car (rotated)
        car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
        car_surface.fill(RED)
        rotated_car = pygame.transform.rotate(car_surface, -math.degrees(self.theta))
        car_rect = rotated_car.get_rect(center=(self.x, self.y))
        screen.blit(rotated_car, car_rect.topleft)

        # Draw the wheel (rotated)
        wheel_surface = pygame.Surface((car_width / 3, car_height / 5), pygame.SRCALPHA)
        wheel_surface.fill(BLUE)
        rotated_wheel = pygame.transform.rotate(wheel_surface, -math.degrees(self.theta) - math.degrees(self.phi))
        wheel_rect = rotated_wheel.get_rect(center=(self.x + (L / 2 - 10) * math.cos(self.theta), self.y + (L / 2 - 10) * math.sin(self.theta)))
        screen.blit(rotated_wheel, wheel_rect.topleft)

def main():
    # Initialize the screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Car Simulation with Sensors and Collision Detection")

    # Set exit button
    x_button_font = pygame.font.SysFont(None, 36)
    x_button_text = x_button_font.render('X', True, WHITE)
    x_button_rect = x_button_text.get_rect(center=(50, 100))

    # Initialize the car
    car = Car(x=WIDTH // 20, y=HEIGHT // 2)

    # Initialize data storage
    top_distances = []
    bottom_distances = []
    last_top_distance = float('inf')
    last_bottom_distance = float('inf')
    distance_to_center = 0

    Kp = 0.1 # Proportional gain
    Kd = 0.05 # Derivative gain
    previous_error = 0

    # Main loop
    running = True
    clock = pygame.time.Clock()

    while running:
        # Draw the road
        screen.fill(BLACK)
        wave_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        wave1_points, wave2_points = draw_road_lanes(wave_surface, WHITE, WIDTH, HEIGHT)
        all_wave_points = wave1_points + wave2_points

        # Action for cliking the exit button
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and x_button_rect.collidepoint(event.pos):
                running = False
            if event.type == pygame.QUIT:
                running = False

        # Key input for movement mechanics
        keys = pygame.key.get_pressed()
        user_steering = False  # Flag to check if user is steering

        # Throttle control (manual)
        if keys[pygame.K_UP]:
            car.V = 6 * scale_m
        elif keys[pygame.K_DOWN]:
            car.V = -6 * scale_m
        else:
            car.V = 0

        # if use manually steers, override the controller
        if keys[pygame.K_LEFT]:
            car.omega = -1
            user_steering = True
        elif keys[pygame.K_RIGHT]:
            car.omega = 1
            user_steering = True
        else:
            # If no manual steering, use a simple proportional controller
            error = distance_to_center
            derivative = (error - previous_error) / dt
            car.omega = -Kp * error - Kd * derivative
            previous_error = error


        # Update the car's state
        car.update_state()
        # Draw the car
        car.draw(screen)
        # Draw the road
        screen.blit(wave_surface, (0, 0))

        # Simulate sensors
        front_x = car.x + (car_width / 2) * math.cos(car.theta)
        front_y = car.y + (car_width / 2) * math.sin(car.theta)
        front_pos = (front_x, front_y)
        bottom_distance, bottom_angle, top_distance, top_angle = simulate_cone_sensor(front_pos, car.theta, 60, sensor_length, screen, all_wave_points, wave1_points, wave2_points)

        if top_distance is None:
            top_distance = last_top_distance
        if bottom_distance is None:
            bottom_distance = last_bottom_distance

        font = pygame.font.SysFont(None, 24)
        if top_distance is not None and top_angle is not None:
            top_info = font.render(f"Top: {top_distance:.2f}m @ {top_angle:.2f}°", True, WHITE)
            screen.blit(top_info, (400, 70))
        if bottom_distance is not None and bottom_angle is not None:
            bottom_info = font.render(f"Bottom: {bottom_distance:.2f}m @ {bottom_angle:.2f}°", True, WHITE)
            screen.blit(bottom_info, (400, 90))

        # Render and display the distance to the center of the road
        # Compute lane deviation and road alignment
        lane_deviation = (top_distance - bottom_distance) / 2  # Signed distance to center
        distance_to_center = lane_deviation  # Assuming lane_deviation represents the distance to center
        center_info = font.render(f"Distance to Center: {distance_to_center:.2f}m", True, WHITE)
        screen.blit(center_info, (400, 110))

        top_distances.append(top_distance)
        bottom_distances.append(bottom_distance)
        last_top_distance = top_distance
        last_bottom_distance = bottom_distance

        pygame.draw.rect(screen, (255, 0, 0), x_button_rect)
        screen.blit(x_button_text, x_button_rect)
        pygame.display.flip()
        clock.tick(60)

    plt.plot(top_distances, label='Top Lane Distance')
    plt.plot(bottom_distances, label='Bottom Lane Distance')
    plt.xlabel('Time (frames)')
    plt.ylabel('Shortest Distance')
    plt.title('Lane Distance Over Time')
    plt.legend()
    pygame.quit()  
    plt.show()

if __name__ == "__main__":
    main()