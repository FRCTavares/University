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

# Constants
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
BLACK = (0, 0, 0)

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

# Define a safe threshold and PD controller parameters
SAFE_THRESHOLD = 2
Kp = 0.1
Kd = 0.01

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
    new_state[3] = max(-max_phi, min(max_phi, new_state[3]))
    return new_state

def draw_center_lane(screen, color, width, height, dash_length=20, gap_length=20):
    """
    Draw a dashed center lane line on the screen.
    """
    top_y = height // 2 - 75
    bottom_y = top_y + 175
    center_y = (top_y + bottom_y) // 2

    x = 0
    while x < width:
        pygame.draw.line(screen, color, (x, center_y), (x + dash_length, center_y), 2)
        x += dash_length + gap_length

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

def detect_collision(car_mask, car_rect, wave_mask, wave_rect):
    """
    Detect collision between the car and the sine waves using masks.
    """
    offset = (wave_rect.left - car_rect.left, wave_rect.top - car_rect.top)
    overlap = car_mask.overlap(wave_mask, offset)
    if overlap:
        collision_point = (car_rect.left + overlap[0], car_rect.top + overlap[1])
        print(f"Collision at {collision_point}")
        return collision_point
    return None

def simulate_sensors(car_pos, car_angle, sensor_angles, sensor_length, screen, all_wave_points, wave1_points, wave2_points):
    """
    Simulate sensors fixed to the car frame and detect if they detect the sine waves.
    """
    detections = []
    top_distance = None
    bottom_distance = None

    for angle_offset in sensor_angles:
        total_angle = normalize_angle(car_angle + math.radians(angle_offset))
        end_x = car_pos[0] + sensor_length * math.cos(total_angle)
        end_y = car_pos[1] + sensor_length * math.sin(total_angle)
        pygame.draw.line(screen, (0, 255, 0), car_pos, (end_x, end_y), 1)

        closest_point = None
        min_distance = sensor_length
        for point in all_wave_points:
            dx = point[0] - car_pos[0]
            dy = point[1] - car_pos[1]
            distance = math.hypot(dx, dy)
            if distance > sensor_length:
                continue
            angle_point = normalize_angle(math.atan2(dy, dx))
            angle_diff = min(abs(total_angle - angle_point), 2 * math.pi - abs(total_angle - angle_point))
            if angle_diff < math.radians(2):
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point
        if closest_point:
            pygame.draw.circle(screen, (255, 255, 0), closest_point, 5)
            detections.append(closest_point)
            if closest_point in wave1_points:
                top_distance = min_distance if top_distance is None else min(top_distance, min_distance)
            elif closest_point in wave2_points:
                bottom_distance = min_distance if bottom_distance is None else min(bottom_distance, min_distance)

    return top_distance, bottom_distance

def simulate_cone_sensor(
    car_pos, car_angle, cone_half_angle_degrees, sensor_length, 
    screen, all_wave_points, wave1_points, wave2_points
):
    """
    Simulate a continuous cone sensor centered on car_angle with width of
    2 * cone_half_angle_degrees. Returns the closest top and bottom wave
    detections: (top_distance, top_angle, bottom_distance, bottom_angle).
    """
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

def pd_controller(error, prev_error, Kp, Kd, dt):
    """
    PD controller to compute the steering angle correction.
    """
    if abs(error) < SAFE_THRESHOLD:
        return 0  # No correction needed if within the safe threshold
    derivative = (error - prev_error) / dt
    control = Kp * error + Kd * derivative
    return control

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
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Car Simulation with Sensors and Collision Detection")

    x_button_font = pygame.font.SysFont(None, 36)
    x_button_text = x_button_font.render('X', True, WHITE)
    x_button_rect = x_button_text.get_rect(center=(50, 100))

    # Initialize the car
    car = Car(x=WIDTH // 2, y=HEIGHT // 2)

    # Initialize data storage
    top_distances = []
    bottom_distances = []
    last_top_distance = float('inf')
    last_bottom_distance = float('inf')

    prev_error = 0
    running = True
    clock = pygame.time.Clock()

    while running:
        screen.fill(BLACK)
        wave_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        wave1_points, wave2_points = draw_sine_waves(wave_surface, WHITE, WIDTH, HEIGHT)
        all_wave_points = wave1_points + wave2_points

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if x_button_rect.collidepoint(event.pos):
                    running = False

        # Key input for movement mechanics
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            car.omega = -1
        elif keys[pygame.K_RIGHT]:
            car.omega = 1
        else:
            car.omega = 0

        if keys[pygame.K_UP]:
            car.V = 5 * scale_m
        elif keys[pygame.K_DOWN]:
            car.V = -5 * scale_m
        else:
            car.V = 0

        # Update car state
        car.update_state()

        # Draw car
        car.draw(screen)

        # Create masks for collision
        car_mask_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
        pygame.draw.rect(car_mask_surface, WHITE, (0, 0, car_width, car_height))
        rotated_car_mask = pygame.transform.rotate(car_mask_surface, -math.degrees(car.theta))
        car_mask = pygame.mask.from_surface(rotated_car_mask)
        car_rect = rotated_car_mask.get_rect(center=(car.x, car.y))

        wave_mask = pygame.mask.from_surface(wave_surface)
        wave_rect = wave_surface.get_rect()
        screen.blit(wave_surface, (0, 0))

        # Simulate sensors
        front_x = car.x + (car_width / 2) * math.cos(car.theta)
        front_y = car.y + (car_width / 2) * math.sin(car.theta)
        front_pos = (front_x, front_y)
        top_distance, top_angle, bottom_distance, bottom_angle = simulate_cone_sensor(
            front_pos, car.theta, 60, sensor_length, screen, all_wave_points, wave1_points, wave2_points
        )

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

        top_distances.append(top_distance)
        bottom_distances.append(bottom_distance)
        last_top_distance = top_distance
        last_bottom_distance = bottom_distance

        if top_distance is not None and bottom_distance is not None:
            # Swap if needed
            if top_distance > bottom_distance:
                top_distance, bottom_distance = bottom_distance, top_distance

            distance_to_center = (bottom_distance - top_distance) / 2
            dist_text = font.render(f"Distance to Lane Center: {distance_to_center:.2f}", True, WHITE)
            screen.blit(dist_text, (400, 110))

            control = pd_controller(distance_to_center, prev_error, Kp, Kd, dt)
            car.phi = control
            prev_error = distance_to_center

        # Check collision
        collision_point = detect_collision(car_mask, car_rect, wave_mask, wave_rect)
        if collision_point:
            pygame.draw.circle(screen, (255, 0, 0), collision_point, 5)

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
    plt.show()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()