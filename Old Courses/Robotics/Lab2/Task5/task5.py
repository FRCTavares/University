import os
import sys
import math
import numpy as np
import pygame
import matplotlib.pyplot as plt
import random

# --------------------------------------------------------------------
# INITIAL SETUP
# --------------------------------------------------------------------
# Initialize pygame environment variable
os.environ["SDL_AUDIODRIVER"] = "dummy"
pygame.init()
pygame.font.init()  # Initialize the font module

# Screen dimensions
scale = 2
WIDTH, HEIGHT = 800 * scale, 600 * scale

# Car attributes
car_width, car_height = 100, 40
L = car_width           # Characteristic length of the car
dt = 1 / 60            # 60 FPS -> Time step
scale_m = 20           # Scale factor for movement
max_phi = math.radians(30)  # Maximum steering angle in radians

# Sensor attributes
sensor_angles = [-60, -45, -30, -15, 0, 15, 30, 45, 60]  # Sensor angles
sensor_length = 300                                      # Sensor range

# Colors
WHITE = (255, 255, 255)
RED   = (200,   0,   0)
BLUE  = (  0,   0, 200)
BLACK = (  0,   0,   0)

# --------------------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------------------
def normalize_angle(angle):
    """
    Normalize an angle to the range [0, 2π).
    This ensures we don't accumulate angles beyond 2π or below 0.
    """
    return angle % (2 * math.pi)

def kinematics_matrix(theta, phi, V, omega, L):
    """
    Compute the rate of change of the robot's state using the kinematics matrix.
    Returns [dx, dy, dtheta, dphi].
    """
    kinematics_mat = np.array([
        [np.cos(theta) * np.cos(phi), 0],
        [np.sin(theta) * np.cos(phi), 0],
        [np.sin(phi) / L,             0],
        [0,                           1]
    ])
    
    velocity_vec = np.array([V, omega])
    state_derivative = np.dot(kinematics_mat, velocity_vec)
    return state_derivative

def simulate_motion(initial_state, V, omega, L, dt, max_phi):
    """
    Simulate the motion of the robot for one time step using Euler integration.
    Clamps steering angle phi to within ±max_phi.
    """
    x, y, theta, phi = initial_state
    derivatives = kinematics_matrix(theta, phi, V, omega, L)
    new_state = initial_state + derivatives * dt
    
    # Clamp the steering angle
    new_state[3] = max(-max_phi, min(max_phi, new_state[3]))
    return new_state

def draw_road_lanes(screen, color, width, height, curve_angle=90):
    """
    Draw a straight road followed by an inclined straight road for consistent lane width.
    Returns lists of points for each lane and the center line.

    Parameters:
    - screen: The pygame screen to draw on.
    - color: The color of the road lines.
    - width: Total width of the road drawing area.
    - height: Total height of the road drawing area.
    - curve_angle: The angle of the inclined road in degrees (default is 90 degrees).
    """
    amplitude1 = 100  # Amplitude of the first wave
    frequency1 = 1   # Frequency of the first wave

    amplitude2 = 100  # Amplitude of the second wave
    frequency2 = 1    # Frequency of the second wave

    wave1_points = []
    wave2_points = []

    # First sine wave
    for x in range(0, width):
        y1 = 100 + int(height // 2 + amplitude1 * math.sin(2 * math.pi * frequency2 * x / width))
        pygame.draw.circle(screen, color, (x, y1), 2)
        wave1_points.append((x, y1))

    # Second sine wave
    for x in range(0, width):
        y2 = -100 + int(height // 2 + amplitude1 * math.sin(2 * math.pi * frequency2 * x / width))
        pygame.draw.circle(screen, color, (x, y2), 2)
        wave2_points.append((x, y2))

    return wave1_points, wave2_points

def simulate_cone_sensor(
    car_pos,
    car_angle,
    cone_half_angle_degrees,
    sensor_length,
    screen,
    all_wave_points,
    wave1_points,
    wave2_points
):
    """
    Simulate a cone sensor from the front of the car.
    The sensor spans 'cone_half_angle_degrees' to each side of 'car_angle'.
    Returns the closest top_distance and bottom_distance with angles.
    """
    cone_half_angle = math.radians(cone_half_angle_degrees)
    
    top_distance = None
    top_angle = None
    bottom_distance = None
    bottom_angle = None

    # Check all wave points to see which ones fall within the sensor cone
    for point in all_wave_points:
        dx = point[0] - car_pos[0]
        dy = point[1] - car_pos[1]
        dist = math.hypot(dx, dy)
        noise_std = 5.0 # Standard deviation of noise
        dist += random.gauss(0, noise_std)  # Add noise to the distance
        # Skip points beyond sensor range
        if dist > sensor_length:
            continue
        
        # Calculate angle difference to see if it's within the cone
        angle_to_point = normalize_angle(math.atan2(dy, dx))
        angle_diff = min(
            abs(car_angle - angle_to_point),
            2 * math.pi - abs(car_angle - angle_to_point)
        )
        
        if angle_diff <= cone_half_angle:
            # Draw a detection line for visualization
            pygame.draw.line(screen, (0,255,0), car_pos, point, 1)
            # Keep track of closest top/bottom wave
            if point in wave1_points:  # top wave
                if top_distance is None or dist < top_distance:
                    top_distance = dist
                    top_angle = math.degrees(angle_diff)
            elif point in wave2_points:  # bottom wave
                if bottom_distance is None or dist < bottom_distance:
                    bottom_distance = dist
                    bottom_angle = math.degrees(angle_diff)
    
    return top_distance, top_angle, bottom_distance, bottom_angle

def detect_and_visualize_collisions(car_rect, car_surface, wave_surface, screen):
    """
    Detect and visualize collisions between the car and the wave lanes.
    """
    # Create masks for the car and the wave surface
    car_mask = pygame.mask.from_surface(car_surface)
    wave_mask = pygame.mask.from_surface(wave_surface)
    
    # Compute offset for mask overlap
    offset = (wave_surface.get_rect().left - car_rect.left, 
              wave_surface.get_rect().top - car_rect.top)
    
    # Detect overlap
    collision_point = car_mask.overlap(wave_mask, offset)
    if collision_point:
        # Calculate collision coordinates on the screen
        collision_coords = (car_rect.left + collision_point[0], car_rect.top + collision_point[1])
        # Draw fluorescent dot at the collision point
        pygame.draw.circle(screen, (255, 0, 255), collision_coords, 8)
        print(f"Collision detected at: {collision_coords}")

# --------------------------------------------------------------------
# PLOTTING FUNCTIONS
# --------------------------------------------------------------------
def draw_distance_plot(screen, top_distances, bottom_distances):
    # Define where and how big the mini-plot should be
    plot_x, plot_y = 650, 50
    plot_width, plot_height = 600, 300
    max_points = 10000  # Number of recent distance points to plot
    
    recent_top = top_distances[-max_points:]
    recent_bottom = bottom_distances[-max_points:]
    if not recent_top or not recent_bottom:
        return
    
    max_val = max(max(recent_top), max(recent_bottom), 1)
    pygame.draw.rect(screen, (50, 50, 50), (plot_x, plot_y, plot_width, plot_height))
    
    for i in range(len(recent_top) - 1):
        x1 = plot_x + (i / (len(recent_top) - 1)) * plot_width
        y1 = plot_y + plot_height - (recent_top[i] / max_val) * plot_height
        x2 = plot_x + ((i+1) / (len(recent_top) - 1)) * plot_width
        y2 = plot_y + plot_height - (recent_top[i+1] / max_val) * plot_height
        pygame.draw.line(screen, (255, 0, 0), (x1, y1), (x2, y2), 2)
    
    for i in range(len(recent_bottom) - 1):
        x1 = plot_x + (i / (len(recent_bottom) - 1)) * plot_width
        y1 = plot_y + plot_height - (recent_bottom[i] / max_val) * plot_height
        x2 = plot_x + ((i+1) / (len(recent_bottom) - 1)) * plot_width
        y2 = plot_y + plot_height - (recent_bottom[i+1] / max_val) * plot_height
        pygame.draw.line(screen, (0, 0, 255), (x1, y1), (x2, y2), 2)

def draw_omega_plot(screen, user_omega_history, controller_omega_history):
    # Define where and how big the mini-plot should be
    plot_x, plot_y = 800, 800
    plot_width, plot_height = 600, 300
    max_points = 300  # Number of recent distance points to plot

    recent_user = user_omega_history[-max_points:]
    recent_ctrl = controller_omega_history[-max_points:]
    if not recent_user or not recent_ctrl:
        return
    
    max_val = max(abs(max(recent_user, default=0)), abs(max(recent_ctrl, default=0)), 1)
    pygame.draw.rect(screen, (50, 50, 50), (plot_x, plot_y, plot_width, plot_height))
    
    for i in range(len(recent_user) - 1):
        x1 = plot_x + (i / (len(recent_user) - 1)) * plot_width
        y1 = plot_y + plot_height - ((recent_user[i] / max_val) * plot_height * 0.5 + plot_height * 0.5)
        x2 = plot_x + ((i+1) / (len(recent_user) - 1)) * plot_width
        y2 = plot_y + plot_height - ((recent_user[i+1] / max_val) * plot_height * 0.5 + plot_height * 0.5)
        pygame.draw.line(screen, (255, 0, 0), (x1, y1), (x2, y2), 2)
    
    for i in range(len(recent_ctrl) - 1):
        x1 = plot_x + (i / (len(recent_ctrl) - 1)) * plot_width
        y1 = plot_y + plot_height - ((recent_ctrl[i] / max_val) * plot_height * 0.5 + plot_height * 0.5)
        x2 = plot_x + ((i+1) / (len(recent_ctrl) - 1)) * plot_width
        y2 = plot_y + plot_height - ((recent_ctrl[i+1] / max_val) * plot_height * 0.5 + plot_height * 0.5)
        pygame.draw.line(screen, (0, 255, 0), (x1, y1), (x2, y2), 2)

# --------------------------------------------------------------------
# CAR CLASS
# --------------------------------------------------------------------
class Car:
    def __init__(self, x, y, theta=0, phi=0, V=0, omega=0):
        """
        This class holds the car state: position (x, y), heading (theta),
        steering angle (phi), forward velocity (V), and steering velocity (omega).
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.phi = phi
        self.V = V
        self.omega = omega

    def update_state(self):
        """
        Update the car's state using the kinematic simulation.
        """
        initial_state = np.array([self.x, self.y, self.theta, self.phi])
        new_state = simulate_motion(initial_state, self.V, self.omega, L, dt, max_phi)
        self.x, self.y, self.theta, self.phi = new_state
        self.theta = normalize_angle(self.theta)

    def draw(self, screen):
        """
        Draw the car as a rectangle with a small rectangle for the front-wheel steering.
        """
        # Draw main body (rotated by theta)
        car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
        car_surface.fill(RED)
        rotated_car = pygame.transform.rotate(car_surface, -math.degrees(self.theta))
        car_rect = rotated_car.get_rect(center=(self.x, self.y))
        screen.blit(rotated_car, car_rect.topleft)

        # Draw front wheel (rotated by both theta + phi)
        wheel_surface = pygame.Surface((car_width / 3, car_height / 5), pygame.SRCALPHA)
        wheel_surface.fill(BLUE)
        rotated_wheel = pygame.transform.rotate(
            wheel_surface,
            -math.degrees(self.theta) - math.degrees(self.phi)
        )
        # Place wheel near the front of the car
        wheel_rect = rotated_wheel.get_rect(
            center=(
                self.x + (L / 2 - 10) * math.cos(self.theta),
                self.y + (L / 2 - 10) * math.sin(self.theta)
            )
        )
        screen.blit(rotated_wheel, wheel_rect.topleft)
        return car_surface

# --------------------------------------------------------------------
# MAIN GAME LOOP
# --------------------------------------------------------------------
def main():
    """
    Main function that initializes the pygame window, runs the simulation,
    and plots data after the simulation ends.
    """
    # Initialize the display window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Car Simulation with Sensors and Collision Detection")

    # Create an 'X' exit button
    x_button_font = pygame.font.SysFont(None, 36)
    x_button_text = x_button_font.render('X', True, WHITE)
    x_button_rect = x_button_text.get_rect(center=(50, 100))

    # Initialize the car at a chosen starting position
    car = Car(x=WIDTH // 20, y=HEIGHT // 2)

    # Initialize sensor and controller variables
    top_distances = []
    bottom_distances = []
    last_top_distance = float('inf')
    last_bottom_distance = float('inf')
    distance_to_center = 0
    top_distance = 40
    bottom_distance = 40
    controller_activated = False
    steering_text = "User Steering"

    # Controller gains
    Kp = 0.5
    Kd = 0.5
    previous_error = 0

    # Arrays to log steering data for plotting
    user_omega_history = []
    controller_omega_history = []

    # Setup clock for consistent framerate
    running = True
    clock = pygame.time.Clock()

    # ----------------------------------------------------------------
    # START SIMULATION LOOP
    # ----------------------------------------------------------------
    while running:
        user_omega = 0
        control_omega = 0

        user_omega_history.append(user_omega)
        controller_omega_history.append(control_omega)

        # Clear screen and draw background
        screen.fill(BLACK)
        wave_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        wave1_points, wave2_points = draw_road_lanes(wave_surface, WHITE, WIDTH, HEIGHT)
        all_wave_points = wave1_points + wave2_points


        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and x_button_rect.collidepoint(event.pos):
                running = False
            if event.type == pygame.QUIT:
                running = False


        # Handle keyboard inputs for user steering
        keys = pygame.key.get_pressed()
        user_steering = False

        # Throttle control (manual up/down arrow keys)
        if keys[pygame.K_UP]:
            car.V = 6 * scale_m
        elif keys[pygame.K_DOWN]:
            car.V = -6 * scale_m
        else:
            car.V = 0

        # Steering control (left/right arrow keys)
        if keys[pygame.K_LEFT]:
            user_omega = -2
            user_steering = True
            controller_activated = False
        elif keys[pygame.K_RIGHT]:
            user_omega = 2
            user_steering = True
            controller_activated = False
        else:
            user_steering = False
        
        # ----------------------------------------------------------------
        # CONTROLLER LOGIC
        # Only act if the car is too close to top or bottom lanes
        # ----------------------------------------------------------------
        safe_threshold = 40
        if not user_steering:
            if top_distance < safe_threshold or bottom_distance < safe_threshold:
                error = distance_to_center
                derivative = (error - previous_error) / dt
                control_omega = -Kp * error - Kd * derivative
                # Clamp controller output
                control_omega = max(-2, min(control_omega, 2))
                previous_error = error
                controller_activated = True

        # Final steering decision
        car.omega = user_omega if user_steering else control_omega

        # ----------------------------------------------------------------
        # DISPLAY STEERING MODE
        # ----------------------------------------------------------------
        font_big = pygame.font.SysFont(None, 48)
        if controller_activated:
            steering_text = "Controller Steering"
            text_color = (0, 255, 0)  # green
        else:
            steering_text = "User Steering"
            text_color = (255, 0, 0)  # red

        status_info = font_big.render(steering_text, True, text_color)
        screen.blit(status_info, (400, 925))

        # Log steering data for later analysis
        user_omega_history.append(user_omega)
        controller_omega_history.append(control_omega)

        # ----------------------------------------------------------------
        # UPDATE & RENDER
        # ----------------------------------------------------------------
        # Update the car's state
        car.update_state()
        # Draw the car
        car.draw(screen)
        car_surface = car.draw(screen)
        # Draw the road
        screen.blit(wave_surface, (0, 0))

        car_rect = pygame.Rect(car.x - car_width // 2, car.y - car_height // 2, car_width, car_height)

        # Detect and visualize collisions
        detect_and_visualize_collisions(car_rect, car_surface, wave_surface, screen)


        # Simulate sensors (cone around the front of the car)
        front_x = car.x + (car_width / 2) * math.cos(car.theta)
        front_y = car.y + (car_width / 2) * math.sin(car.theta)
        front_pos = (front_x, front_y)
        # Return distances from top/bottom waves
        bottom_distance, bottom_angle, top_distance, top_angle = simulate_cone_sensor(
            front_pos, 
            car.theta, 
            60, 
            sensor_length, 
            screen, 
            all_wave_points, 
            wave1_points, 
            wave2_points
        )

        # Preserve lane distance if sensor doesn't detect anything in the current frame
        if top_distance is None:
            top_distance = last_top_distance
        if bottom_distance is None:
            bottom_distance = last_bottom_distance

        # Display top/bottom distance info
        font = pygame.font.SysFont(None, 48)
        if top_distance is not None and top_angle is not None:
            top_info = font.render(f"Top: {top_distance:.2f}m @ {top_angle:.2f}°", True, RED)
            screen.blit(top_info, (200, 70))
        if bottom_distance is not None and bottom_angle is not None:
            bottom_info = font.render(f"Bottom: {bottom_distance:.2f}m @ {bottom_angle:.2f}°", True, BLUE)
            screen.blit(bottom_info, (200, 120))

        # Calculate distance to the center line based on top/bottom wave
        lane_deviation = (top_distance - bottom_distance) / 2  # Signed distance to center
        distance_to_center = lane_deviation
        center_info = font.render(f"Distance to Center: {distance_to_center:.2f}m", True, WHITE)
        screen.blit(center_info, (200, 170))

        # Store distances for plotting
        top_distances.append(top_distance)
        bottom_distances.append(bottom_distance)
        last_top_distance = top_distance
        last_bottom_distance = bottom_distance

        # Draw the exit button
        pygame.draw.rect(screen, (255, 0, 0), x_button_rect)
        screen.blit(x_button_text, x_button_rect)

        draw_distance_plot(screen, top_distances, bottom_distances)
        draw_omega_plot(screen, user_omega_history, controller_omega_history)

        # Update the display
        pygame.display.flip()
        clock.tick(60)

    # ----------------------------------------------------------------
    # AFTER SIMULATION: PLOTTING
    # ----------------------------------------------------------------
    pygame.quit()

    # Plot distances to top/bottom lanes over time
    plt.figure()
    plt.plot(top_distances, label='Top Lane Distance')
    plt.plot(bottom_distances, label='Bottom Lane Distance')
    plt.xlabel('Time (60 frames/s)')
    plt.ylabel('Shortest Distance (meters)')
    plt.title('Lane Distance Over Time')
    plt.legend()
    plt.show()


    #Plot user vs controller steering commands
    plt.figure()
    plt.xlabel('Time (60 frames/s)')
    plt.ylabel('Omega (rad/s)')
    plt.title('User vs Controller Steering')
    plt.plot(user_omega_history, 'r', label='User Omega')
    plt.plot(controller_omega_history, 'g', label='Controller Omega')
    plt.legend()
    plt.show()

# --------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()