import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
import pygame
import sys
import math
import numpy as np
import random

# Bullet class (as previously defined)
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, theta):
        super().__init__()
        self.image = pygame.Surface((10, 5))
        self.image.fill((255, 255, 0))  # Yellow color for bullets
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = 15
        self.theta = theta

    def update(self):
        # Move the bullet in the direction the car is facing
        self.rect.x += self.speed * math.cos(self.theta)
        self.rect.y += self.speed * math.sin(self.theta)


class Zombie(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((25, 25))
        self.image.fill((0, 150, 0))  # Green color for zombies
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(WIDTH)
        self.rect.y = random.randrange(-HEIGHT, 0)
        self.speed = random.uniform(0.5, 1.5)  # Reduced speed range

    def update(self, target_x, target_y):
        # Move towards the car
        dx = target_x - self.rect.x
        dy = target_y - self.rect.y
        dist = math.hypot(dx, dy)
        if dist != 0:
            dx, dy = dx / dist, dy / dist
            self.rect.x += dx * self.speed
            self.rect.y += dy * self.speed


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


# Initialize pygame
pygame.init()

# Screen dimensions
scale=2
WIDTH, HEIGHT = 800*scale, 600*scale

# Colors
WHITE = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Joystick-Like Car Steering")

# Car attributes
car_width, car_height = 100, 40
x, y = WIDTH // 2, HEIGHT // 2
theta, phi = 0, 0
V, omega = 0, 0
L = car_width  # Characteristic length
dt = 1 #60fps == dt=1/60
max_phi = math.radians(30)  # Maximum steering angle in radians

# Initialize a sprite group for bullets before the main loop
bullets = pygame.sprite.Group()

# Initialize zombie group and score before the main loop
zombies = pygame.sprite.Group()
score = 0

# Set up a timer event for spawning zombies
SPAWN_ZOMBIE = pygame.USEREVENT + 1
pygame.time.set_timer(SPAWN_ZOMBIE, 2000)  # Spawn every 2 seconds

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)  # Clear the screen
    initial_state = np.array([x, y, theta, phi])

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == SPAWN_ZOMBIE:
            zombie = Zombie()
            zombies.add(zombie)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Bullet starting position at the front of the car
                bullet_x = x + (car_width / 2) * math.cos(theta)
                bullet_y = y + (car_width / 2) * math.sin(theta)
                bullet = Bullet(bullet_x, bullet_y, theta)
                bullets.add(bullet)

    # Get keys pressed
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:  # Turn left
        omega = -0.02
    elif keys[pygame.K_RIGHT]:  # Turn right
        omega = +0.02
    else:
        omega = 0  # Reset angular velocity

    if keys[pygame.K_UP]:  # Move forward
        V = 5
    elif keys[pygame.K_DOWN]:  # Move backward
        V = -1.0
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
    wheel_surface = pygame.Surface((car_width/3, car_height/5), pygame.SRCALPHA)
    wheel_surface.fill(BLUE)
    rotated_wheel = pygame.transform.rotate(wheel_surface, -math.degrees(theta) -math.degrees(phi))
    wheel_rect = rotated_wheel.get_rect(center=(x+(L/2-10)*math.cos(theta), y+(L/2-10)*math.sin(theta)))
    screen.blit(rotated_wheel, wheel_rect.topleft)

    # Update zombies and bullets
    bullets.update()
    zombies.update(x, y)  # Pass the car's position

    # Check for collisions between bullets and zombies
    hits = pygame.sprite.groupcollide(zombies, bullets, True, True)
    score += len(hits)  # Increase score

    # Update bullet positions and draw them
    bullets.update()
    bullets.draw(screen)
    zombies.draw(screen)

    # Display the score
    font = pygame.font.SysFont(None, 36)
    score_surface = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_surface, (10, 10))

    # Update the display
    pygame.display.flip()

    # Limit frames per second
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()
max_phi = math.radians(45)  # Maximum steering angle in radians

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(WHITE)  # Clear the screen
    initial_state = np.array([x, y, theta, phi])

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == SPAWN_ZOMBIE:
            zombie = Zombie()
            zombies.add(zombie)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Bullet starting position at the front of the car
                bullet_x = x + (car_width / 2) * math.cos(theta)
                bullet_y = y + (car_width / 2) * math.sin(theta)
                bullet = Bullet(bullet_x, bullet_y, theta)
                bullets.add(bullet)

    # Get keys pressed
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:  # Turn left
        omega = -0.1
    elif keys[pygame.K_RIGHT]:  # Turn right
        omega = +0.1
    else:
        omega = 0  # Reset angular velocity

    if keys[pygame.K_UP]:  # Move forward
        V = 1.0
    elif keys[pygame.K_DOWN]:  # Move backward
        V = -1.0
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
    wheel_surface = pygame.Surface((car_width/3, car_height/5), pygame.SRCALPHA)
    wheel_surface.fill(BLUE)
    rotated_wheel = pygame.transform.rotate(wheel_surface, -math.degrees(theta) -math.degrees(phi))
    wheel_rect = rotated_wheel.get_rect(center=(x+(L/2-10)*math.cos(theta), y+(L/2-10)*math.sin(theta)))
    screen.blit(rotated_wheel, wheel_rect.topleft)

    # Update zombies and bullets
    bullets.update()
    zombies.update(x, y)  # Pass the car's position

    # Check for collisions between bullets and zombies
    hits = pygame.sprite.groupcollide(zombies, bullets, True, True)
    score += len(hits)  # Increase score

    # Update bullet positions and draw them
    bullets.update()
    bullets.draw(screen)
    zombies.draw(screen)

    # Display the score
    font = pygame.font.SysFont(None, 36)
    score_surface = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_surface, (10, 10))

    # Update the display
    pygame.display.flip()

    # Limit frames per second
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()