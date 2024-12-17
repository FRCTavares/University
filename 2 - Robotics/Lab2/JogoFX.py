# Import necessary libraries
import os
import sys
import math
import random
import numpy as np
import pygame

# Use a dummy audio driver to avoid sound issues
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Initialize pygame
pygame.init()

# Screen dimensions and scaling factor
SCALE = 2
WIDTH, HEIGHT = 800 * SCALE, 600 * SCALE

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
YELLOW = (255, 255, 0)
GREEN = (0, 150, 0)
GRAY = (50, 50, 50)
LGRAY = (100, 100, 100)
BLACK = (0, 0, 0)
GREEN_HEALTH = (0, 255, 0)
RED_HEALTH = (255, 0, 0)

# Game constants
CAR_WIDTH, CAR_HEIGHT = 100, 40
MAX_PHI = math.radians(45)  # Maximum steering angle in radians
L = CAR_WIDTH  # Characteristic length
DT = 1 / 60  # Time step for 60 FPS
SPAWN_ZOMBIE_EVENT = pygame.USEREVENT + 1

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Joystick-Like Car Steering")

# Set up a timer event for spawning zombies
pygame.time.set_timer(SPAWN_ZOMBIE_EVENT, 2000)  # Spawn every 2 seconds

# Game variables
x, y = WIDTH // 2, HEIGHT // 2
theta, phi = 0, 0
V, omega = 0, 0
score = 0
health = 100  # Initialize health
turret_angle = 0
TURRET_ROT_SPEED = math.radians(2)  # Turret rotation speed per frame

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Sprite groups
bullets = pygame.sprite.Group()
zombies = pygame.sprite.Group()

# Font
font = pygame.font.SysFont(None, 36)


# Bullet class
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y, theta):
        super().__init__()
        self.image = pygame.Surface((10, 5))
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = 15
        self.theta = theta

    def update(self):
        # Move the bullet in the direction the car is facing
        self.rect.x += self.speed * math.cos(self.theta)
        self.rect.y += self.speed * math.sin(self.theta)
        # Remove the bullet if it goes off-screen
        if not screen.get_rect().collidepoint(self.rect.center):
            self.kill()


# Zombie class
class Zombie(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((25, 25))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(WIDTH)
        self.rect.y = random.randrange(-HEIGHT, 0)
        self.speed = random.uniform(0.5, 1.5)  # Reduced speed range

    def update(self, target_x, target_y):
        # Move towards the target (car)
        dx = target_x - self.rect.centerx
        dy = target_y - self.rect.centery
        dist = math.hypot(dx, dy)
        if dist != 0:
            dx, dy = dx / dist, dy / dist
            self.rect.x += dx * self.speed
            self.rect.y += dy * self.speed


# Functions for car kinematics
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


def draw_car(x, y, theta, phi):
    """
    Draw the car and its components on the screen.
    """
    # Car surface
    car_surface = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
    car_surface.fill(RED)
    # Machine gun on top of the car
    gun_width, gun_height = CAR_WIDTH / 4, CAR_HEIGHT / 4
    gun_surface = pygame.Surface((gun_width, gun_height), pygame.SRCALPHA)
    gun_surface.fill(GRAY)
    gun_rect = gun_surface.get_rect(center=(CAR_WIDTH / 2, CAR_HEIGHT / 2))
    # Rotate the car
    rotated_car = pygame.transform.rotate(car_surface, -math.degrees(theta))
    car_rect = rotated_car.get_rect(center=(x, y))
    screen.blit(rotated_car, car_rect.topleft)
    # Rotate the turret
    rotated_turret = pygame.transform.rotate(gun_surface, -math.degrees(theta + turret_angle))
    turret_rect = rotated_turret.get_rect(center=(x, y))
    screen.blit(rotated_turret, turret_rect.topleft)


def handle_events(state):
    """
    Handle user input and events based on the current state.
    """
    global running, V, omega
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == SPAWN_ZOMBIE_EVENT and state == "game":
            zombie = Zombie()
            zombies.add(zombie)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p and state == "game":
                return "pause"
            elif event.key == pygame.K_p and state == "pause":
                return "game"
            if state == "menu":
                if event.key == pygame.K_RETURN:
                    return "game"
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif state == "game":
                if event.key == pygame.K_SPACE:
                    shoot_bullet()
            elif state == "game_over":
                if event.key == pygame.K_RETURN:
                    reset_game()
                    return "menu"
                elif event.key == pygame.K_ESCAPE:
                    running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and state == "menu":
            mouse_pos = event.pos
            if main_menu_buttons["start"].collidepoint(mouse_pos):
                return "game"
            elif main_menu_buttons["quit"].collidepoint(mouse_pos):
                running = False
            elif main_menu_buttons["scores"].collidepoint(mouse_pos):
                pass  # Feature to be implemented
        elif event.type == pygame.MOUSEBUTTONDOWN and state == "pause":
            mouse_pos = event.pos
            # Define button parameters
            button_width, button_height = 200, 50
            button_x = WIDTH / 2 - button_width / 2
            button_y = HEIGHT / 2 + 50
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            if button_rect.collidepoint(mouse_pos):
                return "menu"
    return state


def shoot_bullet():
    """
    Shoot a bullet from the car's current position and orientation.
    """
    gun_length = CAR_HEIGHT // 2
    bullet_x = x + gun_length * math.cos(theta + turret_angle)
    bullet_y = y + gun_length * math.sin(theta + turret_angle)
    bullet = Bullet(bullet_x, bullet_y, theta + turret_angle)
    bullets.add(bullet)


def update_game_state():
    """
    Update the game state, including positions and handling collisions.
    """
    global x, y, theta, phi, score, health
    # Get keys pressed
    keys = pygame.key.get_pressed()
    handle_movement(keys)
    # Simulate motion
    initial_state = np.array([x, y, theta, phi])
    x, y, theta, phi = simulate_motion(initial_state, V, omega, L, DT, MAX_PHI)
    # Update sprites
    bullets.update()
    zombies.update(x, y)
    # Check for bullet collisions
    hits = pygame.sprite.groupcollide(zombies, bullets, True, True)
    score += len(hits)
    # Check for zombie collisions with the car
    for zombie in zombies:
        if zombie.rect.colliderect(pygame.Rect(x - CAR_WIDTH//2, y - CAR_HEIGHT//2, CAR_WIDTH, CAR_HEIGHT)):
            zombie.kill()
            health -= 10
            if health <= 0:
                return "game_over"
    return "game"


def handle_movement(keys):
    """
    Update the car's velocity based on user input.
    """
    global V, omega, turret_angle
    if keys[pygame.K_LEFT]:
        omega = -2
    elif keys[pygame.K_RIGHT]:
        omega = 2
    else:
        omega = 0
    if keys[pygame.K_UP]:
        V = 200
    elif keys[pygame.K_DOWN]:
        V = -200
    else:
        V = 0

    # Rotate turret
    if keys[pygame.K_a]:
        turret_angle -= TURRET_ROT_SPEED
    if keys[pygame.K_d]:
        turret_angle += TURRET_ROT_SPEED
    # Constrain turret_angle between -max_turret_angle and max_turret_angle
    max_turret_angle = math.radians(90)
    turret_angle = max(-max_turret_angle, min(max_turret_angle, turret_angle))


def draw_game_elements():
    """
    Draw all game elements on the screen.
    """
    # Clear the screen
    screen.fill(LGRAY)
    # Draw car
    draw_car(x, y, theta, phi)
    # Draw bullets and zombies
    bullets.draw(screen)
    zombies.draw(screen)
    # Display the score
    score_surface = font.render(f"Score: {score}", True, BLACK)
    screen.blit(score_surface, (10, 10))
    # Draw health bar
    health_bar_length = 200
    health_bar_height = 20
    health_ratio = max(health, 0) / 100
    pygame.draw.rect(screen, RED_HEALTH, (WIDTH - health_bar_length - 10, 10, health_bar_length, health_bar_height))
    pygame.draw.rect(screen, GREEN_HEALTH, (WIDTH - health_bar_length - 10, 10, health_bar_length * health_ratio, health_bar_height))
    health_text = font.render(f"Health: {health}", True, BLACK)
    screen.blit(health_text, (WIDTH - health_bar_length - 10, 40))


def draw_main_menu():
    """
    Draw the main menu on the screen with three buttons.
    """
    screen.fill(BLACK)
    title_font = pygame.font.SysFont(None, 72)
    button_font = pygame.font.SysFont(None, 48)
    
    # Title
    title_surface = title_font.render("Car Zombie Shooter", True, WHITE)
    screen.blit(title_surface, title_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 150)))
    
    # Define button dimensions
    button_width, button_height = 300, 80
    button_x = WIDTH / 2 - button_width / 2
    start_button_y = HEIGHT / 2 - 50
    scores_button_y = HEIGHT / 2 + 50
    quit_button_y = HEIGHT / 2 + 150
    
    # Create button rectangles
    start_button_rect = pygame.Rect(button_x, start_button_y, button_width, button_height)
    scores_button_rect = pygame.Rect(button_x, scores_button_y, button_width, button_height)
    quit_button_rect = pygame.Rect(button_x, quit_button_y, button_width, button_height)
    
    # Draw buttons
    pygame.draw.rect(screen, LGRAY, start_button_rect)
    pygame.draw.rect(screen, LGRAY, scores_button_rect)
    pygame.draw.rect(screen, LGRAY, quit_button_rect)
    
    # Render button text
    start_text = button_font.render("Start Game", True, BLACK)
    scores_text = button_font.render("See Past Scores", True, BLACK)
    quit_text = button_font.render("Quit Game", True, BLACK)
    
    # Blit text onto buttons
    screen.blit(start_text, start_text.get_rect(center=start_button_rect.center))
    screen.blit(scores_text, scores_text.get_rect(center=scores_button_rect.center))
    screen.blit(quit_text, quit_text.get_rect(center=quit_button_rect.center))
    
    # Store button rects for event handling
    global main_menu_buttons
    main_menu_buttons = {
        "start": start_button_rect,
        "scores": scores_button_rect,
        "quit": quit_button_rect
    }


def draw_game_over():
    """
    Draw the game over screen on the screen.
    """
    screen.fill(BLACK)
    game_over_font = pygame.font.SysFont(None, 72)
    score_font = pygame.font.SysFont(None, 48)
    game_over_surface = game_over_font.render("GAME OVER", True, RED)
    final_score_surface = score_font.render(f"Final Score: {score}", True, WHITE)
    retry_surface = score_font.render("Press ENTER to Return to Menu", True, WHITE)
    quit_surface = score_font.render("Press ESC to Quit", True, WHITE)
    screen.blit(game_over_surface, game_over_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 100)))
    screen.blit(final_score_surface, final_score_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2)))
    screen.blit(retry_surface, retry_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 50)))
    screen.blit(quit_surface, quit_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 100)))


def draw_pause_menu():
    """
    Draw the pause menu on the screen.
    """
    pause_font = pygame.font.SysFont(None, 72)
    prompt_font = pygame.font.SysFont(None, 48)
    button_font = pygame.font.SysFont(None, 36)
    
    pause_surface = pause_font.render("PAUSED", True, WHITE)
    resume_surface = prompt_font.render("Press P to Resume", True, WHITE)
    
    # Button parameters
    button_width, button_height = 200, 50
    button_x = WIDTH / 2 - button_width / 2
    button_y = HEIGHT / 2 + 50
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    
    quit_surface = button_font.render("Quit to Menu", True, BLACK)
    
    # Draw pause text and resume prompt
    screen.blit(pause_surface, pause_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 50)))
    screen.blit(resume_surface, resume_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 10)))
    
    # Draw button
    pygame.draw.rect(screen, GRAY, button_rect)
    screen.blit(quit_surface, quit_surface.get_rect(center=button_rect.center))


def reset_game():
    """
    Reset the game variables to their initial states.
    """
    global x, y, theta, phi, V, omega, score, health, bullets, zombies
    x, y = WIDTH // 2, HEIGHT // 2
    theta, phi = 0, 0
    V, omega = 0, 0
    score = 0
    health = 100
    bullets.empty()
    zombies.empty()


def main():
    """
    Main function to run the game.
    """
    global running
    running = True
    state = "menu"

    while running:
        clock.tick(60)  # Limit frames per second
        state = handle_events(state)

        if state == "menu":
            draw_main_menu()
        elif state == "game":
            state = update_game_state()
            draw_game_elements()
        elif state == "game_over":
            draw_game_over()
        elif state == "pause":
            draw_pause_menu()

        pygame.display.flip()  # Update the display

    # Clean up
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()