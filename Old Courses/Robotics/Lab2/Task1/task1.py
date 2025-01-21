import pygame

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Steering Simulation")

# Define car properties
car_width, car_height = 30, 60
car_x = WIDTH // 2 - car_width // 2
car_y = HEIGHT // 2 - car_height // 2
car_speed = 5

# Create a simple car rectangle
car = pygame.Rect(car_x, car_y, car_width, car_height)

# Main loop
running = True
clock = pygame.time.Clock()
while running:
    clock.tick(60)  # Limit to 60 FPS
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Handle key presses
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car.x -= car_speed
    if keys[pygame.K_RIGHT]:
        car.x += car_speed

    # Clear the screen
    window.fill((0, 0, 0))

    # Draw the car
    pygame.draw.rect(window, (255, 0, 0), car)

    # Update the display
    pygame.display.flip()

pygame.quit()