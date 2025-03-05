import pygame
from game import Game

def main():
    # Create an instance of the Game class
    game = Game()
    
    # No need to initialize pygame here, as it's done in game.initialize()
    # pygame.init()
    
    # Start the game loop
    game.run()
    
    # No need to quit pygame here as it's done in game.run()
    # pygame.quit()

if __name__ == "__main__":
    main()