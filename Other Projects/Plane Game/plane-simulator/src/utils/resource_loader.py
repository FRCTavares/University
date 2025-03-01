import os
import pygame

def load_texture(file_path):
    """
    Load a texture from a file and return it as a pygame surface.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Texture file not found: {file_path}")
    texture = pygame.image.load(file_path)
    return texture

def load_model(file_path):
    """
    Load a 3D model from a file.
    This function should be implemented to handle the specific model format.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    # Placeholder for model loading logic
    model = None
    return model

def load_sound(file_path):
    """
    Load a sound from a file and return it as a pygame sound object.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Sound file not found: {file_path}")
    sound = pygame.mixer.Sound(file_path)
    return sound