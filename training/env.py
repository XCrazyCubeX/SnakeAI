# All libraries used for the environment
# Imports
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame


# Initialize Pygame
pygame.init()

# Frame name
pygame.display.set_caption('Flappy Bird')


####################
# Global Variables
####################


# --- Config ---
CELL_SIZE   = 24
GRID_W      = 28
GRID_H      = 22
FPS         = 60
STEP_EVERY  = 110  # ms between snake steps (lower = faster)
WRAP        = False  # True = go through walls, False = die on walls

# Colors
BG          = (18, 18, 22)
GRID        = (28, 28, 34)
SNAKE_HEAD  = (80, 200, 255)
SNAKE_BODY  = (50, 160, 210)
FOOD        = (255, 110, 120)
TEXT        = (230, 230, 235)
ACCENT      = (120, 200, 130)

# Observation Shape
N_CHANNELS = 3
HEIGHT = GRID_H
WIDTH = GRID_W



##########################
# Snake Environment
# Includes:
# Step,
# Observations,
# Render,
# Rewards,
# And reset,
##########################

class Snake(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize flappy bird game, make sure all self values exist
    # Here we also define our observation space and action space

    def __init__(self):
        super(Snake, self).__init__()

        # Define action space
        self.action_space = spaces.Discrete(4)  # up, right, down, left

        self.screen = pygame.display.set_mode(())

        # Observation space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(0,), dtype=np.int32)



    # Get observation
    # Add more observations!!

    def _get_observation(self):
        ...



    # Step function
    # Return observation, reward, done, info
    # Step equals every action happened in game
    # Frame by frame

    def step(self, action):
        ...



    # Reset function
    # Makes sure the game starts with all observations
    # Resets everything

    def reset(self, seed=None, options=None):
        ...




    # Render in game objects
    # User interface User experience

    def render(self, mode='human'):
        ...



    # Defining rewards
    # Using function inside the step function
    # Keeps checking for rewards

    def reward_value(self):
        ...
