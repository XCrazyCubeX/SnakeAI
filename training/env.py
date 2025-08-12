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
N_CHANNELS  = 3
HEIGHT      = GRID_H
WIDTH       = GRID_W



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
    """
    Snake environment met pixel-observations (obs).
    Each env.step(action) moves the snake exactly 1 cell.
    """
    metadata = {'render.modes': ['human']}

    # actions: 0=Up, 1=Right, 2=Down, 3=Left
    _DIRS = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])
    _OPPOSITE = {0: 2, 2: 0, 1: 3, 3: 1}



    def __init__(self, grid_w=GRID_W, grid_h=GRID_H, wrap=False):
        """
        # Initialize flappy bird game, make sure all self values exist
        # Here we also define our observation space and action space
        """
        super(Snake, self).__init__()

        self.W = int(grid_w)
        self.H = int(grid_h)
        self.wrap = bool(wrap)


        # Define action space
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        # Observation space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(0,), dtype=np.int8)

        # internal state, snake food, score...

        self.snake = None           # list of (x,y), head = snake[0]
        self.direction = 1          # start right
        self.food = None
        self.score = 0
        self._rng = None
        self._last_obs = None

        # some pygame stuff
        # only used when rendering
        self._screen = None
        self._surface = None
        self._pygame_inited = False




    # Get observation
    # Add more observations!!

    def _get_observation(self):
        """
        this is where the observations will be made
        :return: obs
        """

        # obs = np.array([])
        ...



    # Step function
    # Return observation, reward, done, info
    # Step equals every action happened in game
    # Frame by frame

    def step(self, action):
        """
        every frame of the game/every action taken
        :param action:
        :return: obs, reward, terminated, truncated, info
        """
        ...



    # Reset function
    # Makes sure the game starts with all observations
    # Resets everything

    def reset(self, seed=None, options=None):
        """

        :param seed:
        :param options:
        :return: return self._get_observation(), self.info

        """
        ...



    # Render in game objects
    # User interface User experience

    def render(self, mode='human'):
        """
        actually just renders everything, nothing special

        """
        ...



    # Defining rewards
    # Using function inside the step function
    # Keeps checking for rewards

    def _get_rewards(self) -> tuple[int, bool, bool]:
        """
        # A function that gives out rewards or punishments
        #
        # :return: reward, terminated, truncated
        """
        # reward = 0
        # terminated = False
        # truncated = False
        # ...
        # ...
        # ...
        # return reward, terminated, truncated
        ...

