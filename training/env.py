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
CELL_SIZE   = 40
GRID_W      = 10
GRID_H      = 10
FPS         = 60
STEP_EVERY  = 200  # ms between snake steps (lower = faster)
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
                                            shape=(0,), dtype=np.uint8) # Use of uint8 instead of int8.

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



    # Reset function
    # Makes sure the game starts with all observations
    # Resets everything

    def reset(self, seed=None, options=None):
        """
        :param seed:
        :param options:
        :return: return self._get_observation(), self.info
        """

        # Starting snake position
        start_x = self.W // 2
        start_y = self.H // 2
        self.snake = [(start_x, start_y), (start_x-1, start_y), (start_x-2, start_y)] # [head] [tail1] [tail2]

        # head only at start
        # Starting food position

        self.food = (np.random.randint(0, self.W), np.random.randint(0, self.H))

        self.direction = 1  # start moving right
        self.score = 0
        self._rng = np.random.default_rng()
        self._last_obs = None

        # For rendering
        self._screen = None
        self._surface = None
        self._pygame_inited = False

        return None, {}  # Gymnasium requires (self.obs, self.info)




    # Render in game objects
    # User interface User experience
    
    def render(self, mode='human'):
        
        # Initialize pygame screen only once
        if not self._pygame_inited:
            pygame.init()
            self._screen = pygame.display.set_mode((self.W * CELL_SIZE, self.H * CELL_SIZE))
            self._surface = pygame.Surface(self._screen.get_size())
            pygame.display.set_caption("Snake Game")
            self._pygame_inited = True

        # Fill background
        self._surface.fill(BG)



        # Draw grid lines
        for x in range(0, self.W * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(self._surface, GRID, (x, 0), (x, self.H * CELL_SIZE))
        for y in range(0, self.H * CELL_SIZE, CELL_SIZE):
            pygame.draw.line(self._surface, GRID, (0, y), (self.W * CELL_SIZE, y))



        # Draw snake this includes the tail
        if self.snake:
            for i, (x, y) in enumerate(self.snake):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = SNAKE_HEAD if i == 0 else SNAKE_BODY
                pygame.draw.rect(self._surface, color, rect, border_radius=6)
                # a tiny inner inset for body segments
                if i != 0:
                    inset = rect.inflate(-8, -8)
                    pygame.draw.rect(self._surface, (color[0] // 2, color[1] // 2, color[2] // 2), inset, border_radius=4)



        # Draw food
        if self.food:
            fx, fy = self.food
            pygame.draw.rect(
                self._surface, FOOD,
                pygame.Rect(fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                border_radius = 6

            )

        # HUD ( Score )
        font = pygame.font.SysFont("consolas,menlo,monospace", 22)
        hud = font.render(f"Score: {self.score}", True, TEXT)
        self._surface.blit(hud, (10, 8))


        # Blit to screen and update ( surface to screen )
        self._screen.blit(self._surface, (0, 0))
        pygame.display.flip()

        # Optional: slow down for human view
        if mode == "human":
            pygame.time.wait(int(1000 / FPS))




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


# Extended code - Windows Hotfix
if __name__ == "__main__":
    try:
        print("✅ Snake environment loaded successfully!")

        env = Snake()
        env.reset()  # initialize snake & food

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.render()
            

        pygame.quit()

    except Exception as e:
        print("❌ Failed to load Snake environment!")
        print("Error:", e)


