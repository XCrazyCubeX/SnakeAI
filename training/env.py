# All libraries used for the environment
# Imports
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

# Initialize Pygame
pygame.init()

# Frame name
pygame.display.set_caption('Snake Game')

####################
# Global Variables
####################

# --- Config ---
CELL_SIZE = 40
GRID_W = 10
GRID_H = 10
FPS = 60
STEP_EVERY = 150  # ms between snake steps (lower = faster)
WRAP = False  # True = go through walls, False = die on walls

# Colors
BG = (18, 18, 22)
GRID = (28, 28, 34)
SNAKE_HEAD = (80, 200, 255)
SNAKE_BODY = (50, 160, 210)
FOOD = (255, 110, 120)
TEXT = (230, 230, 235)
ACCENT = (120, 200, 130)

# Observation Shape
N_CHANNELS = 3
HEIGHT = GRID_H
WIDTH = GRID_W


##########################
# Snake Environment
##########################

class Snake(gym.Env):
    """
    Snake environment with proper observations.
    Each env.step(action) moves the snake exactly 1 cell.
    """
    metadata = {'render.modes': ['human']}

    # actions: 0=Up, 1=Right, 2=Down, 3=Left
    _DIRS = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])
    _OPPOSITE = {0: 2, 2: 0, 1: 3, 3: 1}

    def __init__(self, grid_w=GRID_W, grid_h=GRID_H, wrap=False):
        """
        Initialize snake game with proper observation and action spaces
        """
        super(Snake, self).__init__()

        self.W = int(grid_w)
        self.H = int(grid_h)
        self.wrap = bool(wrap)

        # Define action space
        self.action_space = spaces.Discrete(4)  # up, right, down, left

        # Fixed observation space with proper bounds
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )

        # Internal state
        self.snake = None
        self.direction = 1  # start right
        self.food = None
        self.score = 0
        self._rng = None
        self._last_obs = None

        # Pygame stuff
        self._screen = None
        self._surface = None
        self._pygame_inited = False
        self._last_step_ms = 0
        self._pending_dir = 1
        self._just_reset = False
        self.info = {}

    def _get_observation(self):
        """Get current observation with proper error checking"""
        if not self.snake or len(self.snake) == 0:
            # Return zeros if snake doesn't exist
            return np.zeros(10, dtype=np.float32)

        hx, hy = self.snake[0]
        fx, fy = self.food if self.food is not None else (-1, -1)

        # Calculate relative distances
        dx, dy = fx - hx, fy - hy

        # Distance to walls
        dist_left = float(hx)
        dist_right = float(self.W - 1 - hx)
        dist_up = float(hy)
        dist_down = float(self.H - 1 - hy)

        # Direction one-hot encoding
        dir_one_hot = np.zeros(4, dtype=np.float32)
        if 0 <= self.direction < 4:
            dir_one_hot[self.direction] = 1.0

        # Create observation array with proper dtype
        obs = np.array([
            float(dx), float(dy),
            dist_left, dist_right, dist_up, dist_down,
            dir_one_hot[0], dir_one_hot[1], dir_one_hot[2], dir_one_hot[3]
        ], dtype=np.float32)

        # Check for NaN or infinite values
        if not np.isfinite(obs).all():
            print(f"WARNING: Non-finite observation detected: {obs}")
            # Replace NaN/inf with zeros
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    def _rand_empty_cell(self):
        """Generate random empty cell not occupied by snake"""
        max_attempts = self.W * self.H
        attempts = 0

        while attempts < max_attempts:
            p = (np.random.randint(0, self.W), np.random.randint(0, self.H))
            if p not in self.snake:
                return p
            attempts += 1

        # Fallback if no empty cell found (shouldn't happen in normal gameplay)
        return (0, 0)

    def step(self, action):
        """Execute one step in the environment"""
        # Get current observation first
        obs = self._get_observation()

        # Validate action
        if not (0 <= action <= 3):
            action = 1  # Default to right if invalid action

        # Accept user input, block 180° turns
        if (not self._just_reset) and (action in (0, 1, 2, 3)) \
                and (action != self._OPPOSITE[self.direction]):
            self._pending_dir = action

        # Time-based movement
        now = pygame.time.get_ticks()
        if now - self._last_step_ms < STEP_EVERY:
            if hasattr(self, 'render'):
                self.render()
            return obs, 0.0, False, False, {}

        self._last_step_ms = now
        self.direction = self._pending_dir

        # Compute next head position
        dx, dy = self._DIRS[self.direction]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        # Handle walls
        if self.wrap:
            nx %= self.W
            ny %= self.H
        else:
            if nx < 0 or nx >= self.W or ny < 0 or ny >= self.H:
                # Game over - hit wall
                reward = -10.0
                return obs, reward, True, False, {"reason": "hit_wall"}

        # Check self-collision
        if (nx, ny) in self.snake:
            # Game over - hit self
            reward = -10.0
            return obs, reward, True, False, {"reason": "hit_self"}

        # Move snake
        ate_food = (self.food is not None) and ((nx, ny) == self.food)
        self.snake.insert(0, (nx, ny))

        if ate_food:
            self.score += 1
            self.food = self._rand_empty_cell()
            reward = 10.0  # Reward for eating food
        else:
            self.snake.pop()
            reward = -0.1  # Small negative reward to encourage efficiency

        if self._just_reset:
            self._just_reset = False

        # Get rewards and check game state
        additional_reward, terminated, truncated = self._get_rewards()
        total_reward = reward + additional_reward

        if hasattr(self, 'render'):
            self.render()

        return self._get_observation(), total_reward, terminated, truncated, self.info

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Starting snake position
        start_x = self.W // 2
        start_y = self.H // 2

        self.snake = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
        self.food = self._rand_empty_cell()
        self.direction = 1  # start moving right
        self._pending_dir = 1
        self.score = 0
        self._last_step_ms = pygame.time.get_ticks()
        self._just_reset = True
        self.info = {}

        return self._get_observation(), self.info

    def render(self, mode='human'):
        """Render the game"""
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

        # Draw snake
        if self.snake:
            for i, (x, y) in enumerate(self.snake):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = SNAKE_HEAD if i == 0 else SNAKE_BODY
                pygame.draw.rect(self._surface, color, rect, border_radius=6)
                if i != 0:
                    inset = rect.inflate(-8, -8)
                    pygame.draw.rect(self._surface, (color[0] // 2, color[1] // 2, color[2] // 2), inset,
                                     border_radius=4)

        # Draw food
        if self.food:
            fx, fy = self.food
            pygame.draw.rect(
                self._surface, FOOD,
                pygame.Rect(fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                border_radius=6
            )

        # HUD (Score)
        font = pygame.font.SysFont("consolas,menlo,monospace", 22)
        hud = font.render(f"Score: {self.score}", True, TEXT)
        self._surface.blit(hud, (10, 8))

        # Blit to screen and update
        self._screen.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _get_rewards(self):
        """Calculate rewards, termination, and truncation conditions"""
        reward = 0.0
        terminated = False
        truncated = False

        distance_to_apple = self.snake[0] - self.food

        if distance_to_apple < 10:
            reward =+ 3

        if self.snake[0] == self.food:
            reward =+ 50



        return reward, terminated, truncated

    def close(self):
        """Clean up pygame resources"""
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False