# All libraries used for the environment
# Imports
import gymnasium as gym
import time
import numpy as np
from gymnasium import spaces
import pygame
from sympy.physics.units import action

####################
# Global Variables
####################

# --- Config ---
CELL_SIZE = 40
GRID_W = 10
GRID_H = 10
FPS = 60
STEP_EVERY = 1  # ms between snake steps (lower = faster)
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

        self._clock = None
        # Define action space
        self.action_space = spaces.Discrete(4)  # up, right, down, left

        # Fixed observation space with proper bounds
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )

        # Internal state
        self.snake = None
        self.direction = 1  # start right
        self.food = None
        self._prev_food_dist = None

        self.score = 0
        self._rng = None
        self._last_obs = None

        # Pygame stuff
        self._screen = None
        self._surface = None
        self._pygame_inited = False
        self._last_step_ms = self._now_ms() - int(STEP_EVERY)
        self._pending_dir = 1
        self._just_reset = False
        self.info = {}

    def _get_observation(self):
        """Get current observation with proper error checking"""
        if not self.snake or len(self.snake) == 0:
            # Return zeros if snake doesn't exist
            return np.zeros(10, dtype=np.float32)
        dist_food = self._distance_to_food()

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
            dir_one_hot[0], dir_one_hot[1], dir_one_hot[2], dir_one_hot[3], dist_food,
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
        # throttle movement to STEP_EVERY (ms), always
        now_ms = pygame.time.get_ticks() if pygame.get_init() else int(time.time() * 1000)
        wait_ms = STEP_EVERY - (now_ms - self._last_step_ms)
        if wait_ms > 0:
            if pygame.get_init():
                self._throttle_step()
            else:
                time.sleep(wait_ms / 1000.0)
        self._last_step_ms = pygame.time.get_ticks() if pygame.get_init() else int(time.time() * 1000)

        # keep window responsive
        if pygame.get_init():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
        self.render()
        obs_before = self._get_observation()


        # Validate action and block 180° turns
        if not (0 <= action <= 3):
            action = self.direction
        if action != self._OPPOSITE[self.direction]:
            self.direction = action

        # Move one cell
        dx, dy = self._DIRS[self.direction]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        # Handle walls
        if self.wrap:
            nx %= self.W
            ny %= self.H
        else:
            if nx < 0 or nx >= self.W or ny < 0 or ny >= self.H:
                reward =- 100
                return obs_before, reward, True, False, {"reason": "hit_wall"}

        # Check self-collision (with current body except last tail if we’ll move)
        ate_food = (self.food is not None) and ((nx, ny) == self.food)
        next_body = [(nx, ny)] + self.snake[:-1] if not ate_food else [(nx, ny)] + self.snake
        if (nx, ny) in self.snake[1:] and not ate_food:
            # Would bite itself
            reward = -300.0
            return obs_before, reward, True, False, {"reason": "hit_self"}

        # Apply move
        self.snake.insert(0, (nx, ny))
        if ate_food:
            self.score += 1
            self.food = self._rand_empty_cell()
        else:
            self.snake.pop()

        # Compute reward AFTER moving
        reward, terminated, truncated = self._get_rewards(ate_food)

        self.render()

        return self._get_observation(), reward, terminated, truncated, self.info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start_x = self.W // 2
        start_y = self.H // 2
        self.snake = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
        self.food = self._rand_empty_cell()
        self.direction = 1
        self._pending_dir = 1
        self.score = 0
        self._just_reset = True
        self._last_step_ms = 0
        self.info = {}
        self._prev_food_dist = self._distance_to_food()
        return self._get_observation(), self.info

    def _now_ms(self):
        # Prefer monotonic clock; fall back to pygame if video init is fine
        try:
            if pygame.get_init() and pygame.display.get_init():
                return int(pygame.time.get_ticks())
        except Exception:
            pass
        # Monotonic, cross-platform
        return int(time.perf_counter_ns() // 1_000_000)

    def _throttle_step(self):
        now = self._now_ms()
        elapsed = now - self._last_step_ms
        wait_ms = STEP_EVERY - elapsed

        # Clamp and cast for Windows
        ms = int(max(0, wait_ms))
        if ms > 0:
            try:
                if pygame.get_init():
                    # delay is slightly safer than wait on Windows
                    pygame.time.delay(ms)
                else:
                    time.sleep(ms / 1000.0)
            except Exception:
                # last-ditch fallback
                time.sleep(ms / 1000.0)

        self._last_step_ms = self._now_ms()
    def render(self, mode='human'):
        """Render the game"""
        # Initialize pygame screen only once
        if not self._pygame_inited:
            pygame.init()
            self._screen = pygame.display.set_mode((self.W * CELL_SIZE, self.H * CELL_SIZE))
            self._surface = pygame.Surface(self._screen.get_size())
            pygame.display.set_caption("Snake Game")
            self._pygame_inited = True
            self._clock = pygame.time.Clock()

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

    def _distance_to_food(self):
        hx, hy = self.snake[0]
        fx, fy = self.food if self.food is not None else (-1, -1)

        # Manhattan:
        return abs(fx - hx) + abs(fy - hy)

    def _get_rewards(self, ate_food: bool):
        reward = 0.0
        terminated = False
        truncated = False

        head_x, head_y = self.snake[0]


        # Distance shaping: reward moving closer, punish moving away
        curr_dist = self._distance_to_food()
        if self._prev_food_dist is not None:
            delta = float(self._prev_food_dist - curr_dist)  # positive if closer
            reward += 1 * delta
        self._prev_food_dist = curr_dist

        # encourage for staying alive
        reward += 1

        # Apple eaten bonus
        if ate_food:
            reward += 10.0 + 0.5 * len(self.snake)  # scaled a bit by length

        return reward, terminated, truncated

    def close(self):
        """Clean up pygame resources"""
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False