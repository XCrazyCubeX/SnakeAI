# All libraries used for the environment
# Imports

import gymnasium as gym
import time
import numpy as np
from gymnasium import spaces
import pygame

####################
# Global Variables
####################

# --- Config ---
CELL_SIZE = 40
GRID_W = 10
GRID_H = 10
FPS = 60
STEP_EVERY = 60 # ms between snake steps (lower = faster)
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


# actions: 0=Up, 1=Right, 2=Down, 3=Left
_DIRS = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])
# 0:up,1:right,2:down,3:left
OPPOSITE = (2, 3, 0, 1)


# some sort of windows hotfix
# makes sure the program does not use too much floats.
# instead using ints only

def _int_scalar(x):
    a = np.asarray(x)
    if a.shape == ():  # 0-D numpy -> python int
        return int(a.item())
    if a.ndim == 1 and a.size == 1:  # shape (1,)
        return int(a[0])
    # decode one-hot accidentally passed in
    if a.ndim == 1 and a.size == 4 and ((a == 0) | (a == 1)).all():
        return int(a.argmax())
    return int(a)  # last resort



# Snake Environment
# This is the full snake Environment.
#

class Snake(gym.Env):
    """
    Snake environment with proper observations.
    Each env.step(action) moves the snake exactly 1 cell.
    """

    def __init__(self, throttle=False, grid_w=GRID_W, grid_h=GRID_H, wrap=False):
        super(Snake, self).__init__()

        """
        Initialize snake game with proper observation and action spaces
        :param grid_w: width of grid
        :param grid_h: height of grid
        :param wrap: wrap around the game
        """
        self.render_mode = "human"
        self.throttle = throttle

        # Define action space and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        # Dimensions
        self.W = int(grid_w)
        self.H = int(grid_h)
        self.wrap = bool(wrap)

        # Internal state
        self.snake = None
        self.food = None
        self.score = 0
        self.direction = 1  # make sure it's a plain int
        self._rng = None
        self._last_obs = None


        # Pygame stuff
        self._clock = None
        self._screen = None
        self._surface = None
        self._pygame_inited = False
        self._last_step_ms = self._now_ms() - int(STEP_EVERY)
        self._pending_dir = 1
        self._just_reset = False
        self.info = {}

        self._DIRS = _DIRS





    def _get_observation(self):
        """
        Get current observation with proper error checking
        :param self: observation
        :return obs
        """

        if not self.snake or len(self.snake) == 0:
            # Return zeros if snake doesn't exist
            return np.zeros(13, dtype=np.float32)


        # hx = HeadX, hy = HeadY
        # fx = FoodX, fy = FoodY
        hx, hy = self.snake[0]
        fx, fy = self.food if self.food is not None else (-1, -1)


        #distance to food
        dist_food = self._distance_to_food()

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
        # 13 Observations
        obs = np.array([
            hx,
            hy,
            fx,
            fy,
            dist_left,
            dist_right,
            dist_up,
            dist_down,
            dir_one_hot[0],
            dir_one_hot[1],
            dir_one_hot[2],
            dir_one_hot[3],
            dist_food,
        ], dtype=np.float32)


        # Check for NaN or infinite values
        # I have no idea, it works
        if not np.isfinite(obs).all():
            print(f"WARNING: Non-finite observation detected: {obs}")
            # Replace NaN/inf with zeros
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs




    def step(self, action):

        if self.throttle:
            self._throttle_step()

        # keep window responsive
        if self.render_mode == "human":
            self.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()



        # Validate action and block 180° turns
        action = _int_scalar(action)
        self.direction = _int_scalar(self.direction)

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}, expected 0..3")
        if action != OPPOSITE[self.direction]:
            self.direction = action


        dx, dy = self._DIRS[self.direction]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy


        ate_food = (self.food is not None) and ((nx, ny) == self.food)
        reward, terminated, truncated = self._get_rewards(ate_food)
        if terminated:
            return self._get_observation(), float(reward), True, False, dict(self.info)


        # Apply move
        self.snake.insert(0, (nx, ny))

        if ate_food:
            self.score += 1
            self.food = self._rand_empty_cell()
        else:
            self.snake.pop()



        return self._get_observation(), reward, terminated, truncated, dict(self.info)




    def reset(self, seed=None, options=None):
        """
        Reset the environment and return an initial observation.
        :param seed:
        :param options:
        :return:
        """
        super().reset(seed=seed)
        start_x = self.W // 2
        start_y = self.H // 2
        self.snake = [(start_x, start_y), (start_x - 1, start_y), (start_x - 2, start_y)]
        self.food = self._rand_empty_cell()
        self.direction = 1
        self._pending_dir = 1
        self.score = 0
        self._just_reset = True
        self.info = {}
        return self._get_observation(), self.info




    def render(self, mode='human'):
        """
        Render in everything using pygame
        :param mode:
        :return:
        """
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




    def _get_rewards(self, ate_food: bool):
        """
        Get rewards based on certain conditions
        :param ate_food:
        :return: reward, terminated, truncated
        """
        dx, dy = self._DIRS[self.direction]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        ate_food = (self.food is not None) and ((nx, ny) == self.food)
        reward = 0.0
        terminated = False
        truncated = False


        # Apple eaten bonus
        if ate_food:
            reward += len(self.snake)  # scaled a bit by length

        # Handle walls
        if self.wrap:
            nx %= self.W
            ny %= self.H
        else:
            if nx < 0 or nx >= self.W or ny < 0 or ny >= self.H:
                terminated = True
        if (nx, ny) in self.snake[1:] and not ate_food:
            terminated = True


        if terminated:
            reward -= 50


        reward -= 0.01

        # use proximity distance to grant greater reward for closer distance
        # just a helper
        curr_dist = self._distance_to_food()
        prox = self._proximity_reward(curr_dist, near=1, far=10, max_r=10.0)

        reward += prox // 10

        return reward, terminated, truncated




    def close(self):
        """Clean up pygame resources"""
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False







    # utilities/functions
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
        return(0, 0)

    def _distance_to_food(self):
        hx, hy = self.snake[0]
        fx, fy = self.food if self.food is not None else (-1, -1)

        # Manhattan:
        return abs(fx - hx) + abs(fy - hy)

    def _throttle_step(self):
        now = self._now_ms()
        elapsed = now - self._last_step_ms
        if elapsed < 0 or elapsed > 10 * STEP_EVERY:
            self._last_step_ms = now - int(STEP_EVERY)
            elapsed = STEP_EVERY

        wait_ms = int(max(0, min(STEP_EVERY - elapsed, 100)))  # cap 100ms
        if wait_ms > 0:
            time.sleep(wait_ms / 1000.0)  # één pad, geen pygame delay nodig

        self._last_step_ms = self._now_ms()

    # --- vervang _now_ms geheel ---
    def _now_ms(self):
        return int(time.perf_counter_ns() // 1_000_000)  # monotonic, cross-platform

    def _proximity_reward(self, dist, near=1, far=10, max_r=10.0):
        # near distance (≤1) gets full reward; far (≥10) gets 0; linear in between
        if dist <= near:
            return float(max_r)
        if dist >= far:
            return 0.0
        return float(max_r * (far - dist) / (far - near))  # e.g. 10 * (10 - d) / 9

