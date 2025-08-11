#!/usr/bin/env python3
import os
import sys
import random
import pygame

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

# Avoid ALSA spam on some Linux boxes
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

def draw_grid(surf):
    for x in range(0, GRID_W * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(surf, GRID, (x, 0), (x, GRID_H * CELL_SIZE))
    for y in range(0, GRID_H * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(surf, GRID, (0, y), (GRID_W * CELL_SIZE, y))

def rand_free_cell(occupied):
    while True:
        c = (random.randrange(GRID_W), random.randrange(GRID_H))
        if c not in occupied:
            return c

def add_pos(a, b):
    return (a[0] + b[0], a[1] + b[1])

def wrap_pos(p):
    return (p[0] % GRID_W, p[1] % GRID_H)

def main():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((GRID_W * CELL_SIZE, GRID_H * CELL_SIZE))
    pygame.display.set_caption("🐍 Snake — pygame")
    try:
        font = pygame.font.SysFont("consolas,menlo,monospace", 22)
        big  = pygame.font.SysFont("consolas,menlo,monospace", 36, bold=True)
    except Exception:
        font = pygame.font.Font(None, 22)
        big  = pygame.font.Font(None, 36)

    # Directions
    UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)
    opposite = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

    def reset():
        start = (GRID_W // 2, GRID_H // 2)
        snake = [start, add_pos(start, (-1, 0)), add_pos(start, (-2, 0))]
        direction = RIGHT
        food = rand_free_cell(set(snake))
        score = 0
        step_timer = 0
        paused = False
        alive = True
        return snake, direction, food, score, step_timer, paused, alive

    snake, direction, food, score, step_timer, paused, alive = reset()
    pending_dir = direction

    running = True
    while running:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key in (pygame.K_p, pygame.K_SPACE):
                    if alive:
                        paused = not paused
                elif event.key == pygame.K_r:
                    snake, direction, food, score, step_timer, paused, alive = reset()
                    pending_dir = direction
                elif event.key in (pygame.K_UP, pygame.K_w):
                    if direction != opposite[UP]:
                        pending_dir = UP
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    if direction != opposite[DOWN]:
                        pending_dir = DOWN
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    if direction != opposite[LEFT]:
                        pending_dir = LEFT
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    if direction != opposite[RIGHT]:
                        pending_dir = RIGHT

        # Update
        if alive and not paused:
            step_timer += dt
            if step_timer >= STEP_EVERY:
                step_timer = 0
                # commit direction once per step (prevents double-turn bugs)
                if pending_dir != opposite[direction]:
                    direction = pending_dir

                head = snake[0]
                nxt = add_pos(head, direction)
                if WRAP:
                    nxt = wrap_pos(nxt)

                # collisions
                hit_wall = not (0 <= nxt[0] < GRID_W and 0 <= nxt[1] < GRID_H)
                hit_self = (nxt in snake)
                if (hit_wall and not WRAP) or hit_self:
                    alive = False
                else:
                    snake.insert(0, nxt)
                    if nxt == food:
                        score += 1
                        food = rand_free_cell(set(snake))
                    else:
                        snake.pop()

        # Draw
        screen.fill(BG)
        draw_grid(screen)

        # food
        fx, fy = food
        pygame.draw.rect(
            screen, FOOD,
            pygame.Rect(fx * CELL_SIZE + 2, fy * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4),
            border_radius=6
        )

        # snake
        for i, (x, y) in enumerate(snake):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = SNAKE_HEAD if i == 0 else SNAKE_BODY
            pygame.draw.rect(screen, color, rect, border_radius=6)
            # a tiny inner inset for body segments
            if i != 0:
                inset = rect.inflate(-8, -8)
                pygame.draw.rect(screen, (color[0]//2, color[1]//2, color[2]//2), inset, border_radius=4)

        # HUD
        hud = font.render(f"Score: {score}   Speed: {round(1000/STEP_EVERY,1)} steps/s   P=pause  R=restart  Esc=quit", True, TEXT)
        screen.blit(hud, (10, 8))

        if paused and alive:
            t = big.render("Paused", True, ACCENT)
            screen.blit(t, t.get_rect(center=screen.get_rect().center))

        if not alive:
            over  = big.render("Game Over", True, TEXT)
            tip   = font.render("Press R to restart or Esc to quit", True, TEXT)
            screen.blit(over, over.get_rect(center=(screen.get_width()//2, screen.get_height()//2 - 20)))
            screen.blit(tip,  tip.get_rect(center=(screen.get_width()//2, screen.get_height()//2 + 18)))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
