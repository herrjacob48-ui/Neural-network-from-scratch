"""
CONTROLS:
  T         - toggle training on/off
  H         - toggle control: NN vs. the perfect heuristic (for comparison)
  R         - reset the network (re-randomize weights) and game stats
  SPACE     - hold to fast-forward (multiple train steps per rendered frame)
  ESC / Q   - quit

Requires: pip install pygame numpy
Run:      python game.py
"""

import random
from collections import deque

import numpy as np
import pygame

from network import Network, sigmoid

# ---------------------------------------------------------------- constants
WIDTH, HEIGHT = 640, 420
PADDLE_W, PADDLE_H = 80, 14
PADDLE_Y = HEIGHT - 40
BALL_R = 9
FPS = 60

BALL_SPEED_START = 3.0
BALL_SPEED_MAX = 11.0
BALL_SPEED_STEP = 0.15   # speed increase per drop, regardless of catch/miss

PADDLE_MAX_SPEED = 25.0   # px/frame the paddle can move toward its target
FASTFORWARD_STEPS = 25   # training-only steps done per frame while holding SPACE

BG = (18, 18, 24)
FG = (230, 230, 235)
ACCENT = (90, 200, 255)
GOOD = (110, 220, 140)
BAD = (230, 90, 90)
GRID = (40, 40, 50)


def new_ball():
    x = random.uniform(BALL_R, WIDTH - BALL_R)
    return x, -BALL_R


class GameState:
    def __init__(self):
        self.ball_x, self.ball_y = new_ball()
        self.ball_speed = BALL_SPEED_START
        self.paddle_x = WIDTH / 2 - PADDLE_W / 2
        self.catches = 0
        self.misses = 0
        self.loss_history = deque(maxlen=150)
        self.last_cost = 0.0

    def paddle_center(self):
        return self.paddle_x + PADDLE_W / 2

    def reset_ball(self, caught):
        if caught:
            self.catches += 1
        else:
            self.misses += 1
        self.ball_speed = min(BALL_SPEED_MAX, self.ball_speed + BALL_SPEED_STEP)
        self.ball_x, self.ball_y = new_ball()


def build_network():
    # inputs: ball_x (norm), ball_y (norm), paddle_center_x (norm)
    # output: predicted ideal paddle_center_x (norm)
    return Network([3, 16, 16, 1], sigmoid, rate=0.6)


def get_state_vector(game):
    x = np.array([
        [game.ball_x / WIDTH],
        [game.ball_y / HEIGHT],
        [game.paddle_center() / WIDTH],
    ])
    return x


def ideal_target(game):
    # the perfect teacher: paddle center should just equal the ball's x,
    # clamped so it's an achievable paddle position
    half = PADDLE_W / 2
    ideal_center = min(max(game.ball_x, half), WIDTH - half)
    return ideal_center / WIDTH


def extract_cost_scalar(cost):
    return float(np.array(cost).flatten()[0])


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Catch the Ball — NN Edition")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    big_font = pygame.font.SysFont("consolas", 24, bold=True)

    net = build_network()
    game = GameState()

    training = True
    nn_control = True
    fastforward = False

    running = True
    while running:
        # ------------------------------------------------------- events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_t:
                    training = not training
                elif event.key == pygame.K_h:
                    nn_control = not nn_control
                elif event.key == pygame.K_r:
                    net = build_network()
                    game = GameState()
                elif event.key == pygame.K_SPACE:
                    fastforward = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    fastforward = False

        # ---------------------------------------------- fast-forward training
        # runs extra train-only steps (no rendering) so convergence is visible fast
        if fastforward and training:
            for _ in range(FASTFORWARD_STEPS):
                x = get_state_vector(game)
                y = np.array([[ideal_target(game)]])
                cost = net.train(x, y)
                game.last_cost = extract_cost_scalar(cost)
                game.loss_history.append(game.last_cost)

                game.ball_y += game.ball_speed
                if game.ball_y + BALL_R >= PADDLE_Y:
                    caught = game.paddle_x <= game.ball_x <= game.paddle_x + PADDLE_W
                    game.reset_ball(caught)

        # ------------------------------------------------------------ step
        x = get_state_vector(game)

        prediction = net.forward_pass(x)
        predicted_center_norm = float(prediction[0, 0])

        if training:
            y = np.array([[ideal_target(game)]])
            cost = net.train(x, y)
            game.last_cost = extract_cost_scalar(cost)
            game.loss_history.append(game.last_cost)

        # decide where the paddle *wants* to go
        if nn_control:
            target_center_px = predicted_center_norm * WIDTH
        else:
            target_center_px = ideal_target(game) * WIDTH

        # move paddle toward target, capped by max speed (keeps it "physical")
        current_center = game.paddle_center()
        delta = target_center_px - current_center
        delta = max(-PADDLE_MAX_SPEED, min(PADDLE_MAX_SPEED, delta))
        game.paddle_x += delta
        game.paddle_x = max(0, min(WIDTH - PADDLE_W, game.paddle_x))

        # move ball
        game.ball_y += game.ball_speed
        if game.ball_y + BALL_R >= PADDLE_Y:
            caught = game.paddle_x <= game.ball_x <= game.paddle_x + PADDLE_W
            game.reset_ball(caught)

        # ------------------------------------------------------------ draw
        screen.fill(BG)

        # ground line
        pygame.draw.line(screen, GRID, (0, PADDLE_Y + PADDLE_H + 4),
                          (WIDTH, PADDLE_Y + PADDLE_H + 4), 1)

        # ball
        pygame.draw.circle(screen, ACCENT, (int(game.ball_x), int(game.ball_y)), BALL_R)

        # paddle
        pygame.draw.rect(screen, FG, (game.paddle_x, PADDLE_Y, PADDLE_W, PADDLE_H), border_radius=4)

        # loss sparkline (bottom strip)
        if len(game.loss_history) > 1:
            hist = list(game.loss_history)
            max_loss = max(hist) or 1e-6
            points = []
            for i, v in enumerate(hist):
                px = 10 + i * ((WIDTH - 20) / max(1, len(hist) - 1))
                py = HEIGHT - 10 - (v / max_loss) * 50
                points.append((px, py))
            pygame.draw.lines(screen, GOOD, False, points, 2)

        # ---- HUD ----
        total = game.catches + game.misses
        acc = (game.catches / total * 100) if total else 0.0

        lines = [
            f"Control: {'NN' if nn_control else 'HEURISTIC'}   (H to toggle)",
            f"Training: {'ON' if training else 'OFF'}   (T to toggle)   {'[FAST]' if fastforward else ''}",
            f"Catches: {game.catches}   Misses: {game.misses}   Accuracy: {acc:.1f}%",
            f"Loss: {game.last_cost:.5f}   Ball speed: {game.ball_speed:.1f}",
            "R: reset network+game    SPACE: hold to fast-forward training",
        ]
        for i, line in enumerate(lines):
            surf = font.render(line, True, FG)
            screen.blit(surf, (10, 8 + i * 20))

        title = big_font.render("Catch the Ball", True, ACCENT)
        screen.blit(title, (WIDTH - title.get_width() - 10, 8))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()