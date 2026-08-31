import math
import random
from collections import deque

import numpy as np
import pygame

from network import Network, sigmoid

# ---------------------------------------------------------------- constants
WIDTH, HEIGHT = 800, 450
FPS = 60

# --- physics (classic cart-pole parameters, SI units) ---
GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
POLE_HALF_LENGTH = 0.5          # meters
POLEMASS_LENGTH = MASSPOLE * POLE_HALF_LENGTH
TAU = 0.02                      # seconds per physics step
MAX_FORCE = 12.0                # Newtons, control authority cap

X_THRESHOLD = 2.2               # meters, episode fails if |x| exceeds this
ANGLE_THRESHOLD = 0.5           # radians (~28.6 deg), episode fails beyond this
EPISODE_LIMIT = 400             # steps; also reset periodically on success,
                                 # so the visible rollout keeps sampling fresh
                                 # random starting perturbations

# --- teacher PD controller gains (tuned to reliably balance) ---
KP_THETA, KD_THETA = 20.0, 5.0
KP_X, KD_X = 0.5, 0.5

# --- rendering ---
PX_PER_METER = 140
CART_W, CART_H = 60, 30
TRACK_Y = HEIGHT - 100
POLE_VISUAL_LEN = 2 * POLE_HALF_LENGTH * PX_PER_METER

FASTFORWARD_STEPS = 15

BG = (18, 18, 24)
FG = (230, 230, 235)
ACCENT = (90, 200, 255)
GOOD = (110, 220, 140)
BAD = (230, 90, 90)
GRID = (40, 40, 50)


# ---------------------------------------------------------------- physics
def physics_step(x, x_dot, theta, theta_dot, force):
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    temp = (force + POLEMASS_LENGTH * theta_dot ** 2 * sintheta) / TOTAL_MASS
    thetaacc = (GRAVITY * sintheta - costheta * temp) / (
        POLE_HALF_LENGTH * (4.0 / 3.0 - MASSPOLE * costheta ** 2 / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    x = x + TAU * x_dot
    x_dot = x_dot + TAU * xacc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * thetaacc
    return x, x_dot, theta, theta_dot


def pd_controller_force(x, x_dot, theta, theta_dot):
    f = KP_THETA * theta + KD_THETA * theta_dot + KP_X * x + KD_X * x_dot
    return max(-MAX_FORCE, min(MAX_FORCE, f))


# ---------------------------------------------------------------- game state
def random_start():
    return (
        random.uniform(-0.3, 0.3),   # x
        random.uniform(-0.2, 0.2),   # x_dot
        random.uniform(-0.15, 0.15), # theta
        random.uniform(-0.2, 0.2),   # theta_dot
    )


class GameState:
    def __init__(self):
        self.x, self.x_dot, self.theta, self.theta_dot = random_start()
        self.steps_alive = 0
        self.best_steps = 0
        self.episodes = 0
        self.loss_history = deque(maxlen=150)
        self.last_cost = 0.0

    def fail(self):
        return abs(self.theta) > ANGLE_THRESHOLD or abs(self.x) > X_THRESHOLD

    def reset_episode(self):
        self.best_steps = max(self.best_steps, self.steps_alive)
        self.episodes += 1
        self.steps_alive = 0
        self.x, self.x_dot, self.theta, self.theta_dot = random_start()


def build_network():
    # inputs: cart_x, cart_x_dot, pole_theta, pole_theta_dot (all normalized)
    # output: normalized force in [0,1], mapped back to [-MAX_FORCE, MAX_FORCE]
    return Network([4, 10, 10, 1], sigmoid, rate=0.8)


def normalize_state(x, x_dot, theta, theta_dot):
    x_n = x / X_THRESHOLD
    x_dot_n = max(-1.5, min(1.5, x_dot / 3.0))
    theta_n = theta / ANGLE_THRESHOLD
    theta_dot_n = max(-1.5, min(1.5, theta_dot / 3.0))
    return np.array([[x_n], [x_dot_n], [theta_n], [theta_dot_n]])


def get_state_vector(game):
    return normalize_state(game.x, game.x_dot, game.theta, game.theta_dot)


def sample_training_state():
    # sampled broadly across the whole state space (see design note above) so
    # the network keeps practicing on hard/off-center states, not just
    # whatever narrow region the on-screen pole currently sits in
    x = random.uniform(-X_THRESHOLD, X_THRESHOLD)
    x_dot = random.uniform(-3.0, 3.0)
    theta = random.uniform(-ANGLE_THRESHOLD, ANGLE_THRESHOLD)
    theta_dot = random.uniform(-3.0, 3.0)
    return x, x_dot, theta, theta_dot


def training_sample():
    x, x_dot, theta, theta_dot = sample_training_state()
    xv = normalize_state(x, x_dot, theta, theta_dot)
    teacher_force = pd_controller_force(x, x_dot, theta, theta_dot)
    y = np.array([[force_to_target(teacher_force)]])
    return xv, y


def force_to_target(force):
    # map force in [-MAX_FORCE, MAX_FORCE] to a sigmoid-friendly [0,1] target
    return (force / MAX_FORCE + 1.0) / 2.0


def target_to_force(y):
    return (y * 2.0 - 1.0) * MAX_FORCE


def extract_cost_scalar(cost):
    return float(np.array(cost).flatten()[0])


# ---------------------------------------------------------------- main loop
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Cart-Pole — NN Edition")
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
        if fastforward and training:
            for _ in range(FASTFORWARD_STEPS):
                # train on a broadly-sampled state (see design note above)
                xv_train, y_train = training_sample()
                cost = net.train(xv_train, y_train)
                game.last_cost = extract_cost_scalar(cost)
                game.loss_history.append(game.last_cost)

                # separately, step the visible rollout using whichever
                # controller is currently steering
                xv = get_state_vector(game)
                teacher_force = pd_controller_force(game.x, game.x_dot, game.theta, game.theta_dot)
                applied_force = teacher_force if not nn_control else target_to_force(
                    float(net.forward_pass(xv)[0, 0])
                )
                game.x, game.x_dot, game.theta, game.theta_dot = physics_step(
                    game.x, game.x_dot, game.theta, game.theta_dot, applied_force
                )
                game.steps_alive += 1
                if game.fail() or game.steps_alive >= EPISODE_LIMIT:
                    game.reset_episode()

        # ------------------------------------------------------------ step
        if training:
            # train on a broadly-sampled state, independent of where the
            # visible pole currently is (see design note above)
            xv_train, y_train = training_sample()
            cost = net.train(xv_train, y_train)
            game.last_cost = extract_cost_scalar(cost)
            game.loss_history.append(game.last_cost)

        # separately, step the visible rollout with whichever controller
        # is currently steering, so you can watch it play live
        xv = get_state_vector(game)
        predicted_force = target_to_force(float(net.forward_pass(xv)[0, 0]))
        teacher_force = pd_controller_force(game.x, game.x_dot, game.theta, game.theta_dot)

        applied_force = predicted_force if nn_control else teacher_force
        game.x, game.x_dot, game.theta, game.theta_dot = physics_step(
            game.x, game.x_dot, game.theta, game.theta_dot, applied_force
        )
        game.steps_alive += 1
        if game.fail() or game.steps_alive >= EPISODE_LIMIT:
            game.reset_episode()

        # ------------------------------------------------------------ draw
        screen.fill(BG)

        cart_px = WIDTH / 2 + game.x * PX_PER_METER
        track_half_px = X_THRESHOLD * PX_PER_METER
        pygame.draw.line(
            screen, GRID,
            (WIDTH / 2 - track_half_px, TRACK_Y + CART_H / 2 + 4),
            (WIDTH / 2 + track_half_px, TRACK_Y + CART_H / 2 + 4), 2
        )

        # cart
        cart_rect = pygame.Rect(0, 0, CART_W, CART_H)
        cart_rect.center = (cart_px, TRACK_Y)
        cart_color = BAD if game.fail() else FG
        pygame.draw.rect(screen, cart_color, cart_rect, border_radius=4)

        # pole (theta=0 is straight up)
        pivot = (cart_px, TRACK_Y - CART_H / 2)
        tip_x = pivot[0] + POLE_VISUAL_LEN * math.sin(game.theta)
        tip_y = pivot[1] - POLE_VISUAL_LEN * math.cos(game.theta)
        pygame.draw.line(screen, ACCENT, pivot, (tip_x, tip_y), 6)
        pygame.draw.circle(screen, ACCENT, (int(tip_x), int(tip_y)), 7)
        pygame.draw.circle(screen, FG, (int(pivot[0]), int(pivot[1])), 4)

        # loss sparkline
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
        lines = [
            f"Control: {'NN' if nn_control else 'PD CONTROLLER'}   (H to toggle)",
            f"Training: {'ON' if training else 'OFF'}   (T to toggle)   {'[FAST]' if fastforward else ''}",
            f"Balanced steps: {game.steps_alive}   Best: {game.best_steps}   Episodes: {game.episodes}",
            f"Loss: {game.last_cost:.5f}   theta: {math.degrees(game.theta):.1f} deg",
            "R: reset network+game    SPACE: hold to fast-forward training",
        ]
        for i, line in enumerate(lines):
            surf = font.render(line, True, FG)
            screen.blit(surf, (10, 8 + i * 20))

        title = big_font.render("Cart-Pole", True, ACCENT)
        screen.blit(title, (WIDTH - title.get_width() - 10, 8))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()