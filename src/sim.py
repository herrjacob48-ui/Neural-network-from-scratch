import pygame
import numpy as np
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Inverted Pendulum PID Balancing Simulation")
clock = pygame.time.Clock()

# --- Physical Constants & Parameters ---
M = 2.0      # Mass of the cart (kg)
m = 0.5      # Mass of the pendulum bob (kg)
L = 1.5      # Length of the pendulum rod (m)
g = 9.81     # Gravity acceleration (m/s^2)
b = 0.1      # Friction/damping coefficient for the cart

# --- State Variables ---
# x: cart position (m), x_dot: cart velocity (m/s)
# theta: angle from UPWARD vertical (rad), theta_dot: angular velocity (rad/s)
x = 0.0
x_dot = 0.0
theta = 0.5  # Small initial offset so you can watch the PID catch and balance it!
theta_dot = 0.0

# --- PID Controller Constants ---
# Tune these gains to adjust how aggressively the cart balances
Kp_theta = -150.0  # Proportional gain for angle error
Kd_theta = -30.0   # Derivative gain for angular velocity error
Ki_theta = -5.0    # Integral gain for steady-state error

Kp_x = 2.5         # Proportional gain to keep cart near center
Kd_x = 4.0         # Derivative gain to damp cart horizontal drift

# PID State Variables
integral_theta_error = 0.0

# --- Dynamic Dynamic Setpoints (Targets) ---
target_theta = 0.0      # Ideal target is 0.0 (perfectly straight up)
target_theta_dot = 0.0  # Ideal target angular velocity is 0.0 (stationary)

# --- Visual Scaling & Loop Configuration ---
PIXELS_PER_METER = 150
ORIGIN_X = WIDTH // 2
TRACK_Y = HEIGHT // 2 + 100  # Shift track down slightly

running = True
dt = 1 / 100.0  # 100 Hz frequency

while running:
    # 1. Event Handling & Interactive Target Adjustment
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    # 2. PID Controller Calculation
    # Normalize theta error to handle wrapping if needed
    theta_error = theta - target_theta
    theta_dot_error = theta_dot - target_theta_dot
    
    # Accumulate integral error (with anti-windup clamping)
    integral_theta_error += theta_error * dt
    integral_theta_error = np.clip(integral_theta_error, -2.0, 2.0)

    # Compute required force to balance the pendulum at the given angle/velocity
    # We add a secondary outer control loop (Kp_x, Kd_x) to prevent the cart from drifting off screen
    force = (Kp_theta * theta_error + 
             Kd_theta * theta_dot_error + 
             Ki_theta * integral_theta_error +
             Kp_x * (x - 0.0) + 
             Kd_x * x_dot)

    # Clip maximum motor force to realistic limits (Newtons)
    force = np.clip(force, -100.0, 100.0)

    # 3. Physics Engine (Inverted Pendulum Equations of Motion)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    denom = M + m * sin_t**2

    # Note the inverted signs on the gravity 'g' terms compared to the downward layout
    x_ddot = (force - b * x_dot + m * L * theta_dot**2 * sin_t - m * g * sin_t * cos_t) / denom
    theta_ddot = (force * cos_t - b * x_dot * cos_t + m * L * theta_dot**2 * sin_t * cos_t + (M + m) * g * sin_t) / (L * denom)

    # Numerical Integration (Euler-Cromer)
    x_dot += x_ddot * dt
    x += x_dot * dt
    
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt

    # 4. Graphics Rendering
    screen.fill((255, 255, 255))

    # Track Line
    pygame.draw.line(screen, (200, 200, 200), (0, TRACK_Y), (WIDTH, TRACK_Y), 2)

    # Coordinate mapping: theta = 0 points straight UP now
    cart_pixel_x = int(ORIGIN_X + x * PIXELS_PER_METER)
    cart_pixel_y = TRACK_Y

    bob_pixel_x = int(cart_pixel_x + L * sin_t * PIXELS_PER_METER)
    bob_pixel_y = int(cart_pixel_y - L * cos_t * PIXELS_PER_METER) # Negative sign flips it upward

    # Draw Pendulum
    pygame.draw.line(screen, (0, 0, 0), (cart_pixel_x, cart_pixel_y), (bob_pixel_x, bob_pixel_y), 4)

    # Draw Cart
    cart_w, cart_h = 80, 40
    cart_rect = pygame.Rect(cart_pixel_x - cart_w // 2, cart_pixel_y - cart_h // 2, cart_w, cart_h)
    pygame.draw.rect(screen, (50, 100, 255), cart_rect)
    pygame.draw.rect(screen, (0, 0, 0), cart_rect, 2)

    # Draw Pendulum Bob
    pygame.draw.circle(screen, (255, 50, 50), (bob_pixel_x, bob_pixel_y), 15)
    pygame.draw.circle(screen, (0, 0, 0), (bob_pixel_x, bob_pixel_y), 15, 2)

    # Text UI for monitoring current states vs target states
    font = pygame.font.SysFont(None, 24)
    screen.blit(font.render(f"Required PID Force: {force:.2f} N", True, (0, 0, 0)), (20, 20))

  

    pygame.display.flip()
    clock.tick(100) # 100 Hz capping

pygame.quit()
sys.exit()

    







pygame.quit()
sys.exit()