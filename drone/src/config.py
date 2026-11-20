"""
All the constants: the world, the drone's physics, the reward and PPO training.
Units: meters, seconds, kilograms, newtons.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "drone_ppo.pt"

# ---------------------------------------------------------------- world
WORLD_W, WORLD_H = 20.0, 12.0     # width and height of the area (the ground is at y=0)
G = 9.81

# ---------------------------------------------------------------- drone (side view: two propellers)
MASS = 1.0                        # the drone's mass without payload
PAYLOAD_RANGE = (0.0, 0.5)        # a delivery package
ARM = 0.25                        # distance of each propeller from the center
INERTIA = 0.02                    # moment of inertia
T_HOVER = MASS * G / 2            # thrust per propeller needed to hover (without payload)
T_MAX = 2.0 * T_HOVER             # maximum thrust per propeller (thrust-to-weight ratio 2)
DRAG = 0.35                       # air drag (relative to the speed through the air)
ANG_DRAG = 0.02
MOTOR_TAU = 0.05                  # the motors respond with a small delay (not instantly)

PHYS_DT = 0.01                    # physics step
CONTROL_DT = 0.04                 # the network decides 25 times per second
SUBSTEPS = int(round(CONTROL_DT / PHYS_DT))

# ---------------------------------------------------------------- wind
WIND_MAX = 4.0                    # m/s - maximum constant wind in training
GUST_STD = 1.5                    # gust strength
GUST_TAU = 1.5                    # how long a gust lasts (seconds)

# ---------------------------------------------------------------- reinforcement learning
EPISODE_SECONDS = 10.0
OBS_SIZE = 11                     # see drone.observe
ACT_SIZE = 2                      # left propeller thrust, right propeller thrust
CRASH_PENALTY = -10.0
MAX_TILT = 1.4                    # radians (80 degrees) - more than this = flipping over

PPO = dict(
    n_envs=128,                   # 128 drones learn in parallel
    n_steps=64,                   # steps per drone in each collection round = 8,192 samples
    total_steps=3_000_000,
    epochs=10,
    minibatch=2048,
    gamma=0.99,
    gae_lambda=0.95,
    clip=0.2,
    lr=3e-4,
    vf_coef=0.5,
    max_grad_norm=0.5,
    hidden=64,
    log_std_init=-0.5,
    eval_every=40,                # every 40 updates - a test on validation scenarios
    seed=0,
)
