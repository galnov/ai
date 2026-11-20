"""
All the project's constants in one place: world, car, sensors, rewards and training.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TRACKS_DIR = ROOT / "tracks"
MODEL_PATH = MODELS_DIR / "driver.pt"

# ---------------------------------------------------------------- world
WORLD_W, WORLD_H = 1200, 800       # world size in units (= pixels when drawn)
TRACK_SPACING = 8.0                # distance between points on the track's center line
TRACK_WIDTH_RANGE = (80.0, 115.0)  # width of random training tracks

# ---------------------------------------------------------------- car
DT = 0.05                          # time step in seconds (20 decisions per second)
MAX_SPEED = 260.0                  # units per second
ACCEL = 220.0                      # acceleration when pressing the gas
BRAKE = 480.0                      # deceleration when braking
COAST_DRAG = 40.0                  # deceleration when pressing nothing
FRICTION = 0.15                    # friction proportional to speed
TURN_RATE = 3.2                    # maximum turning speed (radians per second)
TURN_SPEED_FULL = 60.0             # below this speed the steering is less effective
HIGH_SPEED_TURN_LOSS = 0.30        # at top speed steering is 30% weaker - you must brake before corners
CAR_RADIUS = 7.0                   # for collisions
CAR_LENGTH, CAR_WIDTH = 22.0, 11.0  # for drawing

# actions: (steering, throttle)  steering: -1 left, 0 straight, 1 right | throttle: 1 gas, 0 nothing, -1 brake
ACTIONS = [
    (0, 1),    # gas straight
    (-1, 1),   # gas + left
    (1, 1),    # gas + right
    (-1, 0),   # left without gas
    (1, 0),    # right without gas
    (0, -1),   # brake
]
ACTION_NAMES = ["Gas", "Gas + left", "Gas + right", "Left", "Right", "Brake"]
N_ACTIONS = len(ACTIONS)

# ---------------------------------------------------------------- sensors
RAY_ANGLES = np.radians([-90, -60, -35, -15, 0, 15, 35, 60, 90])
RAY_LENGTH = 260.0
N_RAYS = len(RAY_ANGLES)
OBS_SIZE = N_RAYS + 1              # 9 distance sensors + speed

# ---------------------------------------------------------------- rewards
REWARD_CRASH = -5.0
REWARD_STUCK = -2.0
REWARD_TIME = -0.02                # a small "penalty" every step - it pays to hurry
STUCK_STEPS = 60                   # 3 seconds without progress = stuck
MAX_EPISODE_STEPS = 1500           # 75 seconds

# ---------------------------------------------------------------- training (DQN)
TRAIN = dict(
    n_envs=16,                 # 16 cars learn in parallel (sharing one "brain")
    total_steps=40_000,        # steps of all cars together = 640,000 transitions
    buffer_size=200_000,       # experience memory (replay buffer)
    batch_size=256,
    lr=3e-4,
    gamma=0.99,                # how much the future matters (0.99 = about 5 seconds ahead)
    eps_start=1.0,
    eps_end=0.03,
    eps_decay_fraction=0.4,    # epsilon decreases over the first 40% of training
    learning_starts=1_000,     # random steps before learning starts
    tau=0.005,                 # soft update of the target network
    grad_clip=10.0,
    new_track_every=500,       # a new random track every 500 steps
    hidden=128,
    eval_every=2_500,          # test on validation tracks and keep the best network
    eval_tracks=10,
    eval_steps=600,            # 30 seconds of driving on each validation track
    snapshot_steps=(1_000, 1_150, 1_300, 1_500),  # "snapshots" of the brain early in learning, for comparison in the app
    seed=0,
)

VALIDATION_SEED = 2024         # validation tracks - fixed
TEST_SEED = 777                # final test tracks - different from anything the agent has seen
