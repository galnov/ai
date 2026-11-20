"""
All the constants: the maze, the ship's physics, its sensors, the reward and PPO training.
Units: meters, seconds, radians.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "pilot_ppo.pt"

# ---------------------------------------------------------------- the maze
CELL = 4.0                  # every corridor segment is a 4 m cube: a narrow tunnel
MAZE_ROOMS = 4              # 4 x 4 x 4 "rooms" joined by passages -> a grid of 9 x 9 x 9 cells
TRAIN_SIZES = (3, 4)        # in training, mazes of 3x3x3 and 4x4x4 rooms
TRAIN_POOL = 1500           # training mazes, generated once at the start (seeds never used for testing)

# ---------------------------------------------------------------- the ship
SHIP_RADIUS = 0.6           # the ship is a ball of radius 0.6 m for collisions
V_MIN, V_MAX = 2.0, 11.0    # the throttle sets a target speed between 2 and 11 m/s
ACCEL_K = 2.5               # how fast the velocity follows the nose (1/s) - lower = more drift
PITCH_RATE = 2.2            # maximum rate of lifting / lowering the nose (rad/s, about 126 degrees per second)
YAW_RATE = 2.2              # maximum turning rate left/right - both around the ship's own axes, like a spaceship
DT = 0.1                    # the pilot decides 10 times per second
SUBSTEPS = 2                # physics steps per decision (for accurate collisions)

# ---------------------------------------------------------------- the sensors
# A depth camera: 9 x 9 rays in front of the ship, spread over 100 degrees.
# Each ray measures the distance to the nearest wall (up to 24 m).
RAYS_X, RAYS_Y = 9, 9
FOV = 100.0                 # degrees, horizontal and vertical
RAY_RANGE = 24.0
N_RAYS = RAYS_X * RAYS_Y
# The navigation beacon: the ship knows where the next two cells on the way out are (like GPS),
# but not the walls around them - it has to see those with its depth camera.
BEACON_SCALE = 8.0
OBS_SIZE = N_RAYS + 3 + 3 + 3 + 3   # depth image, beacon 1, beacon 2, velocity, previous action = 93
ACT_SIZE = 3                        # pitch, yaw, throttle (each between -1 and 1)
ACTION_NAMES = ["Pitch", "Yaw", "Throttle"]

# ---------------------------------------------------------------- reward
R_PROGRESS = 0.5            # per meter of progress along the way out
R_EXIT = 20.0               # reaching the exit
R_CRASH = -20.0             # touching a wall ends the flight
R_SMOOTH = 0.02             # a small cost for jerky steering (change of action)
TIME_LIMIT = 120.0          # seconds per flight in training

# ---------------------------------------------------------------- PPO
PPO = dict(
    n_envs=64,              # 64 ships fly in parallel, each in its own random maze
    n_steps=128,            # steps per ship in each collection round = 8,192 samples
    total_steps=3_000_000,   # about 366 updates; learning levels off after about 2 million steps
    hidden=128,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip=0.2,
    epochs=10,
    minibatch=2048,
    vf_coef=0.5,
    max_grad_norm=0.5,
    log_std_init=-0.5,
    eval_every=20,          # every 20 updates: a test on the validation mazes
    seed=0,
)
VAL_SEEDS = list(range(10_000, 10_032))     # 32 validation mazes (4x4x4 rooms)
TEST_SEEDS = list(range(20_000, 20_100))    # 100 test mazes, never used in training
SNAPSHOT_UPDATES = (1, 10, 20, 40)          # "brains" saved early in training, for the learning stages
