"""
All the project's constants: the game rules, the network's input, and the training settings.

The game uses whole numbers only (centimeters and "ticks" of 50 ms). That way the Python version
(used for training) and the JavaScript version (used in the browser) produce exactly the same game,
tick by tick, from the same seed. app/static/game.js has the same constants - keep them in sync.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
RECORDINGS_DIR = ROOT / "data" / "recordings"

BC_PATH = MODELS_DIR / "imitation.pt"        # behavioral cloning: learned from human runs
DQN_PATH = MODELS_DIR / "dqn.pt"             # reinforcement learning from scratch
COMBO_PATH = MODELS_DIR / "combo.pt"         # imitation first, then reinforcement learning

# ---------------------------------------------------------------- time and speed
TICK_MS = 50                 # one game step = 50 ms (20 steps per second)
V0 = 50                      # starting speed: 50 cm per tick = 10 m/s
VMAX = 110                   # top speed: 110 cm per tick = 22 m/s
ACCEL_EVERY = 40             # speed goes up by 1 cm/tick every 40 ticks (top speed after 2 minutes)

# ---------------------------------------------------------------- the runner
N_LANES = 3
LANE_TICKS = 4               # changing lanes takes 4 ticks (0.2 s)
JUMP_TICKS = 13              # a jump lasts 13 ticks (0.65 s)
JUMP_HEIGHT = 120            # peak height of a jump (cm)
SLIDE_TICKS = 12             # a slide lasts 12 ticks (0.6 s)
HALF_LEN = 35                # the runner's half-length along the track (cm), for collisions

# ---------------------------------------------------------------- obstacles
LOW, HIGH, TRAIN = 1, 2, 3   # LOW = hurdle (jump over it), HIGH = overhead bar (slide under it), TRAIN = change lanes
OBSTACLE_NAMES = {LOW: "hurdle", HIGH: "overhead bar", TRAIN: "train"}
HURDLE_CLEAR = 50            # a jump clears a hurdle when the runner is higher than 50 cm
BAR_LEN = 50                 # hurdles and overhead bars are 50 cm long
FIRST_ROW = 3000             # the first obstacles appear 30 m from the start
GEN_AHEAD = 7000             # obstacles are generated up to 70 m ahead of the runner
MAX_OBSTACLES = 32           # obstacle slots per game

# ---------------------------------------------------------------- actions
N_ACTIONS = 5
ACTION_NAMES = ["Run", "Left", "Right", "Jump", "Slide"]
NOOP, LEFT, RIGHT, JUMP, SLIDE = range(5)

# ---------------------------------------------------------------- what the network sees
# A "picture" of the track ahead: 3 channels (hurdles, bars, trains) x 32 distance bins x 3 lanes,
# plus 9 numbers about the runner (lane, lane change, jump, slide, speed).
N_BINS = 32
BIN_CM = 150                 # each bin is 1.5 m, so the network sees 48 m ahead
GRID_SHAPE = (3, N_BINS, N_LANES)
N_FEATURES = 9
OBS_SIZE = 3 * N_BINS * N_LANES + N_FEATURES

# ---------------------------------------------------------------- rewards (reinforcement learning)
R_ALIVE = 0.1                # every tick the runner survives
R_CRASH = -2.0               # crashing ends the run
R_ACTION = -0.01             # a tiny cost for every key press, so the runner doesn't fidget

# ---------------------------------------------------------------- limits
TRAIN_EPISODE_TICKS = 3000   # training runs are cut after 2.5 minutes (with a random starting speed)
EVAL_TICKS = 6000            # evaluation runs are cut after 5 minutes
GHOST_TICKS = 12000          # in the app the AI runs for up to 10 minutes
VAL_SEEDS = list(range(1000, 1016))   # 16 fixed tracks for validation during training
TEST_SEEDS = list(range(5000, 5050))  # 50 fixed tracks for the final evaluation

# ---------------------------------------------------------------- training settings
DQN = dict(
    n_envs=32,                # 32 runners play in parallel
    total_steps=60_000,       # steps of all runners together = 1.9 million experiences
    buffer_size=300_000,
    batch_size=256,
    lr=2.5e-4,
    gamma=0.99,               # looks ahead about 100 ticks (5 seconds)
    tau=0.005,                # soft update of the target network
    learning_starts=2_000,
    train_every=1,
    eps_start=1.0,
    eps_end=0.02,
    eps_fraction=0.4,         # epsilon goes down during the first 40% of training
    eval_every=2_500,
    seed=0,
)
COMBO = dict(                 # changes for "imitation first, then reinforcement learning" (DQfD)
    pretrain_updates=4_000,   # first: learn only from the human runs
    total_steps=30_000,       # then: half the reinforcement-learning steps of plain DQN
    eps_start=0.2,            # the network already plays reasonably, so less random exploration
    demo_fraction=0.25,       # a quarter of every batch comes from the human runs
    margin=0.8,               # the human's action must be worth at least 0.8 more than any other
    margin_weight=1.0,
)
BC = dict(
    epochs=20,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    val_fraction=0.15,        # recordings set aside for validation (whole runs, not single ticks)
    seed=0,
)
