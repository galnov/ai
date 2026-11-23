"""
Pixel Breakout AI - all the settings in one place.

Units: pixels of the network's screen (48 x 48) and seconds. x goes right, y goes DOWN (like an image).

    y = 0  ┌────────────────────────────────────────────────┐
           │                                                │
    y = 6  │  ▄▄▄▄▄ ▄▄▄▄▄ ▄▄▄▄▄ ▄▄▄▄▄ ▄▄▄▄▄ ▄▄▄▄▄ ▄▄▄▄▄ ▄▄▄▄▄│  5 rows x 8 bricks
           │  ...                                           │
           │                      •  ball                   │
    y = 44 │                 ▀▀▀▀▀▀▀▀  paddle               │
    y = 48 └────────────────────────────────────────────────┘  below the paddle: the ball is lost

The same numbers are in docs/game.js (the JavaScript copy for the web page) - keep them in sync.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "breakout_ppo.pt"
SNAPSHOT_DIR = MODELS_DIR / "snapshots"

# ---------------------------------------------------------------- the screen and the bricks
SCREEN = 48                # the network sees 48 x 48 pixels
BRICK_ROWS = 5
BRICK_COLS = 8
BRICK_W = 6                # a brick's cell (it is drawn 1 pixel narrower, so there is a gap)
BRICK_H = 3                # (drawn 2 pixels tall)
BRICK_TOP = 6              # the first row starts at y = 6
PADDLE_W = 8
PADDLE_Y = 44              # the paddle's top edge (it is 2 pixels tall)

# ---------------------------------------------------------------- motion
DT = 1 / 15                # one decision every 1/15 s
SUBSTEPS = 4               # physics steps per decision (a ball can't jump over a 2-pixel brick)
PADDLE_SPEED = 66.0        # pixels per second - the same for you and the AI
BALL_SPEED = 34.0          # pixels per second at the start of a ball
BALL_SPEED_MAX = 50.0
SPEEDUP = 1.02             # the ball gets 2% faster with every brick
MAX_ANGLE = 60             # degrees from vertical when the ball hits the edge of the paddle
MIN_VY = 0.3               # the ball always moves at least this much vertically (as a share of its speed)

# ---------------------------------------------------------------- a game
LIVES = 3
SERVE_ANGLE = (15, 45)     # a new ball leaves the paddle at a random angle in this range (degrees, left or right)
MAX_STEPS = 3000           # a game ends after 200 s (3000 decisions) no matter what

# ---------------------------------------------------------------- the network's input and output
FRAMES = 2                 # it sees the last 2 screens: one screen shows where things are, two show where they move
ACTIONS = ["stay", "left", "right"]

# ---------------------------------------------------------------- rewards
BRICK_REWARD = 1.0
MISS_REWARD = -1.0

# ---------------------------------------------------------------- PPO
PPO = dict(
    n_envs=64, n_steps=128, total_steps=10_000_000,
    lr=2.5e-4, gamma=0.99, lam=0.95, clip=0.1, epochs=4, minibatch=1024,
    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
)
SNAPSHOT_STEPS = (200_000, 1_000_000, 3_000_000)   # the "learning stages" in the web page
EVAL_EVERY = 25            # updates
EVAL_GAMES = 32
SEED = 0
