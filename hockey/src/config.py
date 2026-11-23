"""
Air Hockey AI - all the settings in one place.

Units: meters, seconds. The table is seen from above:
    x: from -0.5 (left) to +0.5 (right)                       TABLE_W = 1.0
    y: from -1.0 (the bottom player's goal) to +1.0 (the top player's goal)   TABLE_L = 2.0
The bottom player defends y = -1 and attacks y = +1. Each player stays in its own half.

The same numbers are in docs/physics.js (the JavaScript copy for the web page) - keep them in sync.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "hockey_ppo.pt"          # the best version (against the bot, in the second half of training)
LAST_PATH = MODELS_DIR / "hockey_last.pt"          # the version at the end of training
SNAPSHOT_DIR = MODELS_DIR / "snapshots"

# ---------------------------------------------------------------- the table
HALF_W = 0.5               # half the table's width
HALF_L = 1.0               # half the table's length
GOAL_HALF = 0.15           # the goal mouth: |x| < 0.15 at y = +-1
PUCK_R = 0.035             # puck radius
PADDLE_R = 0.05            # paddle ("mallet") radius

# ---------------------------------------------------------------- physics
DT = 1 / 30                # one decision every 1/30 s
SUBSTEPS = 8               # physics steps per decision (1/240 s each, so a fast puck can't jump through a paddle)
PADDLE_VMAX = 2.6          # m/s - the same top speed for the AI and the human
PADDLE_ACCEL = 30.0        # m/s^2 - how fast a paddle can change its velocity
PUCK_VMAX = 6.0            # m/s
PUCK_FRICTION = 0.25       # per second: the puck keeps exp(-0.25 t) of its speed (an air table glides)
WALL_E = 0.88              # how bouncy the walls are (1 = no loss)
PADDLE_E = 0.85            # how bouncy a paddle hit is
SERVE_SPEED = 0.25         # the puck starts slowly, toward the player who serves

# ---------------------------------------------------------------- a point
STALL_TIME = 7.0           # the real air hockey "7-second rule": a player who keeps the puck in their own half
                           # for 7 seconds commits a foul, and the opponent gets the puck
POINT_TIME = 30.0          # seconds; a point with no goal and no foul for this long is replayed (rare)
MATCH_GOALS = 7            # in the app: first to 7 wins

# ---------------------------------------------------------------- what a player sees and does
# Everything is seen from the player's own side (the top player's view is turned 180 degrees), so one network
# can play either side - that's what makes self-play possible.
#   0-1  own paddle position (x / 0.5, y / 1)        2-3  own paddle velocity (/ PADDLE_VMAX)
#   4-5  opponent paddle position                     6-7  opponent paddle velocity
#   8-9  puck position                               10-11 puck velocity (/ PUCK_VMAX)
#   12   my 7-second clock (time the puck has been in my half / 7 s, 0 if it is in the other half)
#   13   the opponent's 7-second clock
OBS_SIZE = 14
ACT_SIZE = 2               # the velocity the paddle tries to reach: (x, y) in [-1, 1] x PADDLE_VMAX
REACTION_STEPS = 2         # the AI sees the puck and the opponent as they were 2 decisions ago (67 ms) - a human-like
                           # reaction time. Its own paddle it knows without delay, as you feel where your hand is

# ---------------------------------------------------------------- rewards
GOAL_REWARD = 10.0         # +10 for scoring, -10 for conceding
PUSH_REWARD = 0.6          # shaping: +0.6 per meter the puck moves toward the opponent's goal (and - back)
TOUCH_REWARD = 0.05        # a small reward for hitting the puck (helps at the very beginning)
FOUL_REWARD = -3.0         # for the player who broke the 7-second rule - so waiting with the puck doesn't pay
TIMEOUT_REWARD = 0.0

# ---------------------------------------------------------------- PPO + self-play
PPO = dict(
    n_envs=64, n_steps=256, total_steps=12_000_000,
    hidden=128, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=8, minibatch=1024,
    vf_coef=0.5, max_grad_norm=0.5, log_std_init=-0.4,
)
# The opponents during training: at first mostly the hand-written bot, later mostly earlier versions of the
# network itself (a "pool" of snapshots), so it keeps meeting different styles and doesn't overfit to one.
BOT_SHARE_START = 0.9
BOT_SHARE_END = 0.2
POOL_EVERY = 25            # updates between adding the current network to the opponent pool
POOL_SIZE = 12             # how many earlier versions are kept
LATEST_SHARE = 0.4         # of the network opponents, how many play the most recent version
SNAPSHOT_UPDATES = (5, 40, 150, 400)      # saved for the "learning stages" in the app
EVAL_EVERY = 25            # updates between evaluations against the bot
EVAL_POINTS = 400
SEED = 0
