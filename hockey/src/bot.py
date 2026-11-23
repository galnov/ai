"""
The hand-written opponent: simple rules, no learning. Used two ways:
  * as the first opponent in training (a network that has just started can't beat anything else)
  * as a baseline: "does the trained network beat a decent rule-based player?"

It sees the table exactly like the network does (its own view, the same reaction time) and controls its paddle
the same way (a velocity in [-1, 1]), so the comparison is fair. The rules:
  * DEFEND when the puck is in the other half, or coming at us fast: stand between the puck and our goal,
    a little in front of the goal.
  * ATTACK when the puck is in our half and slow or moving away: first get behind the puck (on the line from
    the opponent's goal through the puck), then drive through it toward the opponent's goal.
"""
import numpy as np

from . import config as C

GAIN = 6.0                 # how hard it steers toward the point it wants to be at
SPEED = 0.7                # its top speed as a share of the paddle limit (faster makes it overshoot and lose)
DRIVE = 0.7                # the speed of a shot (a full-speed shot is less accurate: it often comes back)


def bot_actions(obs, skill=1.0, drive=DRIVE):
    """obs: (n, OBS_SIZE) in the player's own view (own goal at y = -1). Returns (n, 2) in [-1, 1]."""
    o = np.asarray(obs, float)
    me = o[:, 0:2] * [C.HALF_W, C.HALF_L]
    puck = o[:, 8:10] * [C.HALF_W, C.HALF_L]
    pv = o[:, 10:12] * C.PUCK_VMAX
    n = len(o)

    # defend: a point on the line from the middle of our goal to the puck, 0.18 m in front of the goal line
    goal = np.array([0.0, -C.HALF_L])
    to_puck = puck - goal
    t = np.clip(0.18 / np.maximum(to_puck[:, 1], 0.05), 0, 1)
    guard = goal + to_puck * t[:, None]
    guard[:, 1] = np.maximum(guard[:, 1], -C.HALF_L + C.PADDLE_R + 0.02)

    # attack: get behind the puck (seen from the opponent's goal), then push through it
    target = np.array([0.0, C.HALF_L])
    aim = target - puck
    aim /= np.linalg.norm(aim, axis=-1, keepdims=True) + 1e-9
    reach = C.PUCK_R + C.PADDLE_R + 0.03
    # a puck right at our end wall: there is no room behind it, so knock it out sideways (toward the middle)
    deep = puck[:, 1] - aim[:, 1] * reach < -C.HALF_L + C.PADDLE_R + 0.01
    side = np.stack([-np.sign(puck[:, 0] + 1e-9), np.full(n, 0.35)], 1)
    side /= np.linalg.norm(side, axis=-1, keepdims=True)
    aim = np.where(deep[:, None], side, aim)
    behind = puck - aim * reach
    lined_up = ((me - behind) ** 2).sum(-1) < 0.04 ** 2
    through = puck + aim * 0.25                                   # drive through the puck
    attack_pt = np.where(lined_up[:, None], through, behind)
    # never pass the puck on the way back (that pushes it into our own goal): if we are in front of it,
    # first step aside, then go down beside it
    in_front = me[:, 1] > puck[:, 1] - 0.02
    beside = np.abs(me[:, 0] - puck[:, 0]) > C.PUCK_R + C.PADDLE_R + 0.03
    away = np.sign(me[:, 0] - puck[:, 0] + 1e-9)
    if_close = np.stack([np.clip(puck[:, 0] + away * 0.16, -C.HALF_W + C.PADDLE_R, C.HALF_W - C.PADDLE_R), me[:, 1]], 1)
    if_beside = np.stack([me[:, 0], puck[:, 1] - 0.12], 1)
    around = np.where(beside[:, None], if_beside, if_close)
    attack_pt = np.where(in_front[:, None], around, attack_pt)

    ours = puck[:, 1] < 0
    coming = pv[:, 1] < -1.0
    attack = ours & ~coming
    point = np.where(attack[:, None], attack_pt, guard)
    act = np.clip((point - me) * GAIN, -1, 1)
    shooting = attack & lined_up & ~in_front
    act = np.where(shooting[:, None], act * drive, act) * SPEED
    if skill < 1.0:                                               # a weaker bot: slower
        act *= skill
    return act
