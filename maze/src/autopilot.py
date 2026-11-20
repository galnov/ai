"""
A hand-written autopilot, for comparison with the network (like the PID controller in the drone project).

The rules:
  * Steer the nose toward a point a little past the next beacon (proportional control on the angle).
  * Slow down when the next segment turns, and when the nose is not yet pointing the right way.
It uses the beacon, but not the depth camera - so it cannot notice that it is
drifting into a wall. The network can.
"""
import numpy as np

from . import config as C

GAIN = 2.5            # how hard to steer for a given angle error
LOOK = 0.25           # aim between the next beacon (0) and the one after it (1)


def autopilot(obs):
    """obs (n, 93) -> actions (n, 3). Uses only the beacon part of the observation."""
    o = np.asarray(obs, float)
    k = C.N_RAYS
    b1 = o[:, k:k + 3] * C.BEACON_SCALE                  # next cell, in the ship's frame (right, up, forward)
    b2 = o[:, k + 3:k + 6] * C.BEACON_SCALE
    aim = b1 + LOOK * (b2 - b1)
    yaw = np.arctan2(aim[:, 0], aim[:, 2])
    pitch = np.arctan2(aim[:, 1], np.hypot(aim[:, 0], aim[:, 2]))
    off = np.arctan2(np.hypot(aim[:, 0], aim[:, 1]), aim[:, 2])           # how far the aim is from the nose

    act = np.zeros((len(o), 3))
    act[:, 0] = np.clip(GAIN * pitch, -1, 1)
    act[:, 1] = np.clip(GAIN * yaw, -1, 1)

    seg = b2 - b1                                         # the direction of the next segment
    seg_len = np.linalg.norm(seg, axis=1)
    fwd = np.where(seg_len > 1e-6, seg[:, 2] / np.maximum(seg_len, 1e-6), 1.0)
    turn_ahead = np.clip(1 - fwd, 0, 1)                    # 0 = straight on, 1 = a 90-degree turn
    err = np.clip(off / 0.8, 0, 1)                         # how far the nose is from the aim
    close = np.clip(1 - np.linalg.norm(b1, axis=1) / (2 * C.CELL), 0, 1)
    slow = np.maximum(turn_ahead * (0.5 + 0.5 * close), err)
    speed = C.V_MAX - (C.V_MAX - C.V_MIN) * slow
    act[:, 2] = np.clip(2 * (speed - C.V_MIN) / (C.V_MAX - C.V_MIN) - 1, -1, 1)
    return act
