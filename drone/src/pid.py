"""
A PID controller - the classic engineering solution for drones, for comparison with the network.

A "cascaded" controller, as in real flight controllers:
  1. Position loop:  what acceleration is needed?   a = Kp·error + Ki·∫error − Kd·velocity
  2. From the desired acceleration, compute the total thrust and the desired tilt angle
  3. Angle loop:  what torque is needed to reach the angle?  τ = I·(Kp·(θ* − θ) − Kd·ω)
  4. From the thrust and the torque, compute the thrust for each propeller
The PID doesn't know the payload's weight or the wind. The integral corrects for them over time.
It has "anti-windup": the integral is limited, so it doesn't accumulate endlessly.
The gains were tuned by hand, for normal conditions.
"""
import numpy as np

from . import config as C

KP, KI, KD = 2.2, 1.0, 2.6          # position loop
KP_ANG, KD_ANG = 160.0, 22.0        # angle loop
MAX_ACC_X, MAX_ACC_Y, MAX_ACC_DOWN = 6.0, 6.0, 4.0
MAX_TILT = 0.6
I_LIMIT = 5.0                       # the integral's contribution is limited to 5 m/s² (anti-windup)


class PID:
    def __init__(self, n):
        self.ix = np.zeros(n)
        self.iy = np.zeros(n)
        self.last_target = np.full((n, 2), np.nan)

    def __call__(self, obs, env):
        tgt = np.stack([env.tx, env.ty], axis=1)
        changed = np.any(tgt != self.last_target, axis=1)        # new target - reset the integral
        self.ix[changed] = 0
        self.iy[changed] = 0
        self.last_target = tgt.copy()
        ex, ey = env.tx - env.x, env.ty - env.y
        dt = C.CONTROL_DT
        self.ix = np.clip(self.ix + ex * dt, -I_LIMIT / KI, I_LIMIT / KI)
        self.iy = np.clip(self.iy + ey * dt, -I_LIMIT / KI, I_LIMIT / KI)
        ax = np.clip(KP * ex + KI * self.ix - KD * env.vx, -MAX_ACC_X, MAX_ACC_X)
        ay = np.clip(KP * ey + KI * self.iy - KD * env.vy, -MAX_ACC_DOWN, MAX_ACC_Y)
        fx, fy = C.MASS * ax, C.MASS * (ay + C.G)          # (nominal mass - without payload)
        th_des = np.clip(np.arctan2(-fx, np.maximum(fy, 0.5)), -MAX_TILT, MAX_TILT)
        F = np.maximum(0, -fx * np.sin(env.th) + fy * np.cos(env.th))
        tau = C.INERTIA * (KP_ANG * (th_des - env.th) - KD_ANG * env.om)
        t1 = F / 2 - tau / (2 * C.ARM)
        t2 = F / 2 + tau / (2 * C.ARM)
        return np.clip(np.stack([t1, t2], axis=1) / C.T_HOVER - 1, -1, 1)


def pid_policy(n):
    """Returns a policy function policy(obs, env) with its own integral memory."""
    return PID(n)
