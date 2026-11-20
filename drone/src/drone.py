"""
The learning environment: 2D drones (side view), computed vectorized with NumPy - many drones in parallel.

The physics (a rigid body in a plane):
    State: position (x, y), velocity (vx, vy), angle θ (0 = level, positive = tilted counter-clockwise), angular velocity ω
    Two propellers at distance ARM on either side of the center, each pushing "up" relative to the drone's body.
        Total force   F = T1 + T2               in the direction (−sinθ, cosθ)
        Torque        τ = (T2 − T1)·ARM          (right one stronger -> the drone tilts left)
        Acceleration  a = F·(−sinθ, cosθ)/m − g − drag·(v − wind)/m
    To move right, the drone must first tilt right (a difference between the propellers), and then correct immediately so it doesn't flip.

Action (2 numbers, each between -1 and 1):  0 = hover thrust, 1 = maximum thrust, -1 = off.
The observation (11 numbers) - see observe().
"""
import numpy as np

from . import config as C


class DroneEnv:
    def __init__(self, n=1, seed=0, randomize=True):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.randomize = randomize          # in training: random wind, payload and targets
        z = lambda: np.zeros(n)
        self.x, self.y, self.vx, self.vy, self.th, self.om = z(), z(), z(), z(), z(), z()
        self.t1, self.t2 = np.full(n, C.T_HOVER), np.full(n, C.T_HOVER)
        self.mass = np.full(n, C.MASS)
        self.wind, self.gust = z(), z()
        self.gust_std = np.full(n, C.GUST_STD)
        self.tx, self.ty = z(), z()
        self.ix, self.iy = z(), z()
        self.prev_a = np.zeros((n, 2))
        self.t, self.retarget_at = z(), z()
        self.d_prev = z()
        self.ep_return = z()
        self.done = np.zeros(n, bool)
        self.reset()

    # ------------------------------------------------------------ reset
    def reset(self, mask=None):
        m = np.ones(self.n, bool) if mask is None else mask
        k = int(m.sum())
        if k == 0:
            return self.observe()
        r = self.rng
        self.x[m] = r.uniform(2, C.WORLD_W - 2, k)
        self.y[m] = r.uniform(2.5, C.WORLD_H - 2, k)
        if self.randomize:
            self.vx[m], self.vy[m] = r.normal(0, 1.0, k), r.normal(0, 1.0, k)
            self.th[m], self.om[m] = r.normal(0, 0.25, k), r.normal(0, 1.0, k)
            self.wind[m] = np.where(r.random(k) < 0.7, r.uniform(-C.WIND_MAX, C.WIND_MAX, k), 0.0)
            self.gust_std[m] = np.where(r.random(k) < 0.5, r.uniform(0, C.GUST_STD, k), 0.0)
            self.mass[m] = C.MASS + np.where(r.random(k) < 0.5, r.uniform(*C.PAYLOAD_RANGE, k), 0.0)
        else:
            self.vx[m] = self.vy[m] = self.th[m] = self.om[m] = 0
        self.gust[m] = 0
        self.t1[m] = self.t2[m] = C.T_HOVER
        self.prev_a[m] = 0
        self.t[m] = 0
        self.ep_return[m] = 0
        self.done[m] = False
        self.new_target(m)
        return self.observe()

    def new_target(self, mask, tx=None, ty=None):
        k = int(mask.sum())
        if k == 0:
            return
        self.tx[mask] = self.rng.uniform(1.5, C.WORLD_W - 1.5, k) if tx is None else tx
        self.ty[mask] = self.rng.uniform(2.5, C.WORLD_H - 1.5, k) if ty is None else ty
        self.ix[mask] = self.iy[mask] = 0
        self.d_prev[mask] = np.hypot(self.tx[mask] - self.x[mask], self.ty[mask] - self.y[mask])
        # in training: sometimes the target changes midway (like points on a delivery route)
        nxt = self.rng.uniform(2.5, 7.0, k)
        never = self.rng.random(k) < 0.3
        self.retarget_at[mask] = np.where(never | (not self.randomize), np.inf, self.t[mask] + nxt)

    def auto_reset(self):
        return self.reset(self.done.copy()) if self.done.any() else self.observe()

    # ------------------------------------------------------------ observation
    def observe(self):
        """
        What the drone "knows" (like a real flight controller - from GPS, gyroscope and accelerometer):
          [Δx, Δy] to the target, [vx, vy], [sinθ, cosθ], ω, [integral of the error x, y], [the previous action x2]
        The integral lets the network "feel" a constant wind or a heavy payload, exactly like the I in a PID controller.
        """
        dx = np.clip(self.tx - self.x, -8, 8) / 4
        dy = np.clip(self.ty - self.y, -8, 8) / 4
        return np.stack([
            dx, dy, self.vx / 4, self.vy / 4, np.sin(self.th), np.cos(self.th), self.om / 6,
            self.ix / 4, self.iy / 4, self.prev_a[:, 0], self.prev_a[:, 1],
        ], axis=1).astype(np.float32)

    # ------------------------------------------------------------ step
    def step(self, actions):
        a = np.clip(np.asarray(actions, dtype=np.float64), -1, 1)
        cmd1, cmd2 = C.T_HOVER * (1 + a[:, 0]), C.T_HOVER * (1 + a[:, 1])
        dt = C.PHYS_DT
        for _ in range(C.SUBSTEPS):
            # the motors respond with a small delay
            k = dt / C.MOTOR_TAU
            self.t1 += (cmd1 - self.t1) * k
            self.t2 += (cmd2 - self.t2) * k
            # wind gusts (a "smooth" random process)
            self.gust += -self.gust / C.GUST_TAU * dt + self.gust_std * np.sqrt(2 * dt / C.GUST_TAU) * self.rng.normal(size=self.n)
            air = self.wind + self.gust
            F = self.t1 + self.t2
            ax = (-F * np.sin(self.th) - C.DRAG * (self.vx - air)) / self.mass
            ay = (F * np.cos(self.th) - C.DRAG * self.vy) / self.mass - C.G
            alpha = ((self.t2 - self.t1) * C.ARM - C.ANG_DRAG * self.om) / C.INERTIA
            self.vx += ax * dt
            self.vy += ay * dt
            self.om += alpha * dt
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.th += self.om * dt
        self.th = (self.th + np.pi) % (2 * np.pi) - np.pi
        self.t += C.CONTROL_DT
        ex, ey = self.tx - self.x, self.ty - self.y
        leak = 1 - C.CONTROL_DT / 6.0
        self.ix = np.clip(self.ix * leak + ex * C.CONTROL_DT, -4, 4)
        self.iy = np.clip(self.iy * leak + ey * C.CONTROL_DT, -4, 4)

        # ---- reward ----
        d = np.hypot(ex, ey)
        reward = (2.0 * (self.d_prev - d)                 # progress toward the target
                  + 0.6 * np.exp(-(d / 0.5) ** 2)         # bonus for hovering precisely at the target
                  + 0.1                                   # "alive bonus" - a crash loses it
                  - 0.004 * self.om ** 2                  # no spinning
                  - 0.02 * (a ** 2).sum(axis=1))          # saving energy
        self.d_prev = d
        crashed = ((self.y < 0.15) | (np.abs(self.th) > C.MAX_TILT) |
                   (self.x < -1) | (self.x > C.WORLD_W + 1) | (self.y > C.WORLD_H + 2))
        reward = np.where(crashed, C.CRASH_PENALTY, reward)
        self.prev_a = a
        self.ep_return += reward

        # switching the target midway (training only)
        switch = (self.t >= self.retarget_at) & ~crashed
        if switch.any():
            self.new_target(switch)

        truncated = (self.t >= C.EPISODE_SECONDS - 1e-9) & ~crashed
        self.done = crashed | truncated
        info = {"crashed": crashed, "dist": d, "ep_return": self.ep_return.copy()}
        return self.observe(), reward.astype(np.float32), crashed, truncated, info
