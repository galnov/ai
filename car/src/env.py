"""
The learning environment: several cars on the same track, computed together (vectorized) with NumPy.

A Gym-style interface:
    obs = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(actions)
    obs = env.auto_reset()      # resets cars that finished, returns the observations for the next step

What the car "sees" (the observation, 10 numbers):
    9 distance sensors to the walls (0 = wall right here, 1 = no wall in range) + the speed (0..1)

The reward at every step:
    + the progress along the track (about 1 at top speed)
    - a small time penalty
    - a large penalty for hitting a wall (and the episode ends)
"""
import numpy as np

from . import config as C
from .track import Track

WINDOW = 14  # how many points to search around the previous position on the center line


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def raycast(origins, angles, wall_a, wall_b, max_len=C.RAY_LENGTH):
    """
    origins (N,2), angles (N,R) -> distances (N,R) to the nearest wall in each direction.
    Intersection of the ray o + t*d with the segment p + u*(q-p).
    """
    d = np.stack([np.cos(angles), np.sin(angles)], axis=-1)          # (N,R,2)
    e = wall_b - wall_a                                               # (S,2)
    po = wall_a[None, None, :, :] - origins[:, None, None, :]         # (N,1,S,2)
    denom = _cross(d[:, :, None, :], e[None, None, :, :])             # (N,R,S)
    safe = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    t = _cross(po, e[None, None, :, :]) / safe
    u = _cross(po, d[:, :, None, :]) / safe
    hit = (t > 0) & (u >= 0) & (u <= 1) & (np.abs(denom) >= 1e-9)
    t = np.where(hit, t, np.inf)
    return np.minimum(t.min(axis=2), max_len)


class CarEnv:
    def __init__(self, track: Track, n_cars=1, rng=None, random_start=True):
        self.n = n_cars
        self.rng = rng or np.random.default_rng()
        self.random_start = random_start
        self.x = np.zeros(n_cars)
        self.y = np.zeros(n_cars)
        self.heading = np.zeros(n_cars)
        self.speed = np.zeros(n_cars)
        self.idx = np.zeros(n_cars, dtype=np.int64)     # nearest point on the center line
        self.progress = np.zeros(n_cars)                 # accumulated distance along the track
        self.best = np.zeros(n_cars)
        self.steps = np.zeros(n_cars, dtype=np.int64)
        self.since_best = np.zeros(n_cars, dtype=np.int64)
        self.laps = np.zeros(n_cars, dtype=np.int64)
        self.lap_start = np.zeros(n_cars, dtype=np.int64)
        self.last_lap_time = np.full(n_cars, np.nan)
        self.best_lap_time = np.full(n_cars, np.nan)
        self.ep_return = np.zeros(n_cars)
        self.done = np.zeros(n_cars, dtype=bool)
        self.rays = np.zeros((n_cars, C.N_RAYS))
        self.set_track(track)

    # ------------------------------------------------------------ track and reset
    def set_track(self, track):
        self.track = track
        self.reset()

    def reset(self, mask=None):
        mask = np.ones(self.n, dtype=bool) if mask is None else mask
        k = int(mask.sum())
        if k == 0:
            return self.observe()
        t = self.track
        if self.random_start:
            idx = self.rng.integers(0, t.n, size=k)
        else:
            idx = np.zeros(k, dtype=np.int64)
        self.idx[mask] = idx
        self.x[mask] = t.center[idx, 0]
        self.y[mask] = t.center[idx, 1]
        heading = np.arctan2(t.tangent[idx, 1], t.tangent[idx, 0])
        if self.random_start:
            heading = heading + self.rng.normal(0, 0.12, size=k)
            self.speed[mask] = self.rng.uniform(0, 0.5, size=k) * C.MAX_SPEED
        else:
            self.speed[mask] = 0
        self.heading[mask] = heading
        for arr in (self.progress, self.best, self.ep_return):
            arr[mask] = 0
        for arr in (self.steps, self.since_best, self.laps, self.lap_start):
            arr[mask] = 0
        self.last_lap_time[mask] = np.nan
        self.done[mask] = False
        return self.observe()

    def respawn(self, mask):
        """Puts crashed cars back in the middle of the track, where they were, at zero speed (for the app)."""
        if not mask.any():
            return
        t = self.track
        i = self.idx[mask]
        self.x[mask] = t.center[i, 0]
        self.y[mask] = t.center[i, 1]
        self.heading[mask] = np.arctan2(t.tangent[i, 1], t.tangent[i, 0])
        self.speed[mask] = 0
        self.best[mask] = self.progress[mask]
        self.since_best[mask] = 0
        self.done[mask] = False

    def auto_reset(self):
        return self.reset(self.done.copy()) if self.done.any() else self.observe()

    # ------------------------------------------------------------ observation
    def observe(self):
        origins = np.stack([self.x, self.y], axis=1)
        angles = self.heading[:, None] + C.RAY_ANGLES[None, :]
        self.rays = raycast(origins, angles, self.track.wall_a, self.track.wall_b)
        return np.concatenate([self.rays / C.RAY_LENGTH, (self.speed / C.MAX_SPEED)[:, None]],
                              axis=1).astype(np.float32)

    # ------------------------------------------------------------ position on the track
    def _locate(self):
        """Finds the nearest point on the center line (only around the previous position) and the distance from it."""
        t = self.track
        offs = np.arange(-WINDOW, WINDOW + 1)
        k = (self.idx[:, None] + offs[None, :]) % t.n                  # (N,W)
        a = t.center[k]                                                # (N,W,2)
        b = t.center[(k + 1) % t.n]
        ab = b - a
        p = np.stack([self.x, self.y], axis=1)[:, None, :]
        u = np.clip(np.sum((p - a) * ab, axis=2) / np.sum(ab * ab, axis=2), 0, 1)
        proj = a + ab * u[..., None]
        dist = np.linalg.norm(p - proj, axis=2)
        j = dist.argmin(axis=1)
        rows = np.arange(self.n)
        new_idx = k[rows, j]
        s = t.cum[new_idx] + u[rows, j] * t.seg_len[new_idx]          # position along the track
        return new_idx, s, dist[rows, j]

    # ------------------------------------------------------------ step
    def step(self, actions, controls=None):
        """
        actions  - an action number for each car (0..5, see config.ACTIONS)
        controls - (optional) direct (steer, throttle) arrays, e.g. for a human driver on the keyboard
        """
        if controls is None:
            act = np.array(C.ACTIONS)[np.asarray(actions, dtype=np.int64)]
            steer, throttle = act[:, 0].astype(float), act[:, 1].astype(float)
        else:
            steer, throttle = (np.asarray(c, dtype=float) for c in controls)
        dt = C.DT
        t = self.track
        _, s_old, _ = self._locate()

        # --- simple physics ---
        v = self.speed
        v = v + np.where(throttle > 0, C.ACCEL, 0) * dt
        v = v - np.where(throttle < 0, C.BRAKE, 0) * dt
        v = v - np.where(throttle == 0, C.COAST_DRAG, 0) * dt
        v = v - v * C.FRICTION * dt
        v = np.clip(v, 0, C.MAX_SPEED)
        grip = np.minimum(1.0, v / C.TURN_SPEED_FULL) * (1 - C.HIGH_SPEED_TURN_LOSS * v / C.MAX_SPEED)
        self.heading = self.heading + steer * C.TURN_RATE * grip * dt
        self.x = self.x + np.cos(self.heading) * v * dt
        self.y = self.y + np.sin(self.heading) * v * dt
        self.speed = v

        # --- progress along the track ---
        new_idx, s_new, dist = self._locate()
        ds = s_new - s_old
        ds = np.where(ds > t.length / 2, ds - t.length, ds)            # crossing the start line
        ds = np.where(ds < -t.length / 2, ds + t.length, ds)
        self.idx = new_idx
        self.progress += ds
        self.steps += 1

        improved = self.progress > self.best + 1e-6
        self.best = np.maximum(self.best, self.progress)
        self.since_best = np.where(improved, 0, self.since_best + 1)

        # laps
        lap_now = np.floor(self.progress / t.length).astype(np.int64)
        new_lap = lap_now > self.laps
        lap_time = (self.steps - self.lap_start) * dt
        self.last_lap_time = np.where(new_lap, lap_time, self.last_lap_time)
        better = new_lap & ~(lap_time >= self.best_lap_time)          # (nan means there's no lap yet)
        self.best_lap_time = np.where(better, lap_time, self.best_lap_time)
        self.lap_start = np.where(new_lap, self.steps, self.lap_start)
        self.laps = np.maximum(self.laps, lap_now)

        # --- reward and termination ---
        crashed = dist > (t.width / 2 - C.CAR_RADIUS)
        stuck = self.since_best >= C.STUCK_STEPS
        truncated = (self.steps >= C.MAX_EPISODE_STEPS) & ~crashed & ~stuck

        reward = ds / (C.MAX_SPEED * dt) + C.REWARD_TIME
        reward = np.where(crashed, C.REWARD_CRASH, reward)
        reward = np.where(stuck & ~crashed, reward + C.REWARD_STUCK, reward)
        terminated = crashed | stuck
        self.ep_return += reward
        self.done = terminated | truncated

        obs = self.observe()
        info = {"crashed": crashed, "stuck": stuck, "laps": self.laps.copy(),
                "ep_return": self.ep_return.copy(), "progress": self.progress / t.length}
        return obs, reward.astype(np.float32), terminated, truncated, info
