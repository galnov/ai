"""
The learning environment: spaceships flying through 3D mazes. Computed with NumPy for many ships in parallel,
each in its own maze.

The ship's orientation - spaceship controls (like the game Descent): in space there is no "up".
The ship has its own three axes: forward f (the nose), up u (the top of the screen) and right r = f x u.
  * Turn (yaw): the nose turns toward the ship's own right:    f <- f cos a + r sin a
  * Pitch: the nose lifts toward the ship's own up:            f <- f cos b + u sin b,  u <- u cos b - f sin b
  * So "right" always means the right side of the screen and "up" the top of the screen - also in vertical
    shafts - and nothing turns unless the pilot turns it.
  * After several turns, what was the floor may appear at the side of the screen. In a square corridor
    with the same grid on every wall, that looks exactly the same.

The motion:
  * Throttle: sets a target speed between 2 and 11 m/s.
  * Inertia: the velocity follows the nose gradually (dv/dt = k·(target - v)), so in a sharp turn the ship
    drifts outward - it has to slow down before corners. At top speed the turning radius (v / turn rate)
    is 5 m: wider than the whole corridor.
  * Collisions: 26 points on the ship's surface (a ball of radius 0.6 m) are checked against the maze cells.

The sensors (the observation, 93 numbers):
  * a depth camera: 9 x 9 rays over 100 degrees in front of the ship, each giving the distance to the
    nearest wall (by stepping through the grid cells along the ray - the DDA algorithm used in games)
  * the navigation beacon: where the next two cells on the way out are, relative to the ship
  * the ship's velocity relative to its own axes, and the previous action

The reward:  +0.5 per meter of progress along the way out, +20 for the exit, -20 for touching a wall,
and a small cost for jerky steering.
"""
import numpy as np

from . import config as C
from .maze import Maze

def _normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def ray_directions():
    """The 81 camera rays in the ship's own frame (right, up, forward), top row first."""
    t = np.tan(np.radians(C.FOV / 2))
    xs = np.linspace(-t, t, C.RAYS_X)
    ys = np.linspace(t, -t, C.RAYS_Y)
    d = np.array([[x, y, 1.0] for y in ys for x in xs])
    return _normalize(d)


def collision_offsets():
    """Points on the ship's surface that are checked against the walls (26 directions)."""
    pts = [np.array([i, j, k], float) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    return np.array([p / np.linalg.norm(p) for p in pts]) * C.SHIP_RADIUS


def start_axes(d):
    """Initial (f, u): the nose along d; "up" is the world's up unless the nose points straight up or down."""
    f = np.asarray(d, float) / max(np.linalg.norm(d), 1e-9)
    u = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.5 else np.array([1.0, 0.0, 0.0])
    return f, u


RAY_DIRS = ray_directions()
OFFSETS = collision_offsets()
_POOLS = {}


def training_pool(sizes, n=C.TRAIN_POOL):
    """Training mazes, generated once (making a maze takes a few milliseconds, and ships crash often)."""
    key = (tuple(sizes), n)
    if key not in _POOLS:
        rng = np.random.default_rng(12345)
        seeds = rng.integers(1_000_000, 2**31, n)        # far away from the validation and test seeds
        _POOLS[key] = [Maze(int(s), int(sizes[i % len(sizes)])) for i, s in enumerate(seeds)]
    return _POOLS[key]


class ShipEnv:
    def __init__(self, n, seed=0, sizes=C.TRAIN_SIZES, random_start=0.5, time_limit=C.TIME_LIMIT):
        self.n, self.sizes, self.random_start = n, sizes, random_start
        self.max_steps = int(round(time_limit / C.DT))
        self.rng = np.random.default_rng(seed)
        self.G = 2 * max(sizes) + 1
        G = self.G
        self.solid = np.ones((n, G, G, G), bool)
        self.dist = np.zeros((n, G, G, G), np.int64)
        self.next_dir = np.zeros((n, G, G, G, 3), np.int64)
        self.exit = np.zeros((n, 3), np.int64)
        self.mazes = [None] * n
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.f = np.zeros((n, 3))       # the nose
        self.u = np.zeros((n, 3))       # the ship's up (the top of the screen)
        self.prev_a = np.zeros((n, 3))
        self.t = np.zeros(n, np.int64)
        self.L = np.zeros(n)            # remaining distance to the exit along the way (m)
        self.L0 = np.ones(n)
        self.ep_ret = np.zeros(n)
        self.done = np.zeros(n, bool)
        self.depth = np.zeros((n, C.N_RAYS))

    # ------------------------------------------------------------ mazes and resets
    def load_maze(self, i, maze, start_cell=None):
        G = maze.G
        self.mazes[i] = maze
        self.solid[i] = True
        self.solid[i, :G, :G, :G] = maze.solid
        self.dist[i] = 0
        self.dist[i, :G, :G, :G] = maze.dist
        self.next_dir[i] = 0
        self.next_dir[i, :G, :G, :G] = maze.next_dir
        self.exit[i] = maze.exit
        cell = maze.start if start_cell is None else start_cell
        d = maze.next_dir[cell].astype(float)
        self.pos[i] = maze.center(cell)
        self.f[i], self.u[i] = start_axes(d)
        self.vel[i] = d * (C.V_MIN + 1.0)
        self.prev_a[i] = 0
        self.t[i] = 0
        self.ep_ret[i] = 0
        self.done[i] = False

    def _reset_one(self, i):
        pool = training_pool(self.sizes)
        maze = pool[self.rng.integers(len(pool))]
        start = None
        if self.rng.random() < self.random_start:          # sometimes start somewhere in the middle
            cells = np.argwhere(maze.dist > 0)
            start = tuple(cells[self.rng.integers(len(cells))])
        self.load_maze(i, maze, start)

    def reset(self):
        for i in range(self.n):
            self._reset_one(i)
        self._after_reset(np.arange(self.n))
        return self.observe()

    def set_mazes(self, mazes):
        """Fixed mazes, from their start (for evaluation and the app)."""
        assert len(mazes) == self.n
        for i, m in enumerate(mazes):
            self.load_maze(i, m)
        self._after_reset(np.arange(self.n))
        return self.observe()

    def _after_reset(self, idx):
        L = self.remaining()
        self.L[idx] = L[idx]
        self.L0[idx] = np.maximum(L[idx], 1.0)
        self.depth[idx] = self.cast_rays()[idx]

    def auto_reset(self):
        idx = np.flatnonzero(self.done)
        for i in idx:
            self._reset_one(i)
        if len(idx):
            self._after_reset(idx)
        return self.observe()

    # ------------------------------------------------------------ the world
    def cells(self, p):
        return np.floor(p / C.CELL).astype(np.int64)

    def is_solid(self, env_idx, p):
        """env_idx (M,), p (M, 3) world points -> (M,) True if inside rock or outside the maze."""
        c = self.cells(p)
        inside = np.all((c >= 0) & (c < self.G), axis=1)
        c = np.clip(c, 0, self.G - 1)
        return ~inside | self.solid[env_idx, c[:, 0], c[:, 1], c[:, 2]]

    def remaining(self):
        """Distance left to the exit along the way (m), measured smoothly inside each cell."""
        c = np.clip(self.cells(self.pos), 0, self.G - 1)
        e = np.arange(self.n)
        d = self.dist[e, c[:, 0], c[:, 1], c[:, 2]]
        nd = self.next_dir[e, c[:, 0], c[:, 1], c[:, 2]]
        center = (c + 0.5) * C.CELL
        L = d * C.CELL - np.sum((self.pos - center) * nd, axis=1)
        return np.where(d >= 0, L, self.L)

    def cast_rays(self):
        """(n, 81) distances to the walls, by walking the grid cell by cell along each ray (DDA)."""
        n, R = self.n, C.N_RAYS
        r = np.cross(self.f, self.u)
        D = (RAY_DIRS[None, :, 0:1] * r[:, None] + RAY_DIRS[None, :, 1:2] * self.u[:, None]
             + RAY_DIRS[None, :, 2:3] * self.f[:, None]).reshape(-1, 3)
        P = np.repeat(self.pos / C.CELL, R, axis=0)
        env = np.repeat(np.arange(n), R)
        cell = np.floor(P).astype(np.int64)
        step = np.where(D > 0, 1, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(np.abs(D) > 1e-12, 1.0 / D, np.inf)
            t_delta = np.abs(inv)
            t_max = np.where(np.abs(D) > 1e-12, ((cell + (step > 0)) - P) * inv, np.inf)
        max_t = C.RAY_RANGE / C.CELL
        hit = np.full(len(P), max_t)
        alive = np.ones(len(P), bool)
        rows = np.arange(len(P))
        for _ in range(3 * int(np.ceil(max_t)) + 3):
            ax = np.argmin(t_max, axis=1)
            t_here = t_max[rows, ax]
            cell[rows, ax] += step[rows, ax]
            t_max[rows, ax] += t_delta[rows, ax]
            alive &= t_here < max_t
            inside = np.all((cell >= 0) & (cell < self.G), axis=1)
            cc = np.clip(cell, 0, self.G - 1)
            wall = ~inside | self.solid[env, cc[:, 0], cc[:, 1], cc[:, 2]]
            new_hit = alive & wall
            hit[new_hit] = t_here[new_hit]
            alive &= ~wall
            if not alive.any():
                break
        return np.minimum(hit * C.CELL, C.RAY_RANGE).reshape(n, R)

    def beacons(self):
        """World positions of the centers of the next two cells on the way out."""
        e = np.arange(self.n)
        c = np.clip(self.cells(self.pos), 0, self.G - 1)
        d1 = self.next_dir[e, c[:, 0], c[:, 1], c[:, 2]]
        c1 = np.clip(c + d1, 0, self.G - 1)
        d2 = self.next_dir[e, c1[:, 0], c1[:, 1], c1[:, 2]]
        c2 = np.clip(c1 + d2, 0, self.G - 1)
        return (c1 + 0.5) * C.CELL, (c2 + 0.5) * C.CELL

    def to_body(self, v):
        r = np.cross(self.f, self.u)
        return np.stack([np.sum(v * r, 1), np.sum(v * self.u, 1), np.sum(v * self.f, 1)], axis=1)

    def observe(self):
        b1, b2 = self.beacons()
        return np.concatenate([
            self.depth / C.RAY_RANGE,
            self.to_body(b1 - self.pos) / C.BEACON_SCALE,
            self.to_body(b2 - self.pos) / C.BEACON_SCALE,
            self.to_body(self.vel) / C.V_MAX,
            self.prev_a,
        ], axis=1).astype(np.float32)

    # ------------------------------------------------------------ one decision (0.1 s)
    def step(self, actions):
        a = np.clip(np.asarray(actions, float), -1.0, 1.0)
        live = ~self.done
        a[~live] = 0
        dt = C.DT / C.SUBSTEPS
        target_speed = C.V_MIN + (a[:, 2] + 1) / 2 * (C.V_MAX - C.V_MIN)
        crashed = np.zeros(self.n, bool)
        e_all = np.arange(self.n)
        for _ in range(C.SUBSTEPS):
            self._rotate(a[:, 0] * C.PITCH_RATE * dt, a[:, 1] * C.YAW_RATE * dt, live)
            k = 1 - np.exp(-C.ACCEL_K * dt)
            self.vel = np.where(live[:, None], self.vel + (self.f * target_speed[:, None] - self.vel) * k, self.vel)
            self.pos = np.where((live & ~crashed)[:, None], self.pos + self.vel * dt, self.pos)
            pts = (self.pos[:, None, :] + OFFSETS[None]).reshape(-1, 3)
            hit = self.is_solid(np.repeat(e_all, len(OFFSETS)), pts).reshape(self.n, -1).any(1)
            crashed |= live & hit
        self.t += live

        L = self.remaining()
        progress = np.where(live & ~crashed, self.L - L, 0.0)
        self.L = np.where(live & ~crashed, L, self.L)
        c = np.clip(self.cells(self.pos), 0, self.G - 1)
        reached = live & ~crashed & np.all(c == self.exit, axis=1)
        trunc = live & ~crashed & ~reached & (self.t >= self.max_steps)

        r = (C.R_PROGRESS * progress + C.R_EXIT * reached + C.R_CRASH * crashed
             - C.R_SMOOTH * np.sum((a - self.prev_a) ** 2, axis=1))
        r = np.where(live, r, 0.0)
        self.prev_a = np.where(live[:, None], a, self.prev_a)
        self.ep_ret += r
        term = crashed | reached
        self.done = self.done | term | trunc
        self.depth = self.cast_rays()
        info = {"ep_return": self.ep_ret.copy(), "reached": reached, "crashed": crashed,
                "time": self.t * C.DT, "progress": 1 - self.L / self.L0}
        return self.observe(), r, term, trunc, info

    def _rotate(self, pitch, yaw, live):
        """Turn around the ship's own axes: yaw toward its right, pitch toward its up."""
        f, u = self.f, self.u
        r = np.cross(f, u)
        cy, sy = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        f = f * cy + r * sy
        cp, sp = np.cos(pitch)[:, None], np.sin(pitch)[:, None]
        f, u = f * cp + u * sp, u * cp - f * sp
        f = _normalize(f)                                            # keep the axes exactly perpendicular
        u = _normalize(u - np.sum(u * f, 1, keepdims=True) * f)
        self.f = np.where(live[:, None], f, self.f)
        self.u = np.where(live[:, None], u, self.u)

    # ------------------------------------------------------------ for the app
    def frame(self, i):
        return {"pos": self.pos[i].round(3).tolist(), "f": self.f[i].round(4).tolist(),
                "u": self.u[i].round(4).tolist(), "speed": float(np.linalg.norm(self.vel[i])),
                "depth": (self.depth[i] / C.RAY_RANGE).round(2).tolist()}
