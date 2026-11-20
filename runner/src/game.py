"""
The game: an endless runner on a track with 3 lanes. Computed with NumPy for many games in parallel.

The runner moves forward on its own, faster and faster. Every tick (50 ms) the player chooses one of 5 actions:
run (do nothing), move left, move right, jump, or slide. There are three kinds of obstacles:
    hurdle        - low: jump over it
    overhead bar  - high: slide under it
    train         - long and tall: the only way past is another lane

Everything is whole numbers (centimeters, ticks) and the random numbers come from a small generator
(xorshift32) that is easy to write identically in Python and JavaScript. The browser version
(app/static/game.js) follows exactly the same steps, so a run is fully described by its seed
and the list of actions: that is all a recording needs to store.

The order of one tick (the same in both versions):
    1. apply the action        (start a lane change / jump / slide)
    2. move forward            (by the current speed)
    3. check collisions        (using the lane, height and slide state of this tick)
    4. advance the timers      (lane change, jump, slide)
    5. generate new obstacles  (so there are always obstacles up to 70 m ahead) and drop old ones
"""
import numpy as np

from . import config as C

MASK32 = 0xFFFFFFFF


def xorshift32(x):
    x ^= (x << 13) & MASK32
    x ^= x >> 17
    x ^= (x << 5) & MASK32
    return x & MASK32


def seed_state(seed):
    """The generator's starting state for a seed (game.js: (Math.imul(seed, 2654435761) + 12345) >>> 0)."""
    x = (int(seed) * 2654435761 + 12345) & MASK32
    if x == 0:
        x = 1
    for _ in range(4):
        x = xorshift32(x)
    return x


def jump_height(jt):
    """Height (cm) of the runner jt ticks into a jump: a parabola that peaks at JUMP_HEIGHT."""
    return (4 * C.JUMP_HEIGHT * jt * (C.JUMP_TICKS - jt)) // (C.JUMP_TICKS * C.JUMP_TICKS)


class RunnerGame:
    """n games in parallel. All state arrays have n rows."""

    def __init__(self, n):
        self.n = n
        z = lambda: np.zeros(n, np.int64)
        self.t, self.t_off, self.pz = z(), z(), z()
        self.lane, self.lc_from, self.lc_to, self.lc_t, self.pending = z(), z(), z(), z(), z()
        self.jt, self.st = z(), z()
        self.alive = np.ones(n, bool)
        self.crash_type = z()                        # what the runner hit (0 = nothing yet)
        k = C.MAX_OBSTACLES
        self.ob_lane = np.zeros((n, k), np.int64)
        self.ob_z0 = np.zeros((n, k), np.int64)
        self.ob_z1 = np.zeros((n, k), np.int64)
        self.ob_typ = np.zeros((n, k), np.int64)     # 0 = empty slot
        self.rng = [1] * n                           # generator state of each game (Python ints)
        self.next_z = [0] * n                        # where the next row of obstacles starts
        self.seeds = [0] * n

    # ------------------------------------------------------------ reset
    def reset(self, seeds, t_offs=None, idx=None):
        """Starts new games. idx = which games (default: all). t_offs = start as if t_off ticks had passed (faster)."""
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        t_offs = np.zeros(len(idx), np.int64) if t_offs is None else np.asarray(t_offs, np.int64)
        for arr in (self.t, self.pz, self.lc_from, self.lc_to, self.lc_t, self.pending, self.jt, self.st,
                    self.crash_type):
            arr[idx] = 0
        self.lane[idx] = 1
        self.lc_from[idx] = 1
        self.lc_to[idx] = 1
        self.t_off[idx] = t_offs
        self.alive[idx] = True
        self.ob_typ[idx] = 0
        for j, e in enumerate(idx):
            self.seeds[e] = int(seeds[j])
            self.rng[e] = seed_state(seeds[j])
            self.next_z[e] = C.FIRST_ROW
            self._generate(e)

    # ------------------------------------------------------------ obstacles
    def speed(self):
        return np.minimum(C.VMAX, C.V0 + (self.t + self.t_off) // C.ACCEL_EVERY)

    def _rand(self, e, n):
        self.rng[e] = xorshift32(self.rng[e])
        return self.rng[e] % n

    def _place(self, e, lane, z0, z1, typ):
        free = np.flatnonzero(self.ob_typ[e] == 0)
        if len(free):
            k = free[0]
            self.ob_lane[e, k], self.ob_z0[e, k], self.ob_z1[e, k], self.ob_typ[e, k] = lane, z0, z1, typ

    def _gen_row(self, e):
        """One row of obstacles. There is always at least one way through."""
        z = self.next_z[e]
        v = int(min(C.VMAX, C.V0 + (self.t[e] + self.t_off[e]) // C.ACCEL_EVERY))
        r = self._rand(e, 100)
        n = 1 if r < 35 else (2 if r < 80 else 3)
        if n == 1:
            lanes = [self._rand(e, 3)]
        elif n == 2:
            free = self._rand(e, 3)
            lanes = [l for l in range(3) if l != free]
        else:
            lanes = [0, 1, 2]
        types = []
        for _ in lanes:
            r = self._rand(e, 100)
            types.append(C.LOW if r < 35 else (C.HIGH if r < 65 else C.TRAIN))
        if n == 3 and all(t == C.TRAIN for t in types):        # three trains = no way through: open one lane
            k = self._rand(e, 3)
            types[k] = C.LOW if self._rand(e, 2) == 0 else C.HIGH
        end = z
        for lane, typ in zip(lanes, types):
            length = C.BAR_LEN if typ != C.TRAIN else 800 + self._rand(e, 17) * 100
            self._place(e, lane, z, z + length, typ)
            end = max(end, z + length)
        gap_ticks = 18 + self._rand(e, 16)                     # 0.9 to 1.7 seconds until the next row
        self.next_z[e] = end + v * gap_ticks

    def _generate(self, e):
        old = (self.ob_typ[e] > 0) & (self.ob_z1[e] < self.pz[e] - 1000)
        self.ob_typ[e, old] = 0
        while self.next_z[e] < self.pz[e] + C.GEN_AHEAD:
            self._gen_row(e)

    # ------------------------------------------------------------ one tick
    def effective_lane(self):
        """The lane used for collisions: during a lane change, the old lane for the first half."""
        return np.where(self.lc_t > C.LANE_TICKS // 2, self.lc_from, np.where(self.lc_t > 0, self.lc_to, self.lane))

    def step(self, actions):
        """Advances all live games by one tick. Returns (crashed this tick (n,), pressed a key (n,))."""
        a = np.asarray(actions, np.int64)
        live = self.alive.copy()
        a = np.where(live, a, C.NOOP)

        # 1. the action
        d = np.where(a == C.LEFT, -1, np.where(a == C.RIGHT, 1, 0))
        busy = self.lc_t > 0
        self.pending = np.where((d != 0) & busy, d, self.pending)
        target = self.lane + d
        start = (d != 0) & ~busy & (target >= 0) & (target < C.N_LANES)
        self._start_change(start, target)

        jump = (a == C.JUMP) & (self.jt == 0)
        self.jt[jump] = 1
        self.st[jump] = 0
        slide = a == C.SLIDE
        self.jt[slide] = 0
        self.st[slide] = C.SLIDE_TICKS

        # 2. move forward
        v = self.speed()
        pz_prev = self.pz.copy()
        self.pz = np.where(live, self.pz + v, self.pz)
        self.t = np.where(live, self.t + 1, self.t)

        # 3. collisions
        lane = self.effective_lane()[:, None]
        h = jump_height(self.jt)[:, None]
        hit = ((self.ob_typ > 0) & (self.ob_lane == lane)
               & (self.pz[:, None] + C.HALF_LEN >= self.ob_z0) & (pz_prev[:, None] - C.HALF_LEN <= self.ob_z1))
        deadly = hit & ((self.ob_typ == C.TRAIN)
                        | ((self.ob_typ == C.LOW) & (h <= C.HURDLE_CLEAR))
                        | ((self.ob_typ == C.HIGH) & (self.st[:, None] == 0)))
        crashed = live & deadly.any(axis=1)
        first = deadly.argmax(axis=1)
        self.crash_type = np.where(crashed, self.ob_typ[np.arange(self.n), first], self.crash_type)
        self.alive &= ~crashed

        # 4. timers
        changing = self.lc_t > 0
        self.lc_t = np.where(changing, self.lc_t - 1, 0)
        done = changing & (self.lc_t == 0)
        self.lane = np.where(done, self.lc_to, self.lane)
        again = done & (self.pending != 0)
        target = self.lane + self.pending
        ok = again & (target >= 0) & (target < C.N_LANES)
        self.pending = np.where(done, 0, self.pending)
        self._start_change(ok, target)

        jumping = self.jt > 0
        self.jt = np.where(jumping, self.jt + 1, 0)
        self.jt[self.jt >= C.JUMP_TICKS] = 0
        self.st = np.maximum(self.st - 1, 0)

        # 5. new obstacles (only for runners still alive)
        for e in np.flatnonzero(self.alive):
            if self.next_z[e] < self.pz[e] + C.GEN_AHEAD:
                self._generate(e)
        return crashed, live & (a != C.NOOP)

    def _start_change(self, mask, target):
        self.lc_from = np.where(mask, self.lane, self.lc_from)
        self.lc_to = np.where(mask, target, self.lc_to)
        self.lc_t = np.where(mask, C.LANE_TICKS, self.lc_t)

    # ------------------------------------------------------------ what the network sees
    def observe(self):
        """(n, OBS_SIZE) float32: the track ahead as a 3 x 32 x 3 picture (flattened) + 9 runner features."""
        n = self.n
        rel0 = (self.ob_z0 - self.pz[:, None])[:, :, None]
        rel1 = (self.ob_z1 - self.pz[:, None])[:, :, None]
        lo = (np.arange(C.N_BINS) * C.BIN_CM)[None, None, :]
        occ = (rel0 < lo + C.BIN_CM) & (rel1 >= lo)                      # (n, K, bins)
        grid = np.zeros((n, 3, C.N_BINS, C.N_LANES), np.float32)
        for c, typ in enumerate((C.LOW, C.HIGH, C.TRAIN)):
            for lane in range(C.N_LANES):
                m = ((self.ob_typ == typ) & (self.ob_lane == lane))[:, :, None]
                grid[:, c, :, lane] = (occ & m).any(axis=1)

        feats = np.zeros((n, C.N_FEATURES), np.float32)
        tgt = np.where(self.lc_t > 0, self.lc_to, self.lane)
        feats[np.arange(n), tgt] = 1.0                                     # 0-2: the lane it is in / heading to
        feats[:, 3] = self.lc_t / C.LANE_TICKS                             # 3: lane change still to go
        feats[:, 4] = self.pending                                         # 4: a queued lane change (-1/0/1)
        feats[:, 5] = jump_height(self.jt) / C.JUMP_HEIGHT                 # 5: height
        feats[:, 6] = self.jt / C.JUMP_TICKS                               # 6: how far into the jump
        feats[:, 7] = self.st / C.SLIDE_TICKS                              # 7: slide time left
        feats[:, 8] = (self.speed() - C.V0) / (C.VMAX - C.V0)              # 8: speed (0 = start, 1 = top)
        return np.concatenate([grid.reshape(n, -1), feats], axis=1)

    def distance_m(self):
        return self.pz / 100.0


# ---------------------------------------------------------------- mirror images (for data augmentation)
MIRROR_ACTION = np.array([C.NOOP, C.RIGHT, C.LEFT, C.JUMP, C.SLIDE])


def mirror_obs(obs):
    """The same moment seen in a mirror: lanes swapped left <-> right."""
    g = C.GRID_SHAPE
    n_grid = g[0] * g[1] * g[2]
    grid = obs[:, :n_grid].reshape(-1, *g)[:, :, :, ::-1].reshape(len(obs), -1)
    f = obs[:, n_grid:].copy()
    f[:, 0:3] = f[:, 2::-1]
    f[:, 4] = -f[:, 4]
    return np.concatenate([grid, f], axis=1).astype(np.float32)
