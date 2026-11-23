"""
The air hockey table: many tables at once (NumPy arrays), for fast training.

Two players on every table: player 0 at the bottom (defends y = -1), player 1 at the top (defends y = +1).
Every decision (1/30 s) is 8 physics steps of 1/240 s:
  1. Paddles: each one tries to reach the velocity its player asked for, but can only change its velocity
     by PADDLE_ACCEL per second and never goes faster than PADDLE_VMAX - the same limits for the AI and for
     a human with a mouse. A paddle can't leave its own half.
  2. The puck glides with a little friction, and bounces off the side walls and the end walls (outside the goal).
  3. Paddle hits: if the puck overlaps a paddle, it is pushed out, and bounces off the paddle as if the paddle
     were a moving wall: the puck's velocity *relative to the paddle* is reflected. A paddle moving into the
     puck hits it hard; a paddle standing still just blocks it.
  4. A goal: the puck's center crosses the end line inside the goal mouth.

The same rules, in the same order, are in docs/physics.js (the web page). They were checked to give the same
result, step by step.
"""
import numpy as np

from . import config as C

LO_X, HI_X = -C.HALF_W, C.HALF_W


class HockeyEnv:
    def __init__(self, n, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.pp = np.zeros((n, 2, 2))      # paddle positions [table, player, xy]
        self.pv = np.zeros((n, 2, 2))      # paddle velocities
        self.kp = np.zeros((n, 2))         # puck position
        self.kv = np.zeros((n, 2))         # puck velocity
        self.t = np.zeros(n)               # time in the current point
        self.score = np.zeros((n, 2), int)
        self.hist = np.zeros((C.REACTION_STEPS + 1, n, 2, C.OBS_SIZE))    # recent observations (reaction time)
        self.reset(np.arange(n), server=self.rng.integers(0, 2, n))

    # ---------------------------------------------------------------- a new point
    def reset(self, idx, server):
        """server: the player who gets the puck (the one who conceded)."""
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        side = np.where(np.asarray(server) == 0, -1.0, 1.0)
        self.pp[idx, 0] = [0.0, -0.8]
        self.pp[idx, 1] = [0.0, 0.8]
        self.pv[idx] = 0
        self.kp[idx, 0] = self.rng.uniform(-0.15, 0.15, idx.size)
        self.kp[idx, 1] = side * 0.3
        self.kv[idx, 0] = 0
        self.kv[idx, 1] = side * C.SERVE_SPEED
        self.t[idx] = 0
        obs = self._observe_now()[idx]
        self.hist[:, idx] = obs[None]

    # ---------------------------------------------------------------- what each player sees
    def _observe_now(self):
        """(n, 2, OBS_SIZE): each player's view from its own side (player 1's view turned 180 degrees)."""
        o = np.zeros((self.n, 2, C.OBS_SIZE))
        for p in (0, 1):
            s = 1.0 if p == 0 else -1.0
            sc = np.array([1 / C.HALF_W, 1 / C.HALF_L])
            o[:, p, 0:2] = s * self.pp[:, p] * sc
            o[:, p, 2:4] = s * self.pv[:, p] / C.PADDLE_VMAX
            o[:, p, 4:6] = s * self.pp[:, 1 - p] * sc
            o[:, p, 6:8] = s * self.pv[:, 1 - p] / C.PADDLE_VMAX
            o[:, p, 8:10] = s * self.kp * sc
            o[:, p, 10:12] = s * self.kv / C.PUCK_VMAX
        return o

    def observe(self, delayed=True):
        """(n, 2, OBS_SIZE). delayed: the puck and the opponent as they were REACTION_STEPS decisions ago
        (what the eyes report), the player's own paddle as it is now (a player feels where its own hand is)."""
        if not delayed:
            return self._observe_now()
        o = self.hist[0].copy()
        o[:, :, 0:4] = self.hist[-1][:, :, 0:4]
        return o

    # ---------------------------------------------------------------- one decision (1/30 s)
    def step(self, actions):
        """actions: (n, 2, 2) in [-1, 1], each player's in its own view.
        Returns reward (n, 2), goal (n: +1 player 0 scored, -1 player 1 scored, 0 none), timeout (n),
        touched (n, 2). Finished points are reset at once (the returned observation is of the new point)."""
        a = np.clip(actions, -1, 1).astype(float)
        a[:, 1] *= -1                                         # player 1's view is turned around
        want = a * C.PADDLE_VMAX
        h = C.DT / C.SUBSTEPS
        live = np.ones(self.n, bool)
        goal = np.zeros(self.n)
        touched = np.zeros((self.n, 2), bool)
        y0 = self.kp[:, 1].copy()
        for _ in range(C.SUBSTEPS):
            m = live[:, None]
            # 1. paddles
            dv = want - self.pv
            dn = np.linalg.norm(dv, axis=-1, keepdims=True)
            lim = C.PADDLE_ACCEL * h
            dv = np.where(dn > lim, dv * (lim / np.maximum(dn, 1e-12)), dv)
            v = self.pv + dv
            vn = np.linalg.norm(v, axis=-1, keepdims=True)
            v = np.where(vn > C.PADDLE_VMAX, v * (C.PADDLE_VMAX / np.maximum(vn, 1e-12)), v)
            p = self.pp + v * h
            lo = np.array([[LO_X + C.PADDLE_R, -C.HALF_L + C.PADDLE_R], [LO_X + C.PADDLE_R, C.PADDLE_R]])
            hi = np.array([[HI_X - C.PADDLE_R, -C.PADDLE_R], [HI_X - C.PADDLE_R, C.HALF_L - C.PADDLE_R]])
            pc = np.clip(p, lo, hi)
            v = np.where(pc != p, 0.0, v)                     # stopped by the wall or the center line
            self.pp = np.where(m[:, :, None], pc, self.pp)
            self.pv = np.where(m[:, :, None], v, self.pv)
            # 2. the puck glides
            kv = self.kv * np.exp(-C.PUCK_FRICTION * h)
            kp = self.kp + kv * h
            kp, kv = self._walls(kp, kv)
            # 3. paddle hits
            for q in (0, 1):
                d = kp - self.pp[:, q]
                dist = np.linalg.norm(d, axis=-1)
                hit = (dist < C.PUCK_R + C.PADDLE_R) & (dist > 1e-9)
                nrm = d / np.maximum(dist, 1e-9)[:, None]
                kp = np.where(hit[:, None], self.pp[:, q] + nrm * (C.PUCK_R + C.PADDLE_R), kp)
                rel = kv - self.pv[:, q]
                vn_ = (rel * nrm).sum(-1)
                bounce = hit & (vn_ < 0)
                kv = np.where(bounce[:, None], kv - (1 + C.PADDLE_E) * vn_[:, None] * nrm, kv)
                touched[:, q] |= bounce & live
            sp = np.linalg.norm(kv, axis=-1, keepdims=True)
            kv = np.where(sp > C.PUCK_VMAX, kv * (C.PUCK_VMAX / np.maximum(sp, 1e-12)), kv)
            kp, kv = self._walls(kp, kv)                      # a hit may have pushed it into a wall
            self.kp = np.where(m, kp, self.kp)
            self.kv = np.where(m, kv, self.kv)
            # 4. goals
            g = live & (np.abs(self.kp[:, 0]) < C.GOAL_HALF) & (np.abs(self.kp[:, 1]) > C.HALF_L)
            goal = np.where(g, np.sign(self.kp[:, 1]), goal)   # puck beyond y = +1: player 0 scored
            live &= ~g
        self.t += C.DT
        timeout = (goal == 0) & (self.t >= C.POINT_TIME)

        # rewards, for both players
        dy = self.kp[:, 1] - y0
        r = np.zeros((self.n, 2))
        r[:, 0] = C.GOAL_REWARD * goal + C.PUSH_REWARD * dy + C.TOUCH_REWARD * touched[:, 0]
        r[:, 1] = -C.GOAL_REWARD * goal - C.PUSH_REWARD * dy + C.TOUCH_REWARD * touched[:, 1]
        r[goal != 0] -= np.stack([C.PUSH_REWARD * dy, -C.PUSH_REWARD * dy], 1)[goal != 0]   # no push reward on the scoring step
        r[timeout] += C.TIMEOUT_REWARD
        self.score[goal > 0, 0] += 1
        self.score[goal < 0, 1] += 1

        # the reaction-time memory, then new points where one ended
        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = self._observe_now()
        done = (goal != 0) | timeout
        server = np.where(goal > 0, 1, np.where(goal < 0, 0, self.rng.integers(0, 2, self.n)))
        self.reset(np.flatnonzero(done), server[done])
        return r, goal, timeout, touched

    @staticmethod
    def _walls(kp, kv):
        kp, kv = kp.copy(), kv.copy()
        lo, hi = LO_X + C.PUCK_R, HI_X - C.PUCK_R
        left, right = kp[:, 0] < lo, kp[:, 0] > hi
        kp[:, 0] = np.where(left, 2 * lo - kp[:, 0], np.where(right, 2 * hi - kp[:, 0], kp[:, 0]))
        kv[:, 0] = np.where(left | right, -C.WALL_E * kv[:, 0], kv[:, 0])
        # the end walls, except the goal mouth
        wall = np.abs(kp[:, 0]) >= C.GOAL_HALF
        end = C.HALF_L - C.PUCK_R
        bot, top = wall & (kp[:, 1] < -end), wall & (kp[:, 1] > end)
        kp[:, 1] = np.where(bot, -2 * end - kp[:, 1], np.where(top, 2 * end - kp[:, 1], kp[:, 1]))
        kv[:, 1] = np.where(bot | top, -C.WALL_E * kv[:, 1], kv[:, 1])
        return kp, kv


def play_points(policy0, policy1, points=C.EVAL_POINTS, n=200, seed=123):
    """Plays until `points` points have finished. Returns (goals for player 0, goals for player 1, replays).
    A policy takes the observations (n, OBS_SIZE) of its side and returns actions (n, 2)."""
    env = HockeyEnv(n, seed)
    g0 = g1 = to = 0
    while g0 + g1 + to < points:
        o = env.observe()
        _, goal, timeout, _ = env.step(np.stack([policy0(o[:, 0]), policy1(o[:, 1])], 1))
        g0, g1, to = g0 + int((goal > 0).sum()), g1 + int((goal < 0).sum()), to + int(timeout.sum())
    return g0, g1, to
