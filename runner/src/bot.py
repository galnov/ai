"""
A rule-based player, written by hand (no learning). Two uses:

1. A baseline for the evaluation: how far does a player with hand-written rules get?
2. Practice recordings (make_bot_demos.py): runs that look a bit like human play (with reaction
   delay and mistakes), so the whole imitation pipeline can be tested before real people play.
   For the real experiment, use recordings of people playing the app.

The rules:
  * a train ahead in my lane  -> move toward the lane where the next train is farthest away
  * a hurdle about to arrive  -> jump at the right moment
  * an overhead bar           -> slide at the right moment
"""
import numpy as np

from . import config as C


def _first_overlap_ticks(pz, v, z0, z1):
    """In how many ticks the runner will first / last overlap an obstacle at [z0, z1] (at the current speed)."""
    first = max(1, -(-(z0 - C.HALF_LEN - pz) // v))          # ceil
    last = (z1 + C.HALF_LEN - pz) // v + 1
    return first, last


class RuleBot:
    """
    skill=1: reacts instantly and never makes mistakes.
    skill<1: reacts later and sometimes misses or presses at the wrong time (more human-like).
    """

    def __init__(self, n, skill=1.0, seed=0):
        self.n, self.skill = n, skill
        self.rng = np.random.default_rng(seed)
        self.delay = np.zeros(n, np.int64)
        self.queue = [[] for _ in range(n)]          # actions waiting for the "reaction delay"

    def reset(self, idx=None):
        for e in (range(self.n) if idx is None else idx):
            self.queue[e] = []

    def act(self, game):
        out = np.zeros(game.n, np.int64)
        v_all = game.speed()
        for e in range(game.n):
            if not game.alive[e]:
                continue
            a = self._decide(game, e, int(v_all[e]))
            if self.skill < 1.0:
                a = self._humanize(e, a)
            out[e] = a
        return out

    # ------------------------------------------------------------ the rules
    def _decide(self, g, e, v):
        pz = int(g.pz[e])
        tgt = int(g.lc_to[e]) if g.lc_t[e] > 0 else int(g.lane[e])
        eff = int(g.effective_lane()[e])
        m = g.ob_typ[e] > 0
        lanes, z0s, z1s, typs = g.ob_lane[e][m], g.ob_z0[e][m], g.ob_z1[e][m], g.ob_typ[e][m]
        ahead = z1s + C.HALF_LEN >= pz

        # how far is the next train in each lane (0 = a train right next to me)
        free = np.full(C.N_LANES, 10**9)
        for l in range(C.N_LANES):
            t = ahead & (lanes == l) & (typs == C.TRAIN)
            if t.any():
                free[l] = max(0, int((z0s[t] - C.HALF_LEN - pz).min()))

        # trains: change lanes in good time
        if g.lc_t[e] == 0 and free[tgt] < v * 16:
            best, best_score = None, free[tgt]
            for l in range(C.N_LANES):
                if l == tgt:
                    continue
                path_ok = all(free[k] > v * 3 * abs(k - tgt) + v * 2 for k in range(min(l, tgt), max(l, tgt) + 1) if k != tgt)
                score = free[l] - 50 * abs(l - tgt)
                if path_ok and score > best_score:
                    best, best_score = l, score
            if best is not None:
                return C.LEFT if best < tgt else C.RIGHT

        # hurdles and bars: jump / slide at the right moment
        for typ, act, window in ((C.LOW, C.JUMP, (2, 11)), (C.HIGH, C.SLIDE, (1, 12))):
            sel = ahead & ((lanes == tgt) | (lanes == eff)) & (typs == typ)
            for z0, z1 in zip(z0s[sel], z1s[sel]):
                first, last = _first_overlap_ticks(pz, v, int(z0), int(z1))
                if typ == C.LOW and g.jt[e] > 0:
                    # already in the air: is the rest of the jump enough?
                    if first + g.jt[e] - 1 >= 2 and last + g.jt[e] - 1 <= 11:
                        continue
                if typ == C.HIGH and g.st[e] >= last:
                    continue
                if first <= window[0] + 1 and last <= window[1]:
                    return act
        return C.NOOP

    # ------------------------------------------------------------ human-like imperfection
    def _humanize(self, e, a):
        q = self.queue[e]
        miss = 0.25 * (1 - self.skill)
        if a != C.NOOP and self.rng.random() > miss and not any(x[1] == a for x in q):
            delay = int(self.rng.integers(0, 1 + round(3 * (1 - self.skill))))
            q.append([delay, a])
        if self.rng.random() < 0.01 * (1 - self.skill):               # a random unnecessary key press
            q.append([0, int(self.rng.integers(1, C.N_ACTIONS))])
        for x in q:
            x[0] -= 1
        ready = [x for x in q if x[0] < 0]
        if ready:
            q.remove(ready[0])
            return ready[0][1]
        return C.NOOP
