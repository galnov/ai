"""
The intersection simulation: cars, a traffic light and changing demand. Computed vectorized with NumPy, so many
intersections (E) can run in parallel during training.

The cars drive according to the IDM (Intelligent Driver Model), a standard model in traffic engineering:
every driver accelerates up to the desired speed, and brakes according to the distance and speed of the vehicle ahead or of the traffic light.
This creates real queues, a stop wave that travels backward, and "start-up time" when the light changes.

Data structure: for each intersection e, lane l and slot i in the lane:
    s[e,l,i]  distance to the stop line (positive = before the intersection, negative = after). inf = an empty slot.
    The cars are sorted: i=0 is closest to the line (the leader).
"""
import numpy as np

from . import config as C

GREEN, YELLOW, ALL_RED = 0, 1, 2
LEN = np.array([v[1] for v in C.VEHICLES])
PROB = np.array([v[2] for v in C.VEHICLES])
ACC_F = np.array([v[3] for v in C.VEHICLES])
STEPS_PER_DECISION = int(round(C.DECISION_EVERY / C.DT))


class Demand:
    """
    Demand (cars per second) for each lane, changing during the episode:
    it moves gradually from a starting rate to an ending rate, so the busy direction can change midway.
    """

    def __init__(self, n_envs, rng):
        self.rng = rng
        self.start = np.zeros((n_envs, C.N_LANES))
        self.end = np.zeros((n_envs, C.N_LANES))
        self.new(np.ones(n_envs, bool))

    def _draw(self, k):
        r = self.rng.uniform(*C.DEMAND_RANGE, size=(k, C.N_LANES))
        # no more than the intersection can handle: the sum of the busiest lane in each phase
        crit = r[:, :2].max(axis=1) + r[:, 2:].max(axis=1)
        limit = self.rng.uniform(0.25, 0.42, size=k)
        return r * np.minimum(1.0, limit / crit)[:, None]

    def new(self, mask):
        k = int(mask.sum())
        if k:
            self.start[mask] = self._draw(k)
            self.end[mask] = self._draw(k)

    def rates(self, t):
        f = np.clip(t / C.EPISODE_SECONDS, 0, 1)[:, None]
        return self.start * (1 - f) + self.end * f

    def set_manual(self, rates):
        """For the app: a constant demand set by the user."""
        self.start[:] = rates
        self.end[:] = rates


class TrafficEnv:
    def __init__(self, n_envs=1, seed=0, demand=None):
        self.E = n_envs
        self.rng = np.random.default_rng(seed)
        self.demand = demand or Demand(n_envs, self.rng)
        shape = (n_envs, C.N_LANES, C.MAX_CARS)
        self.s = np.full(shape, np.inf)
        self.v = np.zeros(shape)
        self.kind = np.zeros(shape, np.int64)
        self.wait = np.zeros(shape)                 # accumulated waiting time per vehicle
        self.committed = np.zeros(shape, bool)      # a vehicle that can no longer stop at yellow
        self.cid = np.zeros(shape, np.int64)        # an ID number (for a fixed color when drawing)
        self.next_id = 0
        self.reset()

    # ------------------------------------------------------------ reset
    def reset(self, mask=None):
        m = np.ones(self.E, bool) if mask is None else mask
        self.s[m] = np.inf
        self.v[m] = 0
        self.wait[m] = 0
        self.committed[m] = False
        if not hasattr(self, "t"):
            E = self.E
            self.t = np.zeros(E)
            self.phase = np.zeros(E, np.int64)
            self.mode = np.zeros(E, np.int64)
            self.mode_timer = np.zeros(E)
            self.green_time = np.zeros(E)
            self.backlog = np.zeros((E, C.N_LANES), np.int64)
            self.total_wait = np.zeros(E)
            self.arrived = np.zeros(E, np.int64)
            self.served = np.zeros(E, np.int64)
            self.switches = np.zeros(E, np.int64)
        self.t[m] = 0
        self.phase[m] = self.rng.integers(0, 2, size=int(m.sum()))
        self.mode[m] = GREEN
        self.mode_timer[m] = 0
        self.green_time[m] = 0
        self.backlog[m] = 0
        self.total_wait[m] = 0
        self.arrived[m] = 0
        self.served[m] = 0
        self.switches[m] = 0
        self.demand.new(m)
        # start with some traffic, so we don't start from an empty intersection
        for _ in range(40):
            self._sim_step(mask=m)
        self.t[m] = 0
        self.total_wait[m] = 0
        self.arrived[m] = 0
        self.served[m] = 0

    # ------------------------------------------------------------ traffic light
    def green_lanes(self):
        """(E,4) - which lanes have green right now."""
        g = np.zeros((self.E, C.N_LANES), bool)
        for ph, lanes in enumerate(C.PHASE_LANES):
            on = (self.phase == ph) & (self.mode == GREEN)
            for l in lanes:
                g[:, l] = on
        return g

    def lane_light(self):
        """(E,4): 0 green, 1 yellow, 2 red - for drawing."""
        out = np.full((self.E, C.N_LANES), 2)
        for ph, lanes in enumerate(C.PHASE_LANES):
            for l in lanes:
                mine = self.phase == ph
                out[mine & (self.mode == GREEN), l] = 0
                out[mine & (self.mode == YELLOW), l] = 1
        return out

    def switch(self, want):
        """want (E,) bool: start a transition (yellow -> all red -> green for the other direction)."""
        go = want & (self.mode == GREEN)
        if not go.any():
            return
        self.mode[go] = YELLOW
        self.mode_timer[go] = C.YELLOW
        self.switches[go] += 1
        # cars too close to the line - continue through the yellow (they can't stop comfortably)
        stop_dist = self.v ** 2 / (2 * C.B_COMF * 1.4) + 1.0
        near = (self.s > 0) & (self.s < stop_dist)
        g = np.zeros((self.E, C.N_LANES), bool)
        for ph, lanes in enumerate(C.PHASE_LANES):
            for l in lanes:
                g[:, l] = go & (self.phase == ph)
        self.committed |= near & g[:, :, None]

    def _update_lights(self, mask):
        dt = C.DT
        m = mask
        self.green_time[m & (self.mode == GREEN)] += dt
        trans = m & (self.mode != GREEN)
        self.mode_timer[trans] -= dt
        to_red = trans & (self.mode == YELLOW) & (self.mode_timer <= 0)
        self.mode[to_red] = ALL_RED
        self.mode_timer[to_red] = C.ALL_RED
        to_green = trans & (self.mode == ALL_RED) & (self.mode_timer <= 0) & ~to_red
        self.mode[to_green] = GREEN
        self.phase[to_green] = 1 - self.phase[to_green]
        self.green_time[to_green] = 0
        self.committed[to_green] = False

    # ------------------------------------------------------------ simulation step
    def _sim_step(self, mask=None, spawn=None):
        m = np.ones(self.E, bool) if mask is None else mask
        dt = C.DT
        self._update_lights(m)
        active = np.isfinite(self.s)
        go = self.green_lanes()[:, :, None]

        # leading vehicle: the previous slot in the lane
        s_lead = np.concatenate([np.full(self.s.shape[:2] + (1,), -np.inf), self.s[:, :, :-1]], axis=2)
        v_lead = np.concatenate([np.zeros(self.s.shape[:2] + (1,)), self.v[:, :, :-1]], axis=2)
        len_lead = np.concatenate([np.zeros(self.s.shape[:2] + (1,)), LEN[self.kind[:, :, :-1]]], axis=2)
        with np.errstate(invalid="ignore"):
            gap = np.where(np.isfinite(s_lead), self.s - s_lead - len_lead, np.inf)
        dv = self.v - v_lead
        # the traffic light as a "standing obstacle" at the stop line
        must_stop = ~go & (self.s > 0) & ~self.committed & active
        stop_gap = self.s - 0.5
        use_stop = must_stop & (stop_gap < gap)
        gap = np.where(use_stop, stop_gap, gap)
        dv = np.where(use_stop, self.v, dv)

        # IDM model
        a = C.A_MAX * ACC_F[self.kind]
        s_star = C.S0 + np.maximum(0, self.v * C.T_HEADWAY + self.v * dv / (2 * np.sqrt(a * C.B_COMF)))
        with np.errstate(divide="ignore", invalid="ignore"):
            inter = np.where(np.isfinite(gap), (s_star / np.maximum(gap, 0.1)) ** 2, 0.0)
        acc = a * (1 - (self.v / C.V0) ** 4 - inter)
        acc = np.clip(acc, -9.0, a)
        v_new = np.maximum(0, self.v + acc * dt)
        s_new = self.s - (self.v + v_new) / 2 * dt
        # safety: don't pass the vehicle in front and don't cross a red light
        min_s = np.where(np.isfinite(s_lead), s_lead + len_lead + 0.5, -np.inf)
        hit = s_new < min_s
        s_new = np.where(hit, min_s, s_new)
        v_new = np.where(hit, np.minimum(v_new, v_lead), v_new)
        red_cross = must_stop & (s_new < 0.3)
        s_new = np.where(red_cross, 0.3, s_new)
        v_new = np.where(red_cross, 0.0, v_new)

        mm = m[:, None, None] & active
        self.s = np.where(mm, s_new, self.s)
        self.v = np.where(mm, v_new, self.v)

        # waiting
        waiting = mm & (self.v < C.WAIT_SPEED) & (self.s > 0)
        self.wait += waiting * dt
        n_wait = waiting.sum(axis=(1, 2)) + self.backlog.sum(axis=1)
        self.total_wait += np.where(m, n_wait * dt, 0)

        # leaving the intersection
        out = mm & (self.s < C.EXIT_AT)
        if out.any():
            self.served += out.sum(axis=(1, 2))
            self.s[out] = np.inf
            self.committed[out] = False
            self._sort()

        # new cars entering
        if spawn is None:
            rate = self.demand.rates(self.t)
            arrive = self.rng.random((self.E, C.N_LANES)) < 1 - np.exp(-rate * dt)
            kinds = self.rng.choice(len(LEN), size=(self.E, C.N_LANES), p=PROB)
        else:
            arrive, kinds = spawn
        arrive = arrive & m[:, None]
        self.arrived += arrive.sum(axis=1)
        self.backlog += arrive.astype(np.int64)
        self._enter(kinds, m)
        self.t[m] += dt
        return n_wait

    def _sort(self):
        order = np.argsort(self.s, axis=2, kind="stable")
        for name in ("s", "v", "kind", "wait", "committed", "cid"):
            setattr(self, name, np.take_along_axis(getattr(self, name), order, axis=2))

    def _enter(self, kinds, m):
        """Lets a car in from the "entrance queue" (backlog) if there's room at the start of the road."""
        count = np.isfinite(self.s).sum(axis=2)                   # (E,4)
        last = np.take_along_axis(self.s, np.maximum(count - 1, 0)[..., None], axis=2)[..., 0]
        last_v = np.take_along_axis(self.v, np.maximum(count - 1, 0)[..., None], axis=2)[..., 0]
        last = np.where(count > 0, last, -np.inf)
        room = (C.ROAD_LENGTH - last) > (LEN[kinds] + C.S0 + 6)
        ok = m[:, None] & (self.backlog > 0) & room & (count < C.MAX_CARS)
        if not ok.any():
            return
        e, l = np.nonzero(ok)
        i = count[e, l]
        self.s[e, l, i] = C.ROAD_LENGTH
        self.v[e, l, i] = np.where(np.isfinite(last[e, l]), np.minimum(C.V0, last_v[e, l] + 2), C.V0)
        self.kind[e, l, i] = kinds[e, l]
        self.wait[e, l, i] = 0
        self.committed[e, l, i] = False
        self.cid[e, l, i] = self.next_id + np.arange(len(e))
        self.next_id += len(e)
        self.backlog[e, l] -= 1

    # ------------------------------------------------------------ interface for reinforcement learning
    def counts(self):
        """(E,4) - the "true" count: how many vehicles are in the camera range in each lane."""
        return ((self.s >= 0) & (self.s <= C.VIEW_RANGE)).sum(axis=2)

    def observe(self, counts=None):
        """
        The agent's observation (6 numbers), ordered relative to the current phase:
          [count in green lane 1, green 2, red 1, red 2, mid-transition?, how long it has been green]
        counts - the count from the camera (if None: the true count from the simulation).
        """
        c = self.counts() if counts is None else np.asarray(counts, dtype=np.float64)
        ph = self.phase
        g = np.where(ph[:, None] == 0, c[:, [0, 1]], c[:, [2, 3]])
        r = np.where(ph[:, None] == 0, c[:, [2, 3]], c[:, [0, 1]])
        return np.concatenate([
            g / C.COUNT_SCALE, r / C.COUNT_SCALE,
            (self.mode != GREEN)[:, None].astype(float),
            np.minimum(self.green_time / 60.0, 1.0)[:, None],
        ], axis=1).astype(np.float32)

    def step(self, actions):
        """
        Decides (0 keep / 1 switch) and runs 5 seconds of simulation.
        Reward = minus the average number of waiting vehicles (divided by 10).
        """
        self.switch(np.asarray(actions) == 1)
        total = np.zeros(self.E)
        for _ in range(STEPS_PER_DECISION):
            total += self._sim_step()
        reward = -(total / STEPS_PER_DECISION) / 10.0
        done = self.t >= C.EPISODE_SECONDS
        return reward.astype(np.float32), done

    def metrics(self):
        """Average waiting time per vehicle, and the average queue."""
        return {
            "avg_wait": self.total_wait / np.maximum(self.arrived, 1),
            "avg_queue": self.total_wait / np.maximum(self.t, 1e-9),
            "served": self.served.copy(),
            "arrived": self.arrived.copy(),
            "switches": self.switches.copy(),
        }


# ---------------------------------------------------------------- baseline controllers for comparison
def fixed_time(env, green=C.FIXED_GREEN):
    """A regular traffic light: switches every 30 seconds, regardless of traffic."""
    return (env.green_time >= green - 1e-6).astype(np.int64)


def max_queue(env, counts=None, min_green=10.0, margin=2):
    """A simple "smart" traffic light: switches when the red direction has more vehicles (and at least 10 seconds of green)."""
    c = env.counts() if counts is None else counts
    ph = env.phase
    g = np.where(ph == 0, c[:, 0] + c[:, 1], c[:, 2] + c[:, 3])
    r = np.where(ph == 0, c[:, 2] + c[:, 3], c[:, 0] + c[:, 1])
    return ((env.green_time >= min_green) & (r > g + margin)).astype(np.int64)
