"""
The simulation that runs in the app: two identical intersections with exactly the same traffic.
  Left: a regular traffic light (fixed or "longest queue")
  Right: the agent, which sees the intersection through the cameras (or through a "perfect sensor")
"""
import base64
import io
import threading

import numpy as np

from src import config as C
from src.camera import Scene, render_cameras, vehicle_list
from src.traffic import LEN, PROB, TrafficEnv, fixed_time, max_queue
from src.traffic import STEPS_PER_DECISION as SPD

# demand: cars per second in each lane [north, south, east, west]
PRESETS = {
    "normal": ("Normal", [0.15, 0.14, 0.12, 0.11]),
    "north": ("Heavy from the north", [0.30, 0.10, 0.08, 0.07]),
    "rush": ("Rush hour", [0.22, 0.20, 0.18, 0.16]),
    "quiet": ("Quiet night", [0.05, 0.03, 0.04, 0.03]),
    "waves": ("Alternating waves", None),
    "custom": ("Manual", None),
}


def jpeg_b64(img, q=80):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


class World:
    def __init__(self, controller=None, counter=None):
        self.controller = controller
        self.counter = counter
        self.scene = Scene(104)          # an intersection that didn't appear in the camera network's training
        self.lock = threading.Lock()
        self.preset = "normal"
        self.custom = [0.15, 0.15, 0.12, 0.12]
        self.left_mode = "fixed"
        self.sensor = "camera" if counter else "perfect"
        self.condition = "day"
        self.reset()

    # ------------------------------------------------------------ settings
    def rates(self):
        t = self.envs[0].t[0]
        if self.preset == "waves":     # every 4 minutes the busy direction changes
            f = 0.5 + 0.5 * np.sin(2 * np.pi * t / 480)
            return np.array([0.06 + 0.24 * f, 0.05 + 0.2 * f, 0.06 + 0.24 * (1 - f), 0.05 + 0.2 * (1 - f)])
        if self.preset == "custom":
            return np.array(self.custom)
        return np.array(PRESETS[self.preset][1])

    def reset(self, seed=None):
        with self.lock:
            seed = int(np.random.default_rng(seed).integers(1 << 30))
            self.envs = []
            for _ in range(2):           # same seed = same starting state in both intersections
                env = TrafficEnv(1, seed=seed)
                self.envs.append(env)
            self.arrivals = np.random.default_rng(seed + 1)
            for env in self.envs:
                env.demand.set_manual(self.rates()[None])
                env.reset()
                env.demand.set_manual(self.rates()[None])
            self.steps = 0
            self.history = []
            self.camera = None
            self.cam_seq = 0
            self.brain = None
            self.cam_rng = np.random.default_rng(seed + 2)

    # ------------------------------------------------------------ decisions
    def _decide(self):
        left, right = self.envs
        a_left = fixed_time(left) if self.left_mode == "fixed" else max_queue(left)
        left.switch(a_left == 1)

        true = right.counts()[0]
        cam = None
        if self.counter is not None or self.sensor == "camera":
            views = render_cameras(self.scene, vehicle_list(right, 0), self.condition, self.cam_rng)
            cam = {"images": [jpeg_b64(v[0]) for v in views], "true": true.tolist()}
            if self.counter is not None:
                pred, dens = self.counter.count(np.stack([v[1] for v in views]))
                cam["pred"] = np.round(pred, 2).tolist()
                cam["density"] = np.round(dens / max(0.05, float(dens.max())), 3).tolist()
            self.cam_seq += 1
            cam["seq"] = self.cam_seq
            self.camera = cam
        counts = np.array(cam["pred"]) if (self.sensor == "camera" and cam and "pred" in cam) else true.astype(float)
        obs = right.observe(counts[None])
        if self.controller is not None:
            a, q = self.controller.act(obs)
            self.brain = {"obs": obs[0].astype(float).round(3).tolist(), "q": q[0].astype(float).round(3).tolist(), "action": int(a[0]),
                          "counts": np.round(counts, 1).tolist()}
        else:
            a = max_queue(right, counts[None])
        right.switch(a == 1)

    # ------------------------------------------------------------ step
    def step(self, n):
        with self.lock:
            for _ in range(n):
                if self.steps % SPD == 0:
                    self._decide()
                for env in self.envs:
                    env.demand.set_manual(self.rates()[None])
                rate = self.rates()
                arrive = self.arrivals.random((1, C.N_LANES)) < 1 - np.exp(-rate * C.DT)
                kinds = self.arrivals.choice(len(LEN), size=(1, C.N_LANES), p=PROB)
                for env in self.envs:          # exactly the same cars enter both intersections
                    env._sim_step(spawn=(arrive.copy(), kinds))
                self.steps += 1
                if self.steps % 4 == 0:
                    self.history.append([round(float(self.envs[0].t[0]), 1)] + [self._waiting(e) for e in self.envs])
                    self.history = self.history[-400:]
            return self.state()

    @staticmethod
    def _waiting(env):
        act = np.isfinite(env.s[0])
        return int(((env.v[0] < C.WAIT_SPEED) & (env.s[0] > 0) & act).sum() + env.backlog[0].sum())

    def _side(self, env):
        act = np.isfinite(env.s[0]) & (env.s[0] < 75)
        l, i = np.nonzero(act)
        cars = np.stack([l, env.s[0][act].round(2), env.kind[0][act], env.cid[0][act],
                         (env.v[0][act] < C.WAIT_SPEED).astype(int)], axis=1).tolist() if len(l) else []
        m = env.metrics()
        queue = (((env.v[0] < C.WAIT_SPEED) & (env.s[0] > 0)) & np.isfinite(env.s[0])).sum(axis=1) + env.backlog[0]
        return {"cars": cars, "lights": env.lane_light()[0].tolist(), "waiting": self._waiting(env),
                "avg_wait": round(float(m["avg_wait"][0]), 2), "served": int(m["served"][0]),
                "queue": queue.tolist(), "switches": int(m["switches"][0])}

    def state(self):
        return {"t": round(float(self.envs[0].t[0]), 1), "rates": np.round(self.rates(), 3).tolist(),
                "left": self._side(self.envs[0]), "right": self._side(self.envs[1]),
                "history": self.history[-1:] if self.history else [], "brain": self.brain,
                "cam_seq": self.cam_seq}
