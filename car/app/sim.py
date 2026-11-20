"""
The simulation that runs on the server for the app: a track, cars (AI drivers and/or a human driver), step by step.
"""
import threading

import numpy as np

from src import config as C
from src.env import CarEnv
from src.track import random_track

N_BRAIN = 14  # how many neurons from each hidden layer we draw


def sample_observations(driver, steps=300):
    """A short drive of the driver on a random track - to know which neurons are usually active."""
    env = CarEnv(random_track(np.random.default_rng(5)), n_cars=4, rng=np.random.default_rng(5))
    obs, out = env.reset(), []
    for _ in range(steps):
        out.append(obs)
        obs, *_ = env.step(driver.act(obs)[0])
        obs = env.auto_reset()
    return np.concatenate(out)


def pick_neurons(driver, k=N_BRAIN):
    """
    Picks the "important" neurons from each hidden layer: usually active (high mean activation)
    and with a large influence on the next layer (large outgoing weights).
    """
    w1, w2, w3 = driver.weights()                     # (H,in), (H,H), (A,H)
    _, _, h1, h2 = driver.act(sample_observations(driver))
    score2 = h2.mean(axis=0) * np.abs(w3).sum(axis=0)
    h2_idx = np.sort(np.argsort(-score2)[:k])
    score1 = h1.mean(axis=0) * np.abs(w2[h2_idx]).sum(axis=0)
    h1_idx = np.sort(np.argsort(-score1)[:k])
    return {
        "h1_idx": h1_idx, "h2_idx": h2_idx,
        "w1": w1[h1_idx].round(3).tolist(),                       # (k, in)
        "w2": w2[np.ix_(h2_idx, h1_idx)].round(3).tolist(),       # (k, k)
        "w3": w3[:, h2_idx].round(3).tolist(),                    # (A, k)
    }


class Sim:
    def __init__(self, drivers):
        """drivers: {name: Driver}  (may be empty - then only human driving)."""
        self.drivers = drivers
        self.brain_cache = {name: pick_neurons(d) for name, d in drivers.items()}
        self.lock = threading.Lock()
        self.track = None
        self.env = None
        self.cars = []

    def set_track(self, track):
        with self.lock:
            self.track = track
            self.env = None
            self.cars = []

    def start(self, cars, spread=False):
        """
        cars: a list of {"kind": "ai"|"human", "driver": name, "label": ..., "color": ...}
        All cars start at the start line, side by side.
        """
        with self.lock:
            n = len(cars)
            self.cars = cars
            if n == 0:
                self.env = None
                return
            self.env = CarEnv(self.track, n_cars=n, rng=np.random.default_rng(0), random_start=False)
            self.env.reset()
            t = self.track
            if spread:   # watch mode: the cars are spread out along the track
                idx = (np.arange(n) * t.n // n) % t.n
                e = self.env
                e.idx[:] = idx
                e.x[:], e.y[:] = t.center[idx, 0], t.center[idx, 1]
                e.heading[:] = np.arctan2(t.tangent[idx, 1], t.tangent[idx, 0])
            else:        # race: all on the start line, side by side
                normal = np.array([-t.tangent[0, 1], t.tangent[0, 0]])
                span = t.width / 2 - C.CAR_RADIUS - 6
                offs = np.linspace(-span, span, n) if n > 1 else np.zeros(1)
                self.env.x += offs * normal[0]
                self.env.y += offs * normal[1]
            self.env.observe()

    def step(self, n_steps, keys):
        with self.lock:
            env = self.env
            if env is None:
                return {"cars": [], "t": 0}
            crash_events = []
            brain = None
            for _ in range(max(0, n_steps)):
                obs = env.observe()
                steer = np.zeros(env.n)
                throttle = np.zeros(env.n)
                for name in {c.get("driver") for c in self.cars if c["kind"] == "ai"}:
                    rows = [i for i, c in enumerate(self.cars) if c["kind"] == "ai" and c["driver"] == name]
                    actions, q, h1, h2 = self.drivers[name].act(obs[rows])
                    act = np.array(C.ACTIONS)[actions]
                    steer[rows], throttle[rows] = act[:, 0], act[:, 1]
                    f = self.focus()
                    if f in rows:
                        r = rows.index(f)
                        bc = self.brain_cache[name]
                        brain = {"driver": name, "obs": obs[f].astype(float).round(3).tolist(),
                                 "q": q[r].astype(float).round(3).tolist(), "action": int(actions[r]),
                                 "h1": h1[r][bc["h1_idx"]].astype(float).round(3).tolist(),
                                 "h2": h2[r][bc["h2_idx"]].astype(float).round(3).tolist()}
                for i, c in enumerate(self.cars):
                    if c["kind"] == "human":
                        steer[i] = (1 if keys.get("right") else 0) - (1 if keys.get("left") else 0)
                        throttle[i] = (1 if keys.get("up") else 0) - (1 if keys.get("down") else 0)
                _, _, _, _, info = env.step(None, controls=(steer, throttle))
                is_ai = np.array([c["kind"] == "ai" for c in self.cars])
                bad = info["crashed"] | (info["stuck"] & is_ai)   # a human driver may stand still
                for i in np.flatnonzero(bad):
                    crash_events.append({"car": int(i), "x": float(env.x[i]), "y": float(env.y[i]),
                                         "stuck": bool(info["stuck"][i] and not info["crashed"][i])})
                env.respawn(bad)
                env.done[:] = False
            env.observe()
            cur_lap = (env.steps - env.lap_start) * C.DT
            return {
                "t": float(env.steps.max() * C.DT),
                "cars": [{
                    "x": round(float(env.x[i]), 1), "y": round(float(env.y[i]), 1),
                    "h": round(float(env.heading[i]), 4), "v": round(float(env.speed[i] / C.MAX_SPEED), 3),
                    "rays": (env.rays[i] / C.RAY_LENGTH).round(3).tolist(),
                    "laps": int(env.laps[i]),
                    "progress": round(float(env.progress[i] / self.track.length), 4),
                    "lap_time": round(float(cur_lap[i]), 2),
                    "last_lap": None if np.isnan(env.last_lap_time[i]) else round(float(env.last_lap_time[i]), 2),
                    "best_lap": None if np.isnan(env.best_lap_time[i]) else round(float(env.best_lap_time[i]), 2),
                } for i in range(env.n)],
                "crashes": crash_events,
                "brain": brain,
            }

    def focus(self):
        """The car whose "brain" is shown: the trained driver if there is one, otherwise the first AI."""
        ai = [i for i, c in enumerate(self.cars) if c["kind"] == "ai"]
        best = [i for i in ai if self.cars[i]["driver"] == "best"]
        return (best or ai or [-1])[0]
