"""
The simulation that runs in the app. Four modes:
  free     - free flight: click on the screen and the drone flies there (a PID drone can be added for comparison)
  delivery - delivery route: mark points, and the drone passes between them among the buildings
  stages   - learning stages: drones with a "brain" from different points in training
  swarm    - swarm show: 40 drones form shapes
"""
import threading

import numpy as np

from src import config as C
from src.drone import DroneEnv
from src.pid import PID

BUILDINGS = [(1.2, 3.2, 3.0), (4.6, 1.8, 5.2), (8.4, 2.2, 2.4), (11.6, 1.6, 6.0), (14.2, 2.4, 3.6), (17.6, 1.4, 4.4)]
# (left x, width, height)  - delivery mode only
SWARM_N = 40


def shape_points(name, n=SWARM_N):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cx, cy = C.WORLD_W / 2, C.WORLD_H / 2 + 0.5
    if name == "circle":
        return np.stack([cx + 4.2 * np.cos(t), cy + 3.6 * np.sin(t)], 1)
    if name == "heart":
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        return np.stack([cx + x * 0.28, cy + y * 0.26 + 0.6], 1)
    if name == "star":
        tt = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        rr = np.where(np.arange(10) % 2 == 0, 4.4, 1.9)
        corners = np.stack([cx + rr * np.sin(tt), cy + rr * np.cos(tt)], 1)
        pts = []
        per = n // 10
        for i in range(10):
            a, b = corners[i], corners[(i + 1) % 10]
            for f in np.linspace(0, 1, per, endpoint=False):
                pts.append(a + (b - a) * f)
        return np.array(pts[:n])
    if name == "wave":
        x = np.linspace(2.5, C.WORLD_W - 2.5, n)
        return np.stack([x, cy + 2.6 * np.sin(x / 2.0)], 1)
    if name == "grid":
        cols = 10
        xs = cx + (np.arange(n) % cols - (cols - 1) / 2) * 1.3
        ys = cy + 2.4 - (np.arange(n) // cols) * 1.6
        return np.stack([xs, ys], 1)
    raise ValueError(name)


def assign(drones, points):
    """Greedy assignment: each point to the drone closest to it (to avoid crossings)."""
    left = list(range(len(drones)))
    out = np.zeros(len(points), int)
    d = np.linalg.norm(drones[:, None] - points[None], axis=2)
    for _ in range(len(points)):
        i, j = np.unravel_index(np.argmin(d), d.shape)
        out[j] = i
        d[i, :] = np.inf
        d[:, j] = np.inf
    res = np.zeros_like(points)
    res[out] = points
    return res


class World:
    def __init__(self, policies, meta):
        """policies: {name: function(obs)->actions}.  meta: a list of stages [{name,label}]."""
        self.policies = policies
        self.meta = meta
        self.lock = threading.Lock()
        self.wind, self.gust, self.payload = 0.0, 0.0, 0.0
        self.compare_pid = True
        self.set_mode("free")

    # ------------------------------------------------------------ modes
    def set_mode(self, mode):
        with self.lock:
            self.mode = mode
            self.events = []
            self.deliveries = 0
            self.route = [] if mode != "delivery" else [(3.0, 8.0), (9.6, 4.6), (16.0, 9.0), (19.0, 7.8)]
            self.route_i = None
            best = "best" if "best" in self.policies else None
            if mode == "free":
                self.drones = [{"ctrl": best or "pid", "label": "PPO network", "color": "#34d1bf"}]
                if self.compare_pid:
                    self.drones.append({"ctrl": "pid", "label": "PID", "color": "#ff8a5b"})
                starts = [(4.0, 3.0), (4.0, 4.4)][:len(self.drones)]
                target = (14.0, 7.0)
            elif mode == "delivery":
                self.drones = [{"ctrl": best or "pid", "label": "Delivery", "color": "#34d1bf"}]
                starts, target = [(0.6, 7.0)], self.route[0]
                self.route_i = [0]
            elif mode == "stages":
                pal = ["#8d99ae", "#c792ea", "#ffb020", "#ff6b6b", "#34d1bf"]
                self.drones = [{"ctrl": m["name"], "label": m["label"], "color": pal[i % len(pal)]}
                               for i, m in enumerate(self.meta)]
                starts = [(3.0, 3.0 + 1.3 * i) for i in range(len(self.drones))]
                target = (15.0, 7.0)
            else:  # swarm
                self.drones = [{"ctrl": best or "pid", "label": "", "color": "#34d1bf"} for _ in range(SWARM_N)]
                starts = [(x, 1.2) for x in np.linspace(1.0, C.WORLD_W - 1.0, SWARM_N)]
                target = None
            n = len(self.drones)
            self.env = DroneEnv(n, seed=1, randomize=False)
            self.starts = np.array(starts, float)
            self.env.x[:], self.env.y[:] = self.starts[:, 0], self.starts[:, 1]
            self.pid = PID(n)
            self._apply_conditions()
            if mode == "swarm":
                self.shape = "heart"
                pts = assign(self.starts, shape_points(self.shape))
                self.env.new_target(np.ones(n, bool), pts[:, 0], pts[:, 1])
            else:
                self.env.new_target(np.ones(n, bool), target[0], target[1])
            self.trails = [[] for _ in range(n)]

    def _apply_conditions(self):
        self.env.wind[:] = self.wind
        self.env.gust_std[:] = self.gust
        self.env.mass[:] = C.MASS + self.payload

    def configure(self, wind=None, gust=None, payload=None, compare_pid=None):
        with self.lock:
            if wind is not None:
                self.wind = float(np.clip(wind, -7, 7))
            if gust is not None:
                self.gust = float(np.clip(gust, 0, 3))
            if payload is not None:
                self.payload = float(np.clip(payload, 0, 0.6))
            self._apply_conditions()
        if compare_pid is not None and compare_pid != self.compare_pid:
            self.compare_pid = bool(compare_pid)
            if self.mode == "free":
                self.set_mode("free")

    def click(self, x, y):
        """A click on the screen: a new target / a point on the route / the next shape."""
        x = float(np.clip(x, 0.8, C.WORLD_W - 0.8))
        y = float(np.clip(y, 1.5, C.WORLD_H - 0.8))
        with self.lock:
            n = self.env.n
            if self.mode in ("free", "stages"):
                self.env.new_target(np.ones(n, bool), x, y)
            elif self.mode == "delivery":
                if len(self.route) >= 8:
                    self.route = []
                self.route.append((x, y))
                if len(self.route) == 1:
                    self.route_i = [0]
                    self.env.new_target(np.ones(n, bool), x, y)

    def set_shape(self, name):
        with self.lock:
            if self.mode != "swarm":
                return
            self.shape = name
            pos = np.stack([self.env.x, self.env.y], 1)
            pts = assign(pos, shape_points(name))
            self.env.new_target(np.ones(self.env.n, bool), pts[:, 0], pts[:, 1])

    def clear_route(self):
        with self.lock:
            self.route = []

    # ------------------------------------------------------------ step
    def _actions(self, obs):
        n = self.env.n
        act = np.zeros((n, 2))
        groups = {}
        for i, d in enumerate(self.drones):
            groups.setdefault(d["ctrl"], []).append(i)
        for ctrl, idx in groups.items():
            if ctrl == "pid":
                act[idx] = self.pid(obs, self.env)[idx]
            else:
                act[idx] = self.policies[ctrl](obs[idx])
        return act

    def _crashed_into_building(self):
        if self.mode != "delivery":
            return np.zeros(self.env.n, bool)
        hit = np.zeros(self.env.n, bool)
        for bx, bw, bh in BUILDINGS:
            hit |= (self.env.x > bx - 0.25) & (self.env.x < bx + bw + 0.25) & (self.env.y < bh + 0.1)
        return hit

    def step(self, n_steps):
        with self.lock:
            env = self.env
            for _ in range(n_steps):
                obs = env.observe()
                _, _, crashed, _, _ = env.step(np.clip(self._actions(obs), -1, 1))
                crashed = crashed | self._crashed_into_building()
                for i in np.flatnonzero(crashed):
                    self.events.append({"type": "crash", "x": float(env.x[i]), "y": float(env.y[i])})
                    self._respawn(i)
                if self.mode == "delivery" and self.route:
                    d = np.hypot(env.tx[0] - env.x[0], env.ty[0] - env.y[0])
                    if d < 0.6:
                        self.route_i[0] = (self.route_i[0] + 1) % len(self.route)
                        if self.route_i[0] == 0:
                            self.deliveries += 1
                            self.events.append({"type": "delivered", "x": float(env.x[0]), "y": float(env.y[0])})
                        tx, ty = self.route[self.route_i[0]]
                        env.new_target(np.array([True]), tx, ty)
                env.done[:] = False
                for i in range(env.n):
                    tr = self.trails[i]
                    tr.append((round(float(env.x[i]), 2), round(float(env.y[i]), 2)))
                    if len(tr) > 90:
                        del tr[0]
            return self.state()

    def _respawn(self, i):
        env = self.env
        env.x[i], env.y[i] = self.starts[i]
        env.vx[i] = env.vy[i] = env.th[i] = env.om[i] = 0
        env.t1[i] = env.t2[i] = C.T_HOVER
        env.ix[i] = env.iy[i] = 0
        self.pid.ix[i] = self.pid.iy[i] = 0
        self.trails[i] = []
        if self.mode == "delivery" and self.route:
            self.route_i[0] = 0
            env.new_target(np.array([True]), *self.route[0])

    def state(self):
        env = self.env
        events, self.events = self.events, []
        r = lambda a: np.round(a, 3).tolist()
        return {
            "mode": self.mode,
            "drones": {"x": r(env.x), "y": r(env.y), "th": r(env.th),
                       "t1": r(env.t1 / C.T_MAX), "t2": r(env.t2 / C.T_MAX),
                       "tx": r(env.tx), "ty": r(env.ty)},
            "meta": [{"label": d["label"], "color": d["color"], "pid": d["ctrl"] == "pid"} for d in self.drones],
            "trails": self.trails if self.mode != "swarm" else [],
            "air": float(self.wind + env.gust.mean()) if env.n else self.wind,
            "events": events,
            "route": self.route, "route_i": self.route_i[0] if self.route_i else None,
            "deliveries": self.deliveries,
            "buildings": BUILDINGS if self.mode == "delivery" else [],
        }

    def value_map(self, policy_value, nx=40, ny=24):
        """The Critic's value map V(s): how "good" it is to be at each point (drone standing still and level)."""
        with self.lock:
            tx, ty = float(self.env.tx[0]), float(self.env.ty[0])
        xs = np.linspace(0.25, C.WORLD_W - 0.25, nx)
        ys = np.linspace(0.25, C.WORLD_H - 0.25, ny)
        X, Y = np.meshgrid(xs, ys)
        e = DroneEnv(X.size, seed=0, randomize=False)
        e.x[:], e.y[:] = X.ravel(), Y.ravel()
        e.vx[:] = e.vy[:] = e.th[:] = e.om[:] = 0
        e.wind[:] = self.wind
        e.new_target(np.ones(X.size, bool), tx, ty)
        v = policy_value(e.observe()).reshape(ny, nx)
        return {"nx": nx, "ny": ny, "v": np.round((v - v.min()) / (np.ptp(v) + 1e-9), 3).tolist()}
