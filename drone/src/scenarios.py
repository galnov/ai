"""
Fixed test scenarios - for a fair comparison between the network and the PID controller.

In each scenario: 200 drones start hovering at a random point and must fly to a random target and hover there.
Conditions: calm / strong wind / gusts / heavy payload / storm (all together).
"""
import numpy as np

from . import config as C
from .drone import DroneEnv

CONDITIONS = {
    "calm": dict(wind=0.0, gust=0.0, payload=0.0),
    "wind": dict(wind=4.0, gust=0.0, payload=0.0),
    "gusts": dict(wind=1.0, gust=2.0, payload=0.0),
    "heavy": dict(wind=0.0, gust=0.0, payload=0.5),
    "storm": dict(wind=3.5, gust=2.0, payload=0.4),
}
CONDITION_NAMES = {"calm": "Calm", "wind": "Strong wind", "gusts": "Gusts", "heavy": "Heavy payload", "storm": "Storm"}


def make_env(condition, n=200, seed=123):
    """An environment with fixed (non-random) conditions, one target per drone."""
    c = CONDITIONS[condition]
    env = DroneEnv(n, seed=seed, randomize=False)
    rng = np.random.default_rng(seed + 1)
    env.wind[:] = c["wind"] * rng.choice([-1, 1], n)
    env.gust_std[:] = c["gust"]
    env.mass[:] = C.MASS + c["payload"]
    env.new_target(np.ones(n, bool))
    return env


def run(policy, condition, n=200, seed=123, seconds=C.EPISODE_SECONDS, record=0):
    """
    Runs and returns metrics:
      success   - at the end: less than 0.5 m from the target and speed below 0.5 m/s
      reach     - time until first getting within 0.5 m (seconds)
      final_err - average distance from the target in the last 3 seconds
      crashes   - percentage of crashes
      energy    - average thrust (relative to hover)
    """
    env = make_env(condition, n, seed)
    obs = env.observe()
    alive = np.ones(n, bool)
    reach = np.full(n, np.nan)
    errs, energy, paths = [], [], []
    steps = int(round(seconds / C.CONTROL_DT))
    for k in range(steps):
        a = policy(obs, env)
        obs, _, crashed, _, info = env.step(a)
        alive &= ~crashed
        d = info["dist"]
        reach = np.where(np.isnan(reach) & (d < 0.5) & alive, env.t, reach)
        if env.t[0] > seconds - 3:
            errs.append(np.where(alive, d, np.nan))
        energy.append(((env.t1 + env.t2) / (2 * C.T_HOVER))[alive].mean() if alive.any() else np.nan)
        if record:
            paths.append(np.stack([env.x[:record], env.y[:record], env.th[:record],
                                   env.t1[:record], env.t2[:record]], axis=1))
        env.done[:] = False
    speed = np.hypot(env.vx, env.vy)
    success = alive & (d < 0.5) & (speed < 0.5)
    out = {
        "success": float(success.mean()),
        "crashes": float((~alive).mean()),
        "reach": float(np.nanmean(reach)) if np.isfinite(reach).any() else float("nan"),
        "final_err": float(np.nanmean(np.array(errs))) if np.isfinite(np.array(errs)).any() else float("nan"),
        "energy": float(np.nanmean(energy)),
    }
    if record:
        out["paths"] = np.array(paths)           # (steps, record, 5)
        out["targets"] = np.stack([env.tx[:record], env.ty[:record]], axis=1)
    return out
