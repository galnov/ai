"""
Run a policy (driver) on a track and measure its performance. Used both in training (validation) and in evaluation.
policy_fn takes an observations array (N,10) and returns actions (N,).
"""
import numpy as np

from . import config as C
from .env import CarEnv
from .track import Track, random_track


def make_tracks(n, seed, hard=False):
    rng = np.random.default_rng(seed)
    if hard:
        return [random_track(rng, width=78.0, kind="polygon") for _ in range(n)]
    return [random_track(rng) for _ in range(n)]


def preset_tracks():
    return [Track.load(p) for p in sorted(C.TRACKS_DIR.glob("*.json"))]


def run_policy(policy_fn, track, steps=C.MAX_EPISODE_STEPS, record=False):
    env = CarEnv(track, n_cars=1, rng=np.random.default_rng(0), random_start=False)
    obs = env.reset()
    path, actions = [], []
    crashed = False
    for _ in range(steps):
        a = policy_fn(obs)
        obs, _, _, _, info = env.step(a)
        actions.append(int(a[0]))
        if record:
            path.append((env.x[0], env.y[0], env.speed[0]))
        if env.done[0]:
            crashed = bool(info["crashed"][0] or info["stuck"][0])
            break
    sim_time = env.steps[0] * C.DT
    return {
        "laps_progress": float(env.progress[0] / track.length),   # how many laps (including partial)
        "laps": int(env.laps[0]),
        "crashed": crashed,
        "best_lap": float(env.best_lap_time[0]),
        "avg_speed": float(env.progress[0] / max(sim_time, 1e-9)),
        "path": np.array(path),
        "actions": np.array(actions),
    }


def validation_score(policy_fn, tracks, steps):
    """Average laps completed in `steps` steps (a crash = early stop = a low score)."""
    return float(np.mean([run_policy(policy_fn, t, steps)["laps_progress"] for t in tracks]))
