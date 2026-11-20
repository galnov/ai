"""
Recordings of runs, and turning them into training data.

A recording is a small JSON file in data/recordings/. Because the game is deterministic, it only needs
the seed (the track) and the key presses (tick, action):

    {"player": "Dana", "source": "human", "seed": 81723, "ticks": 1530, "distance": 1204.5,
     "crash": "train", "presses": [[12, 3], [40, 1], ...], "ai": {"model": "combo", "distance": 2210.0},
     "date": "2026-11-19T18:30:00"}

To learn from a recording, we replay it in the Python version of the game: at every tick we take what
the network would see (the observation) and the action the player chose. That is the training data
for imitation learning: "in this situation, the player did this".
"""
import json
import time
from pathlib import Path

import numpy as np

from . import config as C
from .game import RunnerGame

CRASH_NAMES = {0: None, **C.OBSTACLE_NAMES}


def presses_to_actions(presses, ticks):
    actions = np.zeros(ticks, np.int64)
    for t, a in presses:
        if 0 <= t < ticks:
            actions[t] = a
    return actions


def actions_to_presses(actions):
    return [[int(t), int(a)] for t, a in enumerate(actions) if a != C.NOOP]


def save_recording(rec, folder=C.RECORDINGS_DIR):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rec = {**rec, "date": rec.get("date") or time.strftime("%Y-%m-%dT%H:%M:%S")}
    name = f"{rec.get('source', 'human')}_{time.strftime('%Y%m%d_%H%M%S')}_{rec['seed']}.json"
    path = folder / name
    k = 1
    while path.exists():
        path = folder / name.replace(".json", f"_{k}.json")
        k += 1
    path.write_text(json.dumps(rec, separators=(",", ":")))
    return path


def load_recordings(source="all", folder=C.RECORDINGS_DIR):
    """source: 'human', 'bot' or 'all'."""
    recs = []
    for p in sorted(Path(folder).glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if source == "all" or r.get("source", "human") == source:
            r["_file"] = p.name
            recs.append(r)
    return recs


def pick_source(requested="auto"):
    """'auto' = human recordings if there are any, otherwise the bot's practice recordings."""
    if requested != "auto":
        return requested
    return "human" if load_recordings("human") else "bot"


def replay(seed, actions, t_off=0):
    """Replays a single run. Returns (distance in meters, crash type number, ticks played)."""
    g = RunnerGame(1)
    g.reset([seed], [t_off])
    for t, a in enumerate(actions):
        g.step([a])
        if not g.alive[0]:
            return float(g.pz[0]) / 100, int(g.crash_type[0]), t + 1
    return float(g.pz[0]) / 100, 0, len(actions)


def build_dataset(recs):
    """
    Replays all the recordings together (one parallel game per recording).
    Returns a dict of arrays, one row per tick:
        obs, actions, next_obs, rewards, dones  - also usable as reinforcement-learning transitions
        run                                     - which recording the row came from
    """
    if not recs:
        raise ValueError("no recordings")
    n = len(recs)
    acts = [presses_to_actions(r["presses"], int(r["ticks"])) for r in recs]
    T = max(len(a) for a in acts)
    g = RunnerGame(n)
    g.reset([r["seed"] for r in recs])
    rows = {k: [] for k in ("obs", "actions", "next_obs", "rewards", "dones", "run")}
    obs = g.observe()
    for t in range(T):
        live = g.alive & np.array([t < len(a) for a in acts])
        if not live.any():
            break
        a = np.array([acts[e][t] if t < len(acts[e]) else 0 for e in range(n)])
        g.alive &= live                                  # recordings that ended (without a crash) stop here
        crashed, pressed = g.step(a)
        next_obs = g.observe()
        idx = np.flatnonzero(live)
        r = C.R_ALIVE + C.R_ACTION * pressed + (C.R_CRASH - C.R_ALIVE) * crashed
        rows["obs"].append(obs[idx]); rows["actions"].append(a[idx]); rows["next_obs"].append(next_obs[idx])
        rows["rewards"].append(r[idx].astype(np.float32)); rows["dones"].append(crashed[idx].astype(np.float32))
        rows["run"].append(idx)
        obs = next_obs
    return {k: np.concatenate(v) for k, v in rows.items()}


def summary(recs):
    ticks = sum(int(r["ticks"]) for r in recs)
    dist = [float(r["distance"]) for r in recs]
    return {"runs": len(recs), "minutes": ticks * C.TICK_MS / 60000,
            "mean_distance": float(np.mean(dist)) if dist else 0.0,
            "best_distance": float(np.max(dist)) if dist else 0.0}
