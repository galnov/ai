"""
Flying fixed mazes with any pilot - used by training (validation), the evaluation and the app.

A pilot is any function obs (n, 93) -> actions (n, 3): the network, the autopilot, or a snapshot.
"""
import numpy as np

from . import config as C
from .maze import Maze
from .ship import ShipEnv


def fly(pilot, mazes, time_limit=None, record=False):
    """
    Flies every maze from its start until the exit, a crash or the time limit.
    Returns arrays: reached, crashed, time (s), progress (0-1 of the way), mean speed (m/s).
    With record=True, also the frames of the first maze (for the app).
    """
    if time_limit is None:                    # bigger mazes get more time
        time_limit = max(C.TIME_LIMIT, 0.45 * max(m.path_length() for m in mazes))
    env = ShipEnv(len(mazes), sizes=tuple(sorted({m.rooms for m in mazes})), time_limit=time_limit)
    obs = env.set_mazes(mazes)
    n = len(mazes)
    reached, crashed = np.zeros(n, bool), np.zeros(n, bool)
    t_end = np.full(n, time_limit)
    frames, actions = ([env.frame(0)], []) if record else (None, None)
    recording = record
    dist = np.zeros(n)
    while not env.done.all():
        a = np.clip(np.asarray(pilot(obs), float), -1, 1)
        before = env.pos.copy()
        obs, r, term, trunc, info = env.step(a)
        dist += np.linalg.norm(env.pos - before, axis=1)          # ships that already finished don't move
        reached |= info["reached"]
        crashed |= info["crashed"]
        ended = term | trunc
        t_end = np.where(ended, info["time"], t_end)
        if recording:                                              # frames of the first maze, until it ends
            actions.append(a[0].round(3).tolist())
            fr = env.frame(0)
            if env.done[0]:
                fr["end"] = "exit" if reached[0] else ("crash" if crashed[0] else "timeout")
                recording = False
            frames.append(fr)
    out = {"reached": reached, "crashed": crashed, "time": t_end, "progress": np.clip(info["progress"], 0, 1),
           "speed": dist / np.maximum(t_end, 1e-6)}
    if record:
        out["frames"], out["actions"] = frames, actions
    return out


def summary(res):
    r = res["reached"]
    return {"exit": float(r.mean()), "crash": float(res["crashed"].mean()),
            "time": float(res["time"][r].mean()) if r.any() else None,
            "progress": float(res["progress"].mean()), "speed": float(res["speed"].mean())}


def validation_mazes():
    return [Maze(s, C.MAZE_ROOMS) for s in C.VAL_SEEDS]
