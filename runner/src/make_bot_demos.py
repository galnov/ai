"""
Practice recordings from the rule-based bot, to test the imitation pipeline before people play.

    python -m src.make_bot_demos              # 40 runs
    python -m src.make_bot_demos --runs 20

The bot plays with "human-like" imperfection: a short reaction delay, missed key presses and a few
unnecessary ones. The recordings are saved as source "bot", so they never mix with real human runs:
training uses human recordings whenever there are any (see train_bc.py --source).

For the real experiment: play the app (every run is recorded), then delete the bot_*.json files.
"""
import argparse

import numpy as np

from . import config as C
from .bot import RuleBot
from .game import RunnerGame
from .recordings import CRASH_NAMES, actions_to_presses, save_recording


def main():
    ap = argparse.ArgumentParser(description="Create practice recordings with the rule-based bot")
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--max-ticks", type=int, default=3600, help="cut runs after this many ticks (3 minutes)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.runs
    seeds = rng.integers(100_000, 1_000_000, n)
    skills = rng.uniform(0.75, 0.92, n)                 # some "players" are better than others
    g = RunnerGame(n)
    g.reset(seeds)
    bots = [RuleBot(1, s, seed=args.seed * 1000 + i) for i, s in enumerate(skills)]
    actions = np.zeros((n, args.max_ticks), np.int64)
    ticks = np.full(n, args.max_ticks)
    for t in range(args.max_ticks):
        if not g.alive.any():
            break
        for e in np.flatnonzero(g.alive):
            actions[e, t] = bots[e]._humanize(0, bots[e]._decide(g, e, int(g.speed()[e])))
        was = g.alive.copy()
        g.step(actions[:, t])
        ticks[was & ~g.alive] = t + 1

    for e in range(n):
        save_recording({
            "player": f"bot (skill {skills[e]:.2f})", "source": "bot", "seed": int(seeds[e]),
            "ticks": int(ticks[e]), "distance": float(g.pz[e]) / 100, "crash": CRASH_NAMES[int(g.crash_type[e])],
            "presses": actions_to_presses(actions[e, :ticks[e]]),
        })
    minutes = ticks.sum() * C.TICK_MS / 60000
    print(f"Saved {n} practice runs ({minutes:.0f} minutes of play) to {C.RECORDINGS_DIR}")
    print(f"Distance: mean {g.pz.mean() / 100:.0f} m, best {g.pz.max() / 100:.0f} m")
    print("Next step:  python -m src.train_bc")


if __name__ == "__main__":
    main()
