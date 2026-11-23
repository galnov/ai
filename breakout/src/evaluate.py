"""
How good is the trained player?

    python -m src.evaluate

Every learning stage and the final network play 64 full games (3 balls each) without randomness.
For comparison: a random player, and a scripted player that "cheats" - it reads the ball's position
straight from the game's memory and moves toward it (the network only gets pixels).
"""
import numpy as np

from . import config as C
from .game import Breakout, tracker
from .ppo import load
from .train import play_games

GAMES = 64


def scripted(policy, games=GAMES, seed=123):
    env = Breakout(games, seed)
    rng = np.random.default_rng(0)
    scores, finished = np.full(games, -1), np.zeros(games, bool)
    while not finished.all():
        before = env.score.copy()
        _, _, done = env.step(policy(env, rng))
        new = done & ~finished
        scores[new] = before[new]
        finished |= done
    return scores


def line(name, sc):
    walls = C.BRICK_ROWS * C.BRICK_COLS
    print(f"  {name:<34} {sc.mean():7.1f} bricks per game   (min {sc.min():4d}, max {sc.max():4d})   "
          f"walls cleared: {np.mean(sc // walls):.2f}")


def main():
    print(f"\n{GAMES} games each, 3 balls per game, {C.BRICK_ROWS * C.BRICK_COLS} bricks per wall\n")
    line("random", scripted(lambda e, r: r.integers(0, 3, e.n)))
    for path in sorted(C.SNAPSHOT_DIR.glob("steps_*.pt")):
        m, ck = load(path)
        line(f"network after {ck['steps'] / 1e6:.1f}M decisions", play_games(m, GAMES))
    m, ck = load(C.MODEL_PATH)
    line(f"final network ({ck['steps'] / 1e6:.1f}M)", play_games(m, GAMES))
    line("scripted: follows the ball's x", scripted(lambda e, r: tracker(e)))


if __name__ == "__main__":
    main()
