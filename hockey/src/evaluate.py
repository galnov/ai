"""
How good is the trained player?

    python -m src.evaluate

Plays 1,000 points in each match-up and prints the share of goals each side scored:
  * every learning stage and the final network against the hand-written bot (at full speed and at 70%)
  * the final network against every earlier stage
A share above 50% means the first player scores more often. "Replays" are points where nobody scored in 12 s.
"""
import numpy as np

from . import config as C
from .bot import bot_actions
from .physics import play_points
from .ppo import load

POINTS = 1000


def line(name, res):
    g0, g1, to = res
    share = g0 / max(1, g0 + g1) * 100
    print(f"  {name:<38} {g0:5d} : {g1:<5d}  {share:5.1f}% of the goals   {to:4d} replays")


def main():
    players = []
    for path in sorted(C.SNAPSHOT_DIR.glob("upd_*.pt")):
        m, ck = load(path)
        players.append((f"after {ck['steps'] / 1e6:.2f}M decisions", m))
    final, ck = load(C.MODEL_PATH)
    players.append(("final network", final))

    print(f"\nAgainst the bot ({POINTS} points each)")
    for name, m in players:
        line(f"{name} vs bot", play_points(m.play, bot_actions, POINTS))
        line(f"{name} vs bot at 70% speed", play_points(m.play, lambda o: bot_actions(o, 0.7), POINTS))
    print("\nThe final network against its earlier stages")
    for name, m in players[:-1]:
        line(f"final vs {name}", play_points(final.play, m.play, POINTS))
    print("\nThe bot against itself (a sanity check: should be close to 50%)")
    line("bot vs bot", play_points(bot_actions, bot_actions, POINTS))


if __name__ == "__main__":
    main()
