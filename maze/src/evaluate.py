"""
Evaluation: the network vs. the hand-written autopilot, in exactly the same mazes.

    python -m src.evaluate

Test sets (never used in training):
    100 mazes of 4x4x4 rooms - the size the network trained on (about 410 m of corridor to the exit)
     30 mazes of 6x6x6 rooms - bigger than anything in training (generalization)

Outputs in reports/:
    eval_report.txt        the comparison table
    comparison.png         exits, crashes, flight time and speed, per pilot and maze size
    speed_profile.png      speed along one flight: where each pilot slows down for corners
    learning_stages.png    how the "brain" flew at each stage of training
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .autopilot import autopilot
from .flights import fly, summary
from .maze import Maze

COLORS = {"PPO network": "#2a9d8f", "Autopilot (rules)": "#8d99ae"}


def main():
    from .ppo import Pilot
    try:
        net = Pilot()
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    print(f"Network: {C.MODEL_PATH.name} (update {net.ckpt.get('update')}, "
          f"{net.ckpt.get('steps', 0) / 1e6:.1f}M steps)\n")
    pilots = {"PPO network": net, "Autopilot (rules)": autopilot}
    sets = {"4x4x4 (as in training)": [Maze(s, 4) for s in C.TEST_SEEDS],
            "6x6x6 (bigger, never seen)": [Maze(s, 6) for s in C.TEST_SEEDS[:30]]}

    results = {}
    lines = [f"{'maze set':28s}{'pilot':20s}{'exit':>7s}{'crash':>7s}{'time (s)':>10s}{'speed (m/s)':>13s}"]
    for set_name, mazes in sets.items():
        for p_name, pilot in pilots.items():
            s = summary(fly(pilot, mazes))
            results[(set_name, p_name)] = s
            t = f"{s['time']:.1f}" if s["time"] is not None else "-"
            lines.append(f"{set_name:28s}{p_name:20s}{s['exit'] * 100:6.0f}%{s['crash'] * 100:6.0f}%{t:>10s}{s['speed']:13.2f}")
            print(lines[-1])
    lines.append("")
    lines.append("time = average flight time of the flights that reached the exit; speed = average over all flights")
    text = "\n".join(lines)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "eval_report.txt").write_text(text + "\n")
    (C.REPORTS_DIR / "eval_results.json").write_text(json.dumps({f"{a} | {b}": v for (a, b), v in results.items()}, indent=1))

    # ---------------------------------------------------------------- comparison bars
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    x = np.arange(len(sets))
    w = 0.38
    for k, p_name in enumerate(pilots):
        vals = [results[(s, p_name)] for s in sets]
        off = (k - 0.5) * w
        axes[0].bar(x + off, [v["exit"] * 100 for v in vals], w, color=COLORS[p_name], label=p_name)
        axes[1].bar(x + off, [v["crash"] * 100 for v in vals], w, color=COLORS[p_name])
        axes[2].bar(x + off, [v["speed"] for v in vals], w, color=COLORS[p_name])
    for a, title in zip(axes, ["Reached the exit (%)", "Crashed (%)", "Average speed (m/s)"]):
        a.set_xticks(x, list(sets), fontsize=9)
        a.set_title(title)
        a.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "comparison.png", dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------------- speed profile along one flight
    maze = Maze(C.TEST_SEEDS[0], 4)
    fig, ax = plt.subplots(figsize=(12, 4))
    for p_name, pilot in pilots.items():
        rec = fly(pilot, [maze], record=True)
        sp = [f["speed"] for f in rec["frames"]]
        ax.plot(np.arange(len(sp)) * C.DT, sp, color=COLORS[p_name], label=p_name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    ax.set_title("Speed during one flight: the dips are the corners")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "speed_profile.png", dpi=110)
    plt.close(fig)

    # ---------------------------------------------------------------- learning stages
    snaps = sorted((C.MODELS_DIR / "snapshots").glob("upd_*.pt"))
    if snaps:
        mazes = [Maze(s, 4) for s in C.TEST_SEEDS[:30]]
        names, prog, exits = [], [], []
        for p in snaps + [C.MODEL_PATH]:
            pil = Pilot(p)
            s = summary(fly(pil, mazes))
            names.append("final" if p == C.MODEL_PATH else f"{pil.ckpt.get('steps', 0) / 1e3:.0f}K steps")
            prog.append(s["progress"] * 100)
            exits.append(s["exit"] * 100)
        fig, ax = plt.subplots(figsize=(9, 4))
        xx = np.arange(len(names))
        ax.bar(xx - 0.2, prog, 0.4, color="#e9c46a", label="average progress through the maze (%)")
        ax.bar(xx + 0.2, exits, 0.4, color="#2a9d8f", label="reached the exit (%)")
        ax.set_xticks(xx, names)
        ax.set_title("Learning stages (30 test mazes)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(C.REPORTS_DIR / "learning_stages.png", dpi=110)
        plt.close(fig)
    print(f"\nReports saved to {C.REPORTS_DIR}")


if __name__ == "__main__":
    main()
