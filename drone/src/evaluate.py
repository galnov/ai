"""
Evaluation: the network (PPO) vs. a PID controller, in exactly the same scenarios.

    python -m src.evaluate

In each condition: 200 drones start hovering at a random point and fly to a random target.
Conditions: calm / strong wind / gusts / heavy payload / storm.

Outputs in reports/:
    eval_report.txt         comparison table
    comparison.png          success, crashes, time to reach, accuracy
    trajectories.png        flight paths of PPO and PID
    learning_stages.png     how well the "brain" flies at each stage of training
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .pid import PID
from .ppo import DronePolicy
from .scenarios import CONDITIONS, run

COLORS = {"PPO": "#2a9d8f", "PID": "#e76f51"}


def main():
    ppo = DronePolicy()
    print(f"Network: {C.MODEL_PATH.name} (update {ppo.ckpt.get('update')}, {ppo.ckpt.get('steps', 0) / 1e6:.1f}M steps)\n")
    results = {}
    for c in CONDITIONS:
        results[c] = {"PPO": run(ppo, c, record=6), "PID": run(PID(200), c, record=6)}

    keys = [("success", "success %", 100), ("crashes", "crash %", 100), ("reach", "reach (s)", 1),
            ("final_err", "hover err (m)", 1), ("energy", "thrust/hover", 1)]
    head = f"{'condition':10s} {'ctrl':4s} " + " ".join(f"{k[1]:>13s}" for k in keys)
    lines = ["200 drones per condition, 10 s each, start hovering at a random point.",
             "success = within 0.5 m of the target and slower than 0.5 m/s at the end.", "", head, "-" * len(head)]
    for c in CONDITIONS:
        for ctrl in ("PPO", "PID"):
            r = results[c][ctrl]
            lines.append(f"{c:10s} {ctrl:4s} " + " ".join(f"{r[k] * s:13.2f}" for k, _, s in keys))
    text = "\n".join(lines)
    print(text)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "eval_report.txt").write_text(text + "\n")

    # comparison
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    x = np.arange(len(CONDITIONS))
    for ax, (k, label, s) in zip(axes, keys[:4]):
        for i, ctrl in enumerate(("PPO", "PID")):
            ax.bar(x + (i - 0.5) * 0.38, [results[c][ctrl][k] * s for c in CONDITIONS], 0.38,
                   color=COLORS[ctrl], label=ctrl)
        ax.set_xticks(x, list(CONDITIONS), rotation=20)
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "comparison.png", dpi=110)
    plt.close(fig)

    # trajectories
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(4 * len(CONDITIONS), 6.5))
    for j, c in enumerate(CONDITIONS):
        for i, ctrl in enumerate(("PPO", "PID")):
            ax = axes[i, j]
            r = results[c][ctrl]
            for d in range(r["paths"].shape[1]):
                p = r["paths"][:, d]
                ax.plot(p[:, 0], p[:, 1], color=COLORS[ctrl], lw=1.2, alpha=0.85)
                ax.plot(*p[0, :2], "o", color="#555", ms=3)
            ax.plot(r["targets"][:, 0], r["targets"][:, 1], "x", color="k", ms=7)
            ax.set_xlim(0, C.WORLD_W)
            ax.set_ylim(0, C.WORLD_H)
            ax.set_aspect("equal")
            ax.set_title(f"{ctrl} - {c}", fontsize=10)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "trajectories.png", dpi=100)
    plt.close(fig)

    # learning stages
    snaps = sorted((C.MODELS_DIR / "snapshots").glob("upd_*.pt")) + [C.MODEL_PATH]
    if len(snaps) > 1:
        labels, succ, crash = [], [], []
        for p in snaps:
            pol = DronePolicy(p)
            r = [run(pol, c, n=100, seed=7) for c in ("calm", "storm")]
            labels.append(f"{pol.ckpt.get('steps', 0) / 1e6:.2f}M steps" + ("\n(best)" if p == C.MODEL_PATH else ""))
            succ.append(np.mean([q["success"] for q in r]) * 100)
            crash.append(np.mean([q["crashes"] for q in r]) * 100)
        fig, ax = plt.subplots(figsize=(9, 4))
        xx = np.arange(len(labels))
        ax.bar(xx - 0.2, succ, 0.4, color="#2a9d8f", label="success %")
        ax.bar(xx + 0.2, crash, 0.4, color="#e63946", label="crash %")
        ax.set_xticks(xx, labels)
        ax.set_ylim(0, 105)
        ax.legend()
        ax.set_title("Learning stages (calm + storm)")
        fig.tight_layout()
        fig.savefig(C.REPORTS_DIR / "learning_stages.png", dpi=110)
        plt.close(fig)
    print(f"\nReports saved to {C.REPORTS_DIR}")


if __name__ == "__main__":
    main()
