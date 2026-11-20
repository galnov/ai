"""
Evaluate the trained driver on tracks it has never seen.

    python -m src.evaluate

Outputs in reports/:
    eval_report.txt         - results table
    trajectories.png        - the path the car drove, colored by speed
    actions.png             - which actions the driver uses
    learning_stages.png     - comparison between the learning stages (snapshots) and the final network
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .agent import Driver
from .validation import make_tracks, preset_tracks, run_policy


def summarize(name, results):
    n = len(results)
    full = sum(r["laps"] >= 1 and not r["crashed"] for r in results)
    crashes = sum(r["crashed"] for r in results)
    laps = np.mean([r["laps_progress"] for r in results])
    best = [r["best_lap"] for r in results if np.isfinite(r["best_lap"])]
    speed = np.mean([r["avg_speed"] for r in results])
    return (f"{name:22s} {n:5d} {full:6d}/{n:<3d} {crashes:8d} {laps:8.2f} "
            f"{(np.mean(best) if best else float('nan')):10.1f} {speed / C.MAX_SPEED * 100:9.0f}%")


def plot_trajectories(tracks, results, labels, path):
    k = min(6, len(tracks))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, t, r, name in zip(axes.flat, tracks[:k], results[:k], labels):
        for wall in (t.left, t.right):
            ax.fill(wall[:, 0], wall[:, 1], fill=False, color="#555", lw=1)
        p = r["path"]
        if len(p):
            sc = ax.scatter(p[:, 0], p[:, 1], c=p[:, 2] / C.MAX_SPEED * 100, cmap="turbo", s=2, vmin=0, vmax=100)
        status = "CRASH" if r["crashed"] else f"{r['laps']} laps, best {r['best_lap']:.1f}s"
        ax.set_title(f"{name}: {status}", fontsize=9)
        ax.set_xlim(0, C.WORLD_W)
        ax.set_ylim(C.WORLD_H, 0)
        ax.set_aspect("equal")
        ax.axis("off")
    for ax in list(axes.flat)[k:]:
        ax.axis("off")
    fig.colorbar(sc, ax=axes, fraction=0.02, label="speed (% of max)")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_actions(results, path):
    acts = np.concatenate([r["actions"] for r in results])
    counts = np.bincount(acts, minlength=C.N_ACTIONS) / len(acts) * 100
    names = ["gas", "gas+left", "gas+right", "left", "right", "brake"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, counts, color=["#2f9e44", "#1c7ed6", "#1c7ed6", "#74c0fc", "#74c0fc", "#e03131"])
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:.0f}%", ha="center", va="bottom")
    ax.set_ylabel("% of decisions")
    ax.set_title("How often the driver uses each action")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_stages(tracks, path):
    paths = sorted((C.MODELS_DIR / "snapshots").glob("step_*.pt")) + [C.MODEL_PATH]
    labels, scores, crash = [], [], []
    for p in paths:
        d = Driver(p)
        res = [run_policy(lambda o: d.act(o)[0], t, C.TRAIN["eval_steps"]) for t in tracks]
        step = d.ckpt.get("step", 0)
        labels.append(f"step {step:,}" + ("\n(best)" if p == C.MODEL_PATH else ""))
        scores.append(np.mean([r["laps_progress"] for r in res]))
        crash.append(np.mean([r["crashed"] for r in res]) * 100)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(labels, scores, color="#4c6ef5")
    for b, s, c in zip(bars, scores, crash):
        ax.text(b.get_x() + b.get_width() / 2, s, f"{s:.2f} laps\n{c:.0f}% crash", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("laps in 30 s (test tracks)")
    ax.set_title("Learning stages")
    ax.set_ylim(0, max(scores) * 1.3 + 0.1)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    driver = Driver(C.MODEL_PATH)
    policy = lambda o: driver.act(o)[0]
    print(f"Driver: {C.MODEL_PATH.name} (step {driver.ckpt.get('step')}, validation {driver.ckpt.get('val_score', 0):.2f})\n")

    groups = {
        "test (random)": make_tracks(30, C.TEST_SEED),
        "test (hard: narrow)": make_tracks(15, C.TEST_SEED + 1, hard=True),
        "preset tracks": preset_tracks(),
    }
    header = (f"{'tracks':22s} {'count':>5s} {'no-crash':>10s} {'crashes':>8s} {'laps':>8s} "
              f"{'lap time':>10s} {'pace':>10s}")
    lines = ["Each run: up to 75 seconds, starting from standstill.",
             "pace = distance along the center line per second, as % of the car's max speed",
             "       (can pass 100% because the car cuts corners on the inside).", "", header, "-" * len(header)]
    print("\n".join(lines))
    all_results, first_group = [], None
    for name, tracks in groups.items():
        if not tracks:
            continue
        res = [run_policy(policy, t, record=True) for t in tracks]
        all_results += res
        first_group = first_group or (tracks, res)
        lines.append(summarize(name, res))
        print(lines[-1])
    text = "\n".join(lines)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "eval_report.txt").write_text(text + "\n")

    pre = groups["preset tracks"]
    traj_tracks = (pre + first_group[0])[:6]
    labels = [f"preset {i + 1}" for i in range(len(pre))] + [f"test {i + 1}" for i in range(6)]
    traj_res = [run_policy(policy, t, record=True) for t in traj_tracks]
    plot_trajectories(traj_tracks, traj_res, labels, C.REPORTS_DIR / "trajectories.png")
    plot_actions(all_results, C.REPORTS_DIR / "actions.png")
    plot_stages(make_tracks(10, C.TEST_SEED), C.REPORTS_DIR / "learning_stages.png")
    print(f"\nReports and plots saved to {C.REPORTS_DIR}")


if __name__ == "__main__":
    main()
