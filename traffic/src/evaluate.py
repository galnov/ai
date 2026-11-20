"""
Evaluate the whole system: camera -> counting network -> agent -> traffic light.

    python -m src.evaluate

We compare on the same 6 test scenarios (20 minutes each, identical traffic for all):
  1. A fixed traffic light (30 seconds for each direction)
  2. A "longest queue" traffic light: switches to the direction where more cars are waiting
  3. The agent with a "perfect sensor" (the exact count from the simulation)
  4. The agent with the cameras: the count comes from the camera network (day, dusk, night and rain)

The cameras shoot intersections the network didn't see in training.
Outputs in reports/:  system_report.txt,  system_results.png,  queue_timeline.png
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .agent import Controller
from .camera import Scene, render_cameras, vehicle_list
from .traffic import TrafficEnv, fixed_time, max_queue

TEST_SEEDS = (500, 501, 502, 503, 504, 505)


def run(policy, camera=None, condition="day", log=False):
    """
    policy(env, counts) -> actions.
    camera: a Counter object - if given, the count comes from the cameras.
    """
    E = len(TEST_SEEDS)
    env = TrafficEnv(E, seed=TEST_SEEDS[0])
    scenes = [Scene(100 + e) for e in range(E)]
    rng = np.random.default_rng(7)
    timeline, errors = [], []
    while not (env.t >= C.EPISODE_SECONDS).all():
        true = env.counts()
        if camera is not None:
            imgs = []
            for e in range(E):
                imgs += [v[1] for v in render_cameras(scenes[e], vehicle_list(env, e), condition, rng)]
            counts = camera.count(np.stack(imgs))[0].reshape(E, C.N_LANES)
            errors.append(np.abs(counts - true))
        else:
            counts = true.astype(float)
        env.step(policy(env, counts))
        if log:
            timeline.append(((env.v < C.WAIT_SPEED) & (env.s > 0) & np.isfinite(env.s)).sum(axis=(1, 2))[0]
                            + env.backlog[0].sum())
    m = env.metrics()
    out = {"avg_wait": float(m["avg_wait"].mean()), "avg_queue": float(m["avg_queue"].mean()),
           "served": int(m["served"].sum()), "switches": float(m["switches"].mean())}
    if errors:
        out["count_mae"] = float(np.mean(errors))
    if log:
        out["timeline"] = timeline
    return out


def main():
    agent = Controller()
    camera = None
    try:
        from .vision import Counter
        camera = Counter()
    except FileNotFoundError as e:
        print(f"[!] {e}\n    Continuing without cameras.")
    dqn = lambda env, c: agent.act(env.observe(c))[0]

    rows = []
    t0 = time.time()
    for name, pol, cam, cond in [
        ("fixed-time 30s", lambda e, c: fixed_time(e), None, None),
        ("max-queue rule", lambda e, c: max_queue(e, c), None, None),
        ("DQN + perfect sensor", dqn, None, None),
    ] + ([(f"DQN + camera ({c})", dqn, camera, c) for c in C.CONDITIONS] if camera else []):
        r = run(pol, cam, cond or "day", log=(name in ("fixed-time 30s", "DQN + perfect sensor")))
        rows.append((name, r))
        extra = f" | count: mean error {r['count_mae']:.2f}" if "count_mae" in r else ""
        print(f"  {name:26s} average wait {r['avg_wait']:5.1f} s | average queue {r['avg_queue']:5.1f}{extra}"
              f"   ({time.time() - t0:.0f} s)")

    base = rows[0][1]["avg_wait"]
    lines = ["Same 6 test scenarios (20 min each) for every controller.", "",
             f"{'controller':28s} {'avg wait/veh':>12s} {'vs fixed':>9s} {'avg queue':>10s} {'count MAE':>10s}"]
    for name, r in rows:
        mae = f"{r['count_mae']:.2f}" if "count_mae" in r else "-"
        lines.append(f"{name:28s} {r['avg_wait']:10.1f} s {(1 - r['avg_wait'] / base) * 100:8.0f}% "
                     f"{r['avg_queue']:10.1f} {mae:>10s}")
    text = "\n".join(lines)
    print("\n" + text)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "system_report.txt").write_text(text + "\n")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    names = [n for n, _ in rows]
    vals = [r["avg_wait"] for _, r in rows]
    colors = ["#e76f51", "#e9c46a", "#2a9d8f"] + ["#3d5a80"] * (len(rows) - 3)
    bars = ax.bar(range(len(rows)), vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}s", ha="center", va="bottom")
    ax.set_xticks(range(len(rows)), names, rotation=20, ha="right")
    ax.set_ylabel("average wait per vehicle (s)")
    ax.set_title("Lower is better")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "system_results.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for name, r in rows:
        if "timeline" in r:
            t = np.arange(len(r["timeline"])) * C.DECISION_EVERY / 60
            ax.plot(t, r["timeline"], label=name)
    ax.set_xlabel("minutes")
    ax.set_ylabel("waiting vehicles")
    ax.set_title("Queue over one test scenario")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "queue_timeline.png", dpi=110)
    plt.close(fig)
    print(f"\nReports saved to {C.REPORTS_DIR}.  Next step:  python -m app.server")


if __name__ == "__main__":
    main()
