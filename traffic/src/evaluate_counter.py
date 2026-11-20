"""
Evaluate the camera network on the test set: intersections (backgrounds) the network didn't see in training.

    python -m src.evaluate_counter

Outputs in reports/:
    counter_report.txt      - mean error (MAE), percentage of exact counts, by shooting condition
    counter_scatter.png     - true count vs. the network's count
    counter_examples.png    - images with the density map the network "saw"
    counter_metrics.json    - (also used to train the traffic light with realistic counting noise)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .train_counter import get_device, load_data, metrics
from .vision import Counter


def main():
    d = load_data()
    X, y, cond = d["X_test"], d["y_test"], d["cond_test"]
    counter = Counter(device=get_device())
    preds, dens = [], []
    for i in range(0, len(X), 512):
        p, m = counter.count(X[i:i + 512])
        preds.append(p)
        dens.append(m)
    pred, dens = np.concatenate(preds), np.concatenate(dens)

    overall = metrics(pred, y)
    lines = ["Counting cars in the incoming lane (test set: 10 intersections never seen in training)", "",
             f"{'':10s} {'MAE':>6s} {'exact':>7s} {'±1 car':>7s} {'images':>7s}",
             f"{'all':10s} {overall['mae']:6.2f} {overall['exact'] * 100:6.1f}% {overall['within1'] * 100:6.1f}% {len(y):7d}"]
    per = {}
    for k, name in enumerate(C.CONDITIONS):
        m = cond == k
        if m.any():
            per[name] = metrics(pred[m], y[m])
            r = per[name]
            lines.append(f"{name:10s} {r['mae']:6.2f} {r['exact'] * 100:6.1f}% {r['within1'] * 100:6.1f}% {int(m.sum()):7d}")
    lines += ["", "MAE by true count (are long queues harder?):"]
    for lo, hi in [(0, 0), (1, 3), (4, 7), (8, 12), (13, 99)]:
        m = (y >= lo) & (y <= hi)
        if m.any():
            lines.append(f"  {lo:2d}-{min(hi, int(y.max())):2d} cars: MAE {np.abs(pred[m] - y[m]).mean():.2f}  ({int(m.sum())} images)")
    text = "\n".join(lines)
    print(text)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "counter_report.txt").write_text(text + "\n")
    resid_sd = float(np.std(pred - y))
    with open(C.REPORTS_DIR / "counter_metrics.json", "w") as f:
        json.dump({"overall": overall, "per_condition": per, "residual_sd": resid_sd}, f, indent=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#e9b949", "#e07a5f", "#3d5a80", "#81b29a"]
    for k, name in enumerate(C.CONDITIONS):
        m = cond == k
        ax.scatter(y[m] + np.random.uniform(-0.2, 0.2, m.sum()), pred[m], s=5, alpha=0.4, c=colors[k], label=name)
    lim = max(y.max(), pred.max()) + 1
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("true number of cars")
    ax.set_ylabel("network's count")
    ax.set_title(f"Test MAE = {overall['mae']:.2f} cars")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "counter_scatter.png", dpi=110)
    plt.close(fig)

    rng = np.random.default_rng(0)
    pick = rng.choice(len(y), 16, replace=False)
    fig, axes = plt.subplots(2, 8, figsize=(16, 6.5))
    for ax, i in zip(axes.flat, pick):
        ax.imshow(X[i], cmap="gray", extent=(0, 16, 28, 0))
        ax.imshow(dens[i], cmap="hot", alpha=0.5, extent=(0, 16, 28, 0), vmin=0, vmax=max(0.3, dens[i].max()))
        ok = round(pred[i]) == y[i]
        ax.set_title(f"{C.CONDITIONS[cond[i]]}: true {int(y[i])}\nnet {pred[i]:.1f}", fontsize=8,
                     color="green" if ok else "red")
        ax.axis("off")
    fig.suptitle("Density map: where the network thinks the cars are")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "counter_examples.png", dpi=100)
    plt.close(fig)
    print(f"\nReports saved to {C.REPORTS_DIR}")
    print("Next step:  python -m src.train_agent")


if __name__ == "__main__":
    main()
