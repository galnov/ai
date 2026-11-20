"""
Data exploration - what do the drawings in the dataset look like?

    python -m src.explore_data

Outputs in reports/:
    samples_grid.png   - 10 random drawings from each category
    mean_images.png    - the "average drawing" of each category (mean of 1,000 drawings)
    pixel_stats.png    - how much "ink" a drawing has in each category (simple vs. complex drawings)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_NAMES, NUM_CLASSES, REPORTS_DIR
from .data import load_splits, summarize


def main():
    splits = load_splits(use_custom=False)
    summarize(splits)
    X, y = splits["train"]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    n = 10
    fig, axes = plt.subplots(NUM_CLASSES, n + 1, figsize=((n + 1) * 0.9, NUM_CLASSES * 0.9))
    means, ink = [], []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(y == c)
        pick = rng.choice(idx, n, replace=False)
        axes[c, 0].text(1, 0.5, CLASS_NAMES[c], ha="right", va="center", fontsize=8,
                        transform=axes[c, 0].transAxes)
        axes[c, 0].axis("off")
        for j, i in enumerate(pick):
            axes[c, j + 1].imshow(X[i], cmap="gray_r")
            axes[c, j + 1].axis("off")
        sub = X[idx[:1000]].astype(np.float32)
        means.append(sub.mean(axis=0))
        ink.append((sub > 50).mean(axis=(1, 2)) * 100)
    fig.tight_layout(pad=0.1)
    fig.savefig(REPORTS_DIR / "samples_grid.png", dpi=90)
    plt.close(fig)

    cols = 7
    rows = int(np.ceil(NUM_CLASSES / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.8))
    for ax, m, name in zip(axes.flat, means, CLASS_NAMES):
        ax.imshow(m, cmap="magma")
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    for ax in list(axes.flat)[NUM_CLASSES:]:
        ax.axis("off")
    fig.suptitle("Mean image per category", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "mean_images.png", dpi=100)
    plt.close(fig)

    order = np.argsort([np.median(v) for v in ink])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.boxplot([ink[i] for i in order], showfliers=False)
    ax.set_xticks(range(1, NUM_CLASSES + 1), [CLASS_NAMES[i] for i in order], rotation=70, fontsize=8)
    ax.set_ylabel("% of pixels with ink")
    ax.set_title("How much ink each category uses (simple → complex drawings)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pixel_stats.png", dpi=100)
    plt.close(fig)
    print(f"Plots saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
