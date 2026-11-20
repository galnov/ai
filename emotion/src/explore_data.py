"""
Exploratory data analysis (EDA) - get to know the data before training.

    python -m src.explore_data

Outputs in reports/:
    class_distribution.png   - how many images there are per emotion (imbalance!)
    samples_grid.png         - examples of each emotion
    mean_faces.png           - the "average face" of each emotion
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import EMOTION_COLOR, EMOTIONS, IMG_SIZE, REPORTS_DIR
from .data_sources import class_counts, load_splits, summarize


def to_array(source):
    if isinstance(source, np.ndarray):
        return source
    img = Image.open(source).convert("L")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img)


def main():
    splits = load_splits()
    summarize(splits)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train = splits["train"]

    # 1. class distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(EMOTIONS))
    bottom = np.zeros(len(EMOTIONS))
    for name, alpha in [("train", 1.0), ("val", 0.6), ("test", 0.35)]:
        c = np.array(class_counts(splits[name]))
        ax.bar(x, c, bottom=bottom, color=[EMOTION_COLOR[e] for e in EMOTIONS], alpha=alpha,
               edgecolor="white", label=name)
        bottom += c
    for i, total in enumerate(bottom):
        ax.text(i, total, f"{int(total)}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, EMOTIONS)
    ax.set_title("Images per emotion (train / val / test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)

    # 2. examples + 3. average faces
    rng = np.random.default_rng(0)
    by_label = {i: [s for s, l in train if l == i] for i in range(len(EMOTIONS))}
    n = 8
    fig, axes = plt.subplots(len(EMOTIONS), n, figsize=(n * 1.3, len(EMOTIONS) * 1.4))
    means = []
    for i, e in enumerate(EMOTIONS):
        items = by_label[i]
        pick = rng.choice(len(items), size=min(n, len(items)), replace=False) if items else []
        for j in range(n):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(pick):
                ax.imshow(to_array(items[pick[j]]), cmap="gray")
        axes[i, 0].set_title(e, fontsize=9, loc="left")
        sub = rng.choice(len(items), size=min(500, len(items)), replace=False) if items else []
        means.append(np.mean([to_array(items[k]) for k in sub], axis=0) if len(sub) else np.zeros((IMG_SIZE, IMG_SIZE)))
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "samples_grid.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(EMOTIONS), figsize=(len(EMOTIONS) * 1.8, 2.2))
    for ax, e, m in zip(axes, EMOTIONS, means):
        ax.imshow(m, cmap="gray")
        ax.set_title(e, fontsize=9)
        ax.axis("off")
    fig.suptitle("Mean face per emotion", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "mean_faces.png", dpi=120)
    plt.close(fig)
    print(f"Plots saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
