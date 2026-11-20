"""
Evaluate the network on the test set (drawings the network has never seen).

    python -m src.evaluate

Outputs in reports/:
    test_report.txt          - top-1 and top-3 accuracy, accuracy per category, the common confusions
    confusion_matrix.png     - a 42x42 confusion matrix
    per_class_accuracy.png   - accuracy per category, from easy to hard
    mistakes.png             - examples of the network's mistakes
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from .config import CLASS_NAMES, NUM_CLASSES, REPORTS_DIR
from .data import load_splits
from .utils import get_device, load_model, to_input


@torch.no_grad()
def predict(model, X, device, batch=1024):
    out = []
    for i in range(0, len(X), batch):
        xb = to_input(torch.from_numpy(X[i:i + batch]), device)
        logits = model(xb) + model(torch.flip(xb, dims=[3]))   # average with the mirror image
        out.append(torch.softmax(logits / 2, dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    device = get_device()
    model, ckpt = load_model(device=device)
    X, y = load_splits(use_custom=False)["test"]
    probs = predict(model, X, device)
    pred = probs.argmax(1)
    top3 = (np.argsort(-probs, axis=1)[:, :3] == y[:, None]).any(axis=1)
    acc, acc3 = (pred == y).mean(), top3.mean()
    cm = confusion_matrix(y, pred, labels=range(NUM_CLASSES))
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    off = cm.copy().astype(float)
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(-off.ravel()), off.shape))[0][:12]
    lines = [f"Test top-1 accuracy: {acc * 100:.2f}%",
             f"Test top-3 accuracy: {acc3 * 100:.2f}%   (the right answer is among the 3 top guesses)",
             f"Test drawings: {len(y):,}   Categories: {NUM_CLASSES}", "",
             "Most confused pairs (true -> predicted : count):"]
    lines += [f"  {CLASS_NAMES[a]:12s} -> {CLASS_NAMES[b]:12s} : {int(cm[a, b])}" for a, b in pairs]
    lines += ["", "Per-category accuracy:"]
    lines += [f"  {CLASS_NAMES[i]:12s} {per_class[i] * 100:5.1f}%" for i in np.argsort(-per_class)]
    text = "\n".join(lines)
    print(text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "test_report.txt").write_text(text + "\n")
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump({"top1": float(acc), "top3": float(acc3), "per_class": per_class.tolist()}, f)

    # confusion matrix
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=90, fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (row-normalized) - test accuracy {acc * 100:.1f}%")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=100)
    plt.close(fig)

    # accuracy per category
    order = np.argsort(-per_class)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(range(NUM_CLASSES), per_class[order] * 100,
           color=plt.cm.RdYlGn(per_class[order] ** 2))
    ax.set_xticks(range(NUM_CLASSES), [CLASS_NAMES[i] for i in order], rotation=70, fontsize=8)
    ax.axhline(acc * 100, color="k", ls="--", lw=1, label=f"average {acc * 100:.1f}%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.legend()
    ax.set_title("Accuracy per category")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "per_class_accuracy.png", dpi=100)
    plt.close(fig)

    # mistakes
    wrong = np.flatnonzero(pred != y)
    pick = np.random.default_rng(0).choice(wrong, size=min(32, len(wrong)), replace=False)
    fig, axes = plt.subplots(4, 8, figsize=(14, 7.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, pick):
        ax.imshow(X[i], cmap="gray_r")
        ax.set_title(f"guess: {CLASS_NAMES[pred[i]]} {probs[i, pred[i]] * 100:.0f}%\ntrue: {CLASS_NAMES[y[i]]}",
                     fontsize=7, color="#c0392b")
    fig.suptitle("Some mistakes (many are understandable!)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "mistakes.png", dpi=100)
    plt.close(fig)
    print(f"\nReports and plots saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
