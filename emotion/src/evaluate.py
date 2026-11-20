"""
Evaluate the trained model on the test set (images the network has never seen).

    python -m src.evaluate

Outputs in reports/:
    test_report.txt          - accuracy, precision/recall/F1 per emotion
    confusion_matrix.png     - confusion matrix: which emotion gets confused with which
    predictions_grid.png     - example images with the network's prediction
    gradcam_grid.png         - what the network "looked at" in each image
    metrics.json
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import EMOTIONS, REPORTS_DIR
from .dataset import MEAN, STD, make_loaders
from .gradcam import GradCAM
from .utils import get_device, load_model


@torch.no_grad()
def predict_all(model, loader, device):
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        # Test-Time Augmentation: average of the image and its mirror image
        logits = model(x) + model(torch.flip(x, dims=[3]))
        p = torch.softmax(logits / 2, dim=1).cpu()
        ys.append(y)
        probs.append(p)
        preds.append(p.argmax(1))
    return torch.cat(ys).numpy(), torch.cat(preds).numpy(), torch.cat(probs).numpy()


def plot_confusion(cm, path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v*100:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def denorm(t):
    return (t.squeeze().cpu().numpy() * STD + MEAN).clip(0, 1)


def plot_examples(dataset, ys, preds, probs, path, n=24, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    cols = 8
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.1))
    for ax, i in zip(axes.flat, idx):
        img, _ = dataset[i]
        ok = ys[i] == preds[i]
        ax.imshow(denorm(img), cmap="gray")
        ax.set_title(f"{EMOTIONS[preds[i]]} {probs[i, preds[i]]*100:.0f}%\n(true: {EMOTIONS[ys[i]]})",
                     fontsize=7, color="green" if ok else "red")
        ax.axis("off")
    for ax in list(axes.flat)[len(idx):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_gradcam(model, dataset, device, path, n=16, seed=1):
    cam = GradCAM(model, model.cam_layer)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    cols = 8
    rows = int(np.ceil(len(idx) / cols)) * 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    axes = np.atleast_2d(axes)
    for k, i in enumerate(idx):
        img, y = dataset[i]
        heat, probs = cam(img.unsqueeze(0).to(device))
        r, c = (k // cols) * 2, k % cols
        axes[r, c].imshow(denorm(img), cmap="gray")
        axes[r, c].set_title(f"true: {EMOTIONS[y]}", fontsize=7)
        axes[r + 1, c].imshow(denorm(img), cmap="gray")
        axes[r + 1, c].imshow(heat, cmap="jet", alpha=0.45)
        pred = int(probs.argmax())
        axes[r + 1, c].set_title(f"pred: {EMOTIONS[pred]}", fontsize=7)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Grad-CAM: where the network looks", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    cam.remove()


def main():
    device = get_device()
    model, ckpt = load_model(device=device)
    _, _, _, test_loader = make_loaders(batch_size=64, workers=2)
    ys, preds, probs = predict_all(model, test_loader, device)

    acc = accuracy_score(ys, preds)
    labels = list(range(len(EMOTIONS)))
    report = classification_report(ys, preds, labels=labels, target_names=EMOTIONS, digits=3, zero_division=0)
    cm = confusion_matrix(ys, preds, labels=labels)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    text = (f"Test accuracy: {acc*100:.2f}%   (samples: {len(ys)})\n"
            f"Best validation accuracy during training: {ckpt.get('val_acc', 0)*100:.2f}% "
            f"(epoch {ckpt.get('epoch', '?')})\n\n{report}")
    print(text)
    (REPORTS_DIR / "test_report.txt").write_text(text)
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump({"test_accuracy": float(acc), "confusion_matrix": cm.tolist(), "emotions": EMOTIONS}, f, indent=2)

    plot_confusion(cm, REPORTS_DIR / "confusion_matrix.png")
    plot_examples(test_loader.dataset, ys, preds, probs, REPORTS_DIR / "predictions_grid.png")
    plot_gradcam(model, test_loader.dataset, device, REPORTS_DIR / "gradcam_grid.png")
    print(f"Reports and plots saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
