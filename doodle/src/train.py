"""
Train the network.

    python -m src.train                  # full training
    python -m src.train --quick          # quick check that everything works (NOT saved as the app's model!)

Outputs:
    models/doodle_cnn.pt            - the best network (by validation accuracy)
    reports/history.json
    reports/training_curves.png
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .config import MODEL_PATH, MODELS_DIR, NUM_CLASSES, PER_CLASS, REPORTS_DIR
from .data import load_splits, summarize
from .model import DoodleCNN, augment, count_parameters
from .utils import get_device, save_checkpoint, to_input

QUICK_PATH = MODELS_DIR / "doodle_cnn_quick.pt"


@torch.no_grad()
def evaluate(model, X, y, device, batch=1024):
    model.eval()
    correct, top3, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for i in range(0, len(y), batch):
        xb = to_input(X[i:i + batch], device)
        yb = y[i:i + batch].to(device)
        logits = model(xb)
        loss_sum += ce(logits, yb).item()
        correct += (logits.argmax(1) == yb).sum().item()
        top3 += (logits.topk(3, dim=1).indices == yb[:, None]).any(dim=1).sum().item()
    n = len(y)
    return loss_sum / n, correct / n, top3 / n


def plot_history(h, path):
    ep = [r["epoch"] for r in h]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(ep, [r["train_loss"] for r in h], "o-", label="train")
    a1.plot(ep, [r["val_loss"] for r in h], "o-", label="validation")
    a1.set_title("Loss")
    a2.plot(ep, [r["train_acc"] * 100 for r in h], "o-", label="train (with augmentation)")
    a2.plot(ep, [r["val_acc"] * 100 for r in h], "o-", label="validation top-1")
    a2.plot(ep, [r["val_top3"] * 100 for r in h], "o--", label="validation top-3")
    a2.set_title("Accuracy (%)")
    for a in (a1, a2):
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train the drawing recognition network")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="quick check: a little data, 2 epochs")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.per_class = 2, 1500
    out_path = QUICK_PATH if args.quick else MODEL_PATH

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    splits = load_splits(args.per_class, args.seed)
    summarize(splits)
    # on an NVIDIA GPU (e.g. a T4 in Colab) all the data is loaded into the card's memory, and augmentation runs there too.
    # otherwise - the data stays in regular memory and augmentation runs on the CPU.
    data_device = device if device.type == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    Xtr, ytr = (torch.from_numpy(a).to(data_device) for a in splits["train"])
    Xva, yva = (torch.from_numpy(a) for a in splits["val"])

    model = DoodleCNN(NUM_CLASSES, width=args.width).to(device)
    print(f"Network parameters: {count_parameters(model):,}")
    if args.quick:
        print(f"*** Quick-test mode - the model will be saved to {QUICK_PATH.name} and will NOT be used by the app ***")

    steps_per_epoch = len(ytr) // args.batch_size
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # OneCycle: the learning rate rises at first ("warm-up") and then decreases gradually until the end
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    total_steps=args.epochs * steps_per_epoch)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history, best = [], 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(ytr), device=data_device)
        loss_sum, correct, seen = 0.0, 0, 0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            xb = augment(Xtr[idx].float().div_(255.0).unsqueeze(1)).to(device, non_blocking=True)
            yb = ytr[idx].to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
            if s % 200 == 0:
                print(f"\r  epoch {epoch}: {s}/{steps_per_epoch} batches", end="", flush=True)

        va_loss, va_acc, va_top3 = evaluate(model, Xva, yva, device)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
                        "val_loss": va_loss, "val_acc": va_acc, "val_top3": va_top3,
                        "lr": optimizer.param_groups[0]["lr"]})
        mark = ""
        if va_acc > best:
            best, mark = va_acc, "  * saved"
            save_checkpoint(model, out_path, width=args.width, val_acc=float(va_acc),
                            val_top3=float(va_top3), epoch=epoch, train_samples=int(len(ytr)),
                            quick=bool(args.quick))
        print(f"\rEpoch {epoch:2d}/{args.epochs} | train loss {loss_sum / seen:.3f} acc {correct / seen * 100:5.1f}% | "
              f"val acc {va_acc * 100:5.1f}% top-3 {va_top3 * 100:5.1f}% | {time.time() - t0:4.0f}s{mark}")
        with open(REPORTS_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        plot_history(history, REPORTS_DIR / "training_curves.png")

    print(f"\nDone! Best validation accuracy: {best * 100:.1f}%  (saved to {out_path})")
    print("That was only a check. For real training:  python -m src.train" if args.quick
          else "Next step:  python -m src.evaluate   then   python -m app.server")


if __name__ == "__main__":
    main()
