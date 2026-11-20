"""
Train the network.

Run (from the project folder):
    python -m src.train                    # full training
    python -m src.train --width 32         # smaller, faster network (good for a computer without a GPU)
    python -m src.train --quick            # quick check that everything works (under a minute)
                                           # note: the quick model is saved to a separate file and is not used by the app

Outputs:
    models/emotion_cnn.pt          - the best model (by validation accuracy)
    reports/history.json           - metrics for every epoch
    reports/training_curves.png    - loss and accuracy plots
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .config import EMOTIONS, MIN_TRAIN_SAMPLES, MODEL_PATH, QUICK_MODEL_PATH, REPORTS_DIR
from .data_sources import summarize
from .dataset import compute_class_weights, make_loaders
from .model import EmotionCNN, count_parameters
from .utils import get_device, save_checkpoint, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One epoch. If an optimizer is passed - training mode, otherwise - evaluation mode."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)                  # 1. forward pass
            loss = criterion(logits, y)        # 2. compute the loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()                # 3. compute gradients (backpropagation)
                optimizer.step()               # 4. update the weights
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, [h["train_acc"] * 100 for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] * 100 for h in history], label="validation")
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Train the emotion recognition network")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=64, help="64 = accurate, 32 = fast")
    p.add_argument("--patience", type=int, default=8, help="early stopping after N epochs without improvement")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="quick check on a little data")
    args = p.parse_args()

    if args.quick:
        args.epochs, args.width = 3, 16
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    splits, train_loader, val_loader, _ = make_loaders(
        args.batch_size, args.workers, args.seed, limit=2000 if args.quick else None)
    summarize(splits)
    n_train = len(splits["train"])
    out_path = QUICK_MODEL_PATH if args.quick else MODEL_PATH
    if args.quick:
        print("\n*** Quick-test mode: 2,000 images, 3 epochs, a tiny network. ***")
        print(f"*** The model will be saved to {QUICK_MODEL_PATH.name} and will NOT be used by the app. ***\n")
    elif n_train < MIN_TRAIN_SAMPLES:
        print(f"\n[!] Warning: found only {n_train:,} training images (FER-2013 has about 28,000).")
        print("    Check that the dataset is in data/fer2013/train and data/fer2013/test (see data/README.md).\n")

    model = EmotionCNN(num_classes=len(EMOTIONS), width=args.width).to(device)
    print(f"Network parameters: {count_parameters(model):,}")

    # loss: cross-entropy with class weights and label smoothing (prevents over-confidence)
    class_weights = compute_class_weights(splits["train"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # when validation accuracy stops improving - halve the learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history, best_acc, bad_epochs = [], 0.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_acc)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": float(tr_loss), "train_acc": float(tr_acc),
                        "val_loss": float(va_loss), "val_acc": float(va_acc), "lr": float(lr)})

        mark = ""
        if va_acc > best_acc:
            best_acc, bad_epochs, mark = va_acc, 0, "  * saved"
            save_checkpoint(model, args.width, out_path, val_acc=float(va_acc), epoch=epoch,
                            train_samples=n_train, quick=bool(args.quick))
        else:
            bad_epochs += 1
        print(f"Epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.3f} acc {tr_acc*100:5.1f}% | "
              f"val loss {va_loss:.3f} acc {va_acc*100:5.1f}% | lr {lr:.1e} | {time.time()-t0:.0f}s{mark}")

        with open(REPORTS_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        plot_history(history, REPORTS_DIR / "training_curves.png")

        if bad_epochs >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs")
            break

    print(f"\nDone! Best validation accuracy: {best_acc*100:.1f}%")
    print(f"Model saved to {out_path}")
    if args.quick:
        print("That was only a check. For real training:  python -m src.train")
    else:
        print("Next step: python -m src.evaluate")


if __name__ == "__main__":
    main()
