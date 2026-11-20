"""
Train network 1: counting cars from the camera.

    python -m src.train_counter
    python -m src.train_counter --quick      # quick check (saved to a separate file)

Before that:  python -m src.make_camera_data
Outputs:      models/counter_cnn.pt,  reports/counter_history.json,  reports/counter_curves.png
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .counter import CounterCNN, count_parameters, photometric_augment

QUICK_PATH = C.MODELS_DIR / "counter_cnn_quick.pt"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data():
    if not C.CAMERA_DATA.exists():
        raise FileNotFoundError("The image dataset was not found. Run first:  python -m src.make_camera_data")
    return np.load(C.CAMERA_DATA)


def to_input(x_uint8, device):
    """uint8 (B,H,W) -> float (B,1,H,W) in the range 0..1. Identical in training and inference."""
    return x_uint8.to(device).float().div_(255.0).unsqueeze(1)


@torch.no_grad()
def predict(model, X, device, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = to_input(torch.as_tensor(X[i:i + batch]), device)
        out.append(model(xb)[0].float().cpu())
    return torch.cat(out).numpy()


def metrics(pred, y):
    err = pred - y
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "exact": float((np.round(pred) == y).mean()), "within1": float((np.abs(np.round(pred) - y) <= 1).mean())}


def plot_history(h, path):
    ep = [r["epoch"] for r in h]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    a1.plot(ep, [r["train_loss"] for r in h], "o-", label="train loss (Huber)")
    a1.set_yscale("log")
    a1.set_title("Training loss")
    a2.plot(ep, [r["val_mae"] for r in h], "o-", label="validation MAE (cars)")
    a2.plot(ep, [1 - r["val_within1"] for r in h], "o--", label="validation: off by more than 1 car")
    a2.set_title("Validation")
    for a in (a1, a2):
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train the car counting network")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs = 2
    out_path = QUICK_PATH if args.quick else C.COUNTER_PATH

    torch.manual_seed(0)
    device = get_device()
    print(f"Device: {device}")
    d = load_data()
    data_dev = device if device.type == "cuda" else torch.device("cpu")
    Xtr = torch.from_numpy(d["X_train"]).to(data_dev)
    ytr = torch.from_numpy(d["y_train"]).to(data_dev)
    if args.quick:
        Xtr, ytr = Xtr[:3000], ytr[:3000]
    Xva, yva = d["X_val"], d["y_val"]
    print(f"Training: {len(ytr):,} images | validation: {len(yva):,}")

    model = CounterCNN(args.width).to(device)
    print(f"Network parameters: {count_parameters(model):,}")
    steps = len(ytr) // args.batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps)
    history, best = [], float("inf")
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(ytr), device=data_dev)
        total = 0.0
        for s in range(steps):
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            xb = photometric_augment(Xtr[idx].float().div_(255.0).unsqueeze(1)).to(device, non_blocking=True)
            yb = ytr[idx].to(device, non_blocking=True)
            count, _ = model(xb)
            loss = F.smooth_l1_loss(count, yb)          # Huber: robust to "hard" images
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            if s % 50 == 0:
                print(f"\r  epoch {epoch}: {s}/{steps}", end="", flush=True)
        m = metrics(predict(model, Xva, device), yva)
        history.append({"epoch": epoch, "train_loss": total / steps, "val_mae": m["mae"],
                        "val_exact": m["exact"], "val_within1": m["within1"]})
        mark = ""
        if m["mae"] < best:
            best, mark = m["mae"], "  * saved"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "width": args.width, "val_mae": m["mae"],
                        "val_within1": m["within1"], "epoch": epoch, "quick": bool(args.quick)}, out_path)
        print(f"\rEpoch {epoch:2d}/{args.epochs} | loss {total / steps:.4f} | val MAE {m['mae']:.3f} cars | "
              f"exact {m['exact'] * 100:4.1f}% | within ±1 {m['within1'] * 100:4.1f}% | {time.time() - t0:3.0f}s{mark}")
        with open(C.REPORTS_DIR / "counter_history.json", "w") as f:
            json.dump(history, f, indent=1)
        plot_history(history, C.REPORTS_DIR / "counter_curves.png")
    print(f"\nDone! Best validation MAE: {best:.3f} cars  (saved to {out_path})")
    print("Next step:  python -m src.evaluate_counter")


if __name__ == "__main__":
    main()
