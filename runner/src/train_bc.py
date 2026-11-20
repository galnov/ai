"""
Imitation learning (behavioral cloning): the network learns to do what the human players did.

    python -m src.train_bc                   # learn from the recordings in data/recordings/
    python -m src.train_bc --source bot      # learn from the bot's practice recordings only
    python -m src.train_bc --curve           # experiment: how much human play is needed? (no model saved)

This is ordinary supervised learning, like image classification: the input is the observation (what the
player saw), and the label is the key the player pressed at that tick (or "run" = no key).

Two details that matter:
  * Most ticks have no key press (about 98%). Without help, the network would learn to never press
    anything and still be "98% accurate". So rare keys get a larger weight in the loss.
  * Every run is also used as a mirror image (left <-> right), which doubles the data.

The recordings are split by whole runs: the validation runs were never seen in training.
The best epoch is chosen by how far the network actually gets when it plays, not by the loss.

Outputs:  models/imitation.pt,  reports/imitation_history.json,  reports/imitation_curves.png
          (with --curve: reports/imitation_data_curve.png)
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
from .game import MIRROR_ACTION, mirror_obs
from .nets import RunnerNet, count_parameters, pick_device, save_model
from .players import ModelPlayer, play
from .recordings import build_dataset, load_recordings, pick_source, summary


def split_runs(recs, val_fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(recs))
    n_val = max(1, int(round(val_fraction * len(recs)))) if len(recs) > 1 else 0
    return [recs[i] for i in order[n_val:]], [recs[i] for i in order[:n_val]]


def train(train_recs, val_recs, epochs, device, log=True):
    """Trains one network. Returns (best network, history)."""
    P = C.BC
    torch.manual_seed(P["seed"])
    rng = np.random.default_rng(P["seed"])
    D = build_dataset(train_recs)
    X = np.concatenate([D["obs"], mirror_obs(D["obs"])])
    y = np.concatenate([D["actions"], MIRROR_ACTION[D["actions"]]])
    freq = np.bincount(y, minlength=C.N_ACTIONS) / len(y)
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-4))              # rare keys count more
    w = w / (w * freq).sum()
    V = build_dataset(val_recs) if val_recs else None

    net = RunnerNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
    wt = torch.as_tensor(w, dtype=torch.float32, device=device)
    if log:
        print(f"Training ticks: {len(y):,} (with mirror images) | key presses: {np.mean(y != 0) * 100:.1f}% | "
              f"weights: {np.round(w, 1).tolist()}")

    hist, best, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        perm = torch.as_tensor(rng.permutation(len(y)))
        total, steps = 0.0, 0
        for s in range(0, len(y), P["batch_size"]):
            j = perm[s:s + P["batch_size"]]
            logits = net(Xt[j].to(device))
            loss = F.cross_entropy(logits, yt[j].to(device), weight=wt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        sched.step()

        net.eval()
        h = {"epoch": epoch, "loss": total / steps}
        if V is not None:
            with torch.no_grad():
                pred = net(torch.as_tensor(V["obs"], device=device)).argmax(1).cpu().numpy()
            keys = V["actions"] != 0
            h["key_recall"] = float((pred[keys] == V["actions"][keys]).mean()) if keys.any() else 0.0
            h["press_rate"] = float((pred != 0).mean())
            h["human_press_rate"] = float(keys.mean())
        dist, crashes, _ = play(ModelPlayer(net=net, kind="imitation", device=device), C.VAL_SEEDS, max_ticks=2400)
        h["val_distance"] = float(dist.mean())
        h["crashes"] = np.bincount(crashes, minlength=4)[1:].tolist()
        mark = ""
        if h["val_distance"] > best:
            best, mark = h["val_distance"], "  * best so far"
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        hist.append(h)
        if log:
            extra = (f" | keys right {h['key_recall'] * 100:4.1f}% | presses {h['press_rate'] * 100:.1f}% "
                     f"(humans {h['human_press_rate'] * 100:.1f}%)") if V is not None else ""
            print(f"Epoch {epoch:2d}/{epochs} | loss {h['loss']:.3f}{extra} | plays: {h['val_distance']:6.0f} m "
                  f"| {time.time() - t0:3.0f}s{mark}")
    net.load_state_dict(best_state)
    net.eval()
    return net, hist


def plot_history(hist, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["loss"] for h in hist], color="#e4572e")
    ax[0].set_title("Training loss (weighted cross-entropy)")
    ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [h["val_distance"] for h in hist], color="#2e86ab", marker="o")
    ax[1].set_title("Distance when the network plays (16 validation tracks, 2 min)")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("meters")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def data_curve(train_recs, val_recs, epochs, device):
    """How much play does imitation need? Train on growing parts of the recordings."""
    rng = np.random.default_rng(1)
    order = [train_recs[i] for i in rng.permutation(len(train_recs))]
    points = []
    for frac in (0.1, 0.25, 0.5, 1.0):
        part = order[:max(1, int(round(frac * len(order))))]
        minutes = summary(part)["minutes"]
        _, hist = train(part, val_recs, epochs, device, log=False)
        best = max(h["val_distance"] for h in hist)
        points.append({"fraction": frac, "runs": len(part), "minutes": minutes, "val_distance": best})
        print(f"  {len(part):3d} runs ({minutes:5.1f} minutes of play) -> {best:6.0f} m")
    return points


def main():
    ap = argparse.ArgumentParser(description="Imitation learning from recorded runs")
    ap.add_argument("--source", default="auto", choices=["auto", "human", "bot", "all"],
                    help="which recordings: auto = human runs if there are any, otherwise the bot's")
    ap.add_argument("--epochs", type=int, default=C.BC["epochs"])
    ap.add_argument("--curve", action="store_true", help="experiment: distance vs. minutes of play")
    args = ap.parse_args()

    source = pick_source(args.source)
    recs = load_recordings(source)
    if not recs:
        raise SystemExit("No recordings found. Play the app (every run is recorded), "
                         "or create practice runs:  python -m src.make_bot_demos")
    if source == "bot":
        print("[!] Learning from the BOT's practice recordings, not from people.\n"
              "    For the real experiment, play the app and then run this again.\n")
    s = summary(recs)
    print(f"Recordings ({source}): {s['runs']} runs, {s['minutes']:.1f} minutes of play, "
          f"average distance {s['mean_distance']:.0f} m")
    device = pick_device()
    print(f"Device: {device} | network parameters: {count_parameters(RunnerNet()):,}")
    train_recs, val_recs = split_runs(recs, C.BC["val_fraction"], C.BC["seed"])
    print(f"Training runs: {len(train_recs)} | validation runs: {len(val_recs)}\n")
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.curve:
        print("How much play does imitation need?")
        points = data_curve(train_recs, val_recs, args.epochs, device)
        (C.REPORTS_DIR / "imitation_data_curve.json").write_text(json.dumps(points, indent=1))
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot([p["minutes"] for p in points], [p["val_distance"] for p in points], marker="o", color="#2e86ab")
        ax.set_xlabel("minutes of recorded play used for training")
        ax.set_ylabel("distance when the network plays (m)")
        ax.set_title("Imitation learning: more play -> a better copy?")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(C.REPORTS_DIR / "imitation_data_curve.png", dpi=110)
        print(f"\nSaved {C.REPORTS_DIR / 'imitation_data_curve.png'}")
        return

    net, hist = train(train_recs, val_recs, args.epochs, device)
    best = max(h["val_distance"] for h in hist)
    save_model(net, C.BC_PATH, "imitation", source=source, runs=s["runs"], minutes=s["minutes"],
               val_distance=best)
    (C.REPORTS_DIR / "imitation_history.json").write_text(json.dumps(hist, indent=1))
    plot_history(hist, C.REPORTS_DIR / "imitation_curves.png")
    print(f"\nDone! Best validation distance: {best:.0f} m  (saved to {C.BC_PATH})")
    print("Next step:  python -m src.train_dqn   (and  python -m src.train_dqn --combo)")


if __name__ == "__main__":
    main()
