"""
Training the Breakout player with PPO, from pixels only.

    python -m src.train                   # full training: 10 million decisions (a GPU helps: Colab, about 1 hour)
    python -m src.train --steps 3000000   # shorter

The network gets nothing but the last 2 screens (48 x 48 gray pixels) and the reward: +1 for a brick,
-1 for a lost ball. It is never told where the ball or the paddle is - it has to find them in the pixels.

Every 25 updates it plays 32 full games (3 balls each) without randomness and prints the average number
of bricks. The best version in the second half of training is kept.

Outputs:
    models/breakout_ppo.pt          the best network - the one the web page uses
    models/snapshots/steps_*.pt     the network at earlier stages (the "learning stages" in the web page)
    reports/history.json, reports/training_curves.png
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config as C
from .game import Breakout
from .ppo import PixelNet, ppo_update, save


@torch.no_grad()
def play_games(model, games=C.EVAL_GAMES, seed=123, device="cpu"):
    """Full games without randomness. Returns the number of bricks in each game."""
    env = Breakout(games, seed)
    scores = np.full(games, -1)
    finished = np.zeros(games, bool)
    while not finished.all():
        a = model.play(env.observe(), device)
        before = env.score.copy()
        _, _, done = env.step(a)
        new = done & ~finished
        scores[new] = before[new]
        finished |= done
    return scores


def plot_history(h, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(h["steps"], h["train_bricks"], color="#3d5a80", label="training games (with exploration)")
    axes[0].plot(h["eval_steps"], h["eval_bricks"], "o-", ms=3, color="#e76f51", label="evaluation (no randomness)")
    axes[0].set_title("Bricks per game (3 balls)")
    axes[0].legend()
    axes[1].plot(h["steps"], h["entropy"], color="#2a9d8f")
    axes[1].set_title("Policy entropy (how random its choices are; max 1.10)")
    for a in axes:
        a.set_xlabel("decisions (millions)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    P = dict(C.PPO)
    ap = argparse.ArgumentParser(description="Train a Breakout player from pixels with PPO")
    ap.add_argument("--steps", type=int, default=P["total_steps"])
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, T = P["n_envs"], P["n_steps"]
    n_updates = max(1, args.steps // (N * T))

    model = PixelNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=P["lr"], eps=1e-5)
    env = Breakout(N, seed=args.seed)
    snaps = sorted(C.SNAPSHOT_STEPS)
    games = []
    best = -1.0
    h = {k: [] for k in ("steps", "train_bricks", "entropy", "eval_steps", "eval_bricks")}
    t0 = time.time()
    print(f"Starting on {dev}: {n_updates} updates x {N * T:,} decisions = {n_updates * N * T / 1e6:.1f} million")
    if dev.type == "cpu":
        print("No GPU found - it works, but slowly. On Colab: Runtime > Change runtime type > GPU")

    for u in range(1, n_updates + 1):
        for g in opt.param_groups:
            g["lr"] = P["lr"] * (1 - (u - 1) / n_updates)
        O = torch.zeros((T, N, C.FRAMES, C.SCREEN, C.SCREEN), dtype=torch.uint8, device=dev)
        A = torch.zeros((T, N), dtype=torch.long, device=dev)
        LP, V, R, D = (torch.zeros((T, N), device=dev) for _ in range(4))
        # ---------- 1. play ----------
        for t in range(T):
            o = torch.as_tensor(env.observe(), device=dev)
            with torch.no_grad():
                logits, v = model(o)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            before = env.score.copy()
            r, lost, done = env.step(a.cpu().numpy())
            games.extend(before[done].tolist())
            O[t], A[t], LP[t], V[t] = o, a, dist.log_prob(a), v
            R[t] = torch.as_tensor(r, device=dev)
            D[t] = torch.as_tensor(lost | done, device=dev)          # a lost ball ends the episode for learning
        # ---------- 2. GAE ----------
        with torch.no_grad():
            _, last_v = model(torch.as_tensor(env.observe(), device=dev))
        adv = torch.zeros((T, N), device=dev)
        gae = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            nonterm = 1 - D[t]
            delta = R[t] + P["gamma"] * nv * nonterm - V[t]
            gae = delta + P["gamma"] * P["lam"] * nonterm * gae
            adv[t] = gae
        ret = adv + V
        flat = lambda x: x.reshape((N * T,) + x.shape[2:])
        # ---------- 3. learn ----------
        _, _, ent = ppo_update(model, opt, tuple(flat(x) for x in (O, A, LP, adv, ret)), P)

        steps = u * N * T
        while snaps and steps >= snaps[0]:
            save(model, C.SNAPSHOT_DIR / f"steps_{snaps[0]:08d}.pt", steps=steps)
            snaps.pop(0)
        if u % 5 == 0:
            h["steps"].append(steps / 1e6)
            h["train_bricks"].append(float(np.mean(games[-200:])) if games else 0.0)
            h["entropy"].append(float(ent))
        if u % C.EVAL_EVERY == 0 or u == n_updates:
            model.eval()
            sc = play_games(model, device=dev)
            model.train()
            h["eval_steps"].append(steps / 1e6)
            h["eval_bricks"].append(float(sc.mean()))
            star = ""
            if u < n_updates // 2 or sc.mean() >= best:      # in the second half: keep the best
                if u >= n_updates // 2:
                    best, star = float(sc.mean()), "  * best"
                save(model, C.MODEL_PATH, steps=steps, eval_bricks=float(sc.mean()))
            train_b = np.mean(games[-200:]) if games else 0
            print(f"update {u:4d}/{n_updates} | {steps / 1e6:5.2f}M | training games: {train_b:6.1f} bricks | "
                  f"evaluation: {sc.mean():6.1f} bricks (min {sc.min()}, max {sc.max()}) | entropy {ent:.2f} | "
                  f"{time.time() - t0:5.0f} s{star}")
            C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(C.REPORTS_DIR / "history.json", "w") as f:
                json.dump(h, f)
            plot_history(h, C.REPORTS_DIR / "training_curves.png")
    print(f"\nDone! The best network ({best:.1f} bricks per game) was saved to {C.MODEL_PATH}")
    print("Next steps:  python -m src.evaluate   then   python -m src.export_web")


if __name__ == "__main__":
    main()
