"""
Training the drone with PPO.

    python -m src.train                    # full training (a few minutes)
    python -m src.train --steps 1000000    # shorter

128 drones fly in parallel, each with random wind, payload and targets, and all share the same networks.
Every 40 updates the network is tested on validation scenarios (calm, wind, heavy payload, storm) and the best one is kept.

Outputs:
    models/drone_ppo.pt              the best network - the one the app uses
    models/snapshots/upd_*.pt        the "brain" at early stages (for comparison in the app)
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
from .drone import DroneEnv
from .ppo import ActorCritic, RunningStat, ppo_update
from .scenarios import run

VAL_CONDITIONS = ["calm", "wind", "heavy", "storm"]
SNAPSHOT_UPDATES = (1, 20, 40, 70)


def save(model, path, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden": C.PPO["hidden"], **extra}, path)


def plot_history(h, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].plot(h["steps"], h["ep_return"], color="#3d5a80")
    axes[0].set_title("Average episode return (training)")
    axes[1].plot(h["steps"], np.array(h["crash"]) * 100, color="#e63946", label="crashes (%)")
    axes[1].plot(h["steps"], np.array(h["std"]) * 100, color="#8d99ae", label="policy std (x100)")
    axes[1].set_title("Crashes & exploration noise")
    axes[1].legend()
    for c in VAL_CONDITIONS:
        axes[2].plot(h["val_steps"], np.array(h["val"][c]) * 100, "o-", ms=3, label=c)
    axes[2].set_title("Validation: success rate (%)")
    axes[2].legend()
    for a in axes:
        a.set_xlabel("environment steps (millions)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    P = dict(C.PPO)
    ap = argparse.ArgumentParser(description="Train a drone with PPO")
    ap.add_argument("--steps", type=int, default=P["total_steps"])
    ap.add_argument("--seed", type=int, default=P["seed"])
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    rng = np.random.default_rng(args.seed)
    N, T = P["n_envs"], P["n_steps"]
    n_updates = args.steps // (N * T)

    model = ActorCritic(P["hidden"], P["log_std_init"])
    opt = torch.optim.Adam(model.parameters(), lr=P["lr"], eps=1e-5)
    env = DroneEnv(N, seed=args.seed)
    obs = env.reset()
    obs_rms, ret_rms = RunningStat(C.OBS_SIZE), RunningStat(())
    ret_acc = np.zeros(N)
    eprets, crashes = [], []
    h = {"steps": [], "ep_return": [], "crash": [], "std": [], "val_steps": [], "val": {c: [] for c in VAL_CONDITIONS}}
    best, t0 = -1e9, time.time()
    print(f"Starting: {n_updates} updates x {N * T:,} samples = {n_updates * N * T / 1e6:.1f} million steps")

    def sync_norm():
        model.obs_mean.copy_(torch.as_tensor(obs_rms.mean, dtype=torch.float32))
        model.obs_var.copy_(torch.as_tensor(obs_rms.var, dtype=torch.float32))

    for u in range(1, n_updates + 1):
        for g in opt.param_groups:                       # learning rate decreases linearly
            g["lr"] = P["lr"] * (1 - (u - 1) / n_updates)
        O = np.zeros((T, N, C.OBS_SIZE), np.float32)
        A = np.zeros((T, N, C.ACT_SIZE), np.float32)
        LP, V, R, D = (np.zeros((T, N), np.float32) for _ in range(4))
        # ---------- 1. collecting experience ----------
        for t in range(T):
            obs_rms.update(obs)
            sync_norm()
            with torch.no_grad():
                on = model.norm(torch.as_tensor(obs))
                dist = model.dist(on)
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1)
                v = model.value(on)
            nobs, r, term, trunc, info = env.step(a.numpy())
            # normalize the reward by the standard deviation of the return (stabilizes the Critic's learning)
            ret_acc = ret_acc * P["gamma"] + r
            ret_rms.update(ret_acc)
            rs = np.clip(r / np.sqrt(ret_rms.var + 1e-8), -10, 10)
            if trunc.any():   # time ran out (not a crash): add the value of the last state
                with torch.no_grad():
                    vb = model.value(model.norm(torch.as_tensor(nobs))).numpy()
                rs = rs + np.where(trunc, P["gamma"] * vb, 0)
            done = term | trunc
            ret_acc[done] = 0
            if done.any():
                eprets.extend(info["ep_return"][done].tolist())
                crashes.extend(term[done].tolist())
            O[t], A[t], LP[t], V[t], R[t], D[t] = on.numpy(), a.numpy(), logp.numpy(), v.numpy(), rs, done
            obs = env.auto_reset()

        # ---------- 2. GAE ----------
        with torch.no_grad():
            last_v = model.value(model.norm(torch.as_tensor(obs))).numpy()
        adv = np.zeros((T, N), np.float32)
        gae = 0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            nonterm = 1 - D[t]
            delta = R[t] + P["gamma"] * nv * nonterm - V[t]
            gae = delta + P["gamma"] * P["gae_lambda"] * nonterm * gae
            adv[t] = gae
        ret = adv + V
        batch = tuple(torch.as_tensor(x.reshape((N * T,) + x.shape[2:]))
                      for x in (O, A, LP, adv, ret))

        # ---------- 3. update ----------
        ppo_update(model, opt, batch, P)

        steps_m = u * N * T / 1e6
        if u in SNAPSHOT_UPDATES:
            save(model, C.MODELS_DIR / "snapshots" / f"upd_{u:04d}.pt", update=u, steps=u * N * T)
        if u % 10 == 0:
            h["steps"].append(steps_m)
            h["ep_return"].append(float(np.mean(eprets[-300:])) if eprets else 0.0)
            h["crash"].append(float(np.mean(crashes[-300:])) if crashes else 0.0)
            h["std"].append(float(model.log_std.exp().mean()))
        if u % P["eval_every"] == 0 or u == n_updates:
            with torch.no_grad():
                pol = lambda o, e: model.actor(model.norm(torch.as_tensor(o))).numpy()
                res = {c: run(pol, c, n=64, seed=99) for c in VAL_CONDITIONS}
            score = float(np.mean([r["success"] - r["crashes"] for r in res.values()]))
            h["val_steps"].append(steps_m)
            for c in VAL_CONDITIONS:
                h["val"][c].append(res[c]["success"])
            star = ""
            if score > best:
                best, star = score, "  * saved"
                save(model, C.MODEL_PATH, update=u, steps=u * N * T, val_score=score)
            print(f"update {u:4d}/{n_updates} | {steps_m:5.2f}M steps | return {np.mean(eprets[-300:]):6.1f} | "
                  f"crashes {np.mean(crashes[-300:]) * 100:4.1f}% | validation (success/crash %): "
                  + " ".join(f"{c} {res[c]['success'] * 100:.0f}/{res[c]['crashes'] * 100:.0f}" for c in VAL_CONDITIONS)
                  + f"{star} | {time.time() - t0:4.0f} s")
            C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(C.REPORTS_DIR / "history.json", "w") as f:
                json.dump(h, f)
            plot_history(h, C.REPORTS_DIR / "training_curves.png")
    print(f"\nDone! The best network was saved to {C.MODEL_PATH}")
    print("Next step:  python -m src.evaluate   then   python -m app.server")


if __name__ == "__main__":
    main()
