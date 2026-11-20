"""
Training the pilot with PPO.

    python -m src.train                    # full training
    python -m src.train --steps 1000000    # shorter

64 ships fly in parallel, each in its own maze from a pool of 1,500 random training mazes (3x3x3 and
4x4x4 rooms). Half of the flights start at a random point in the maze instead of the entrance, so the ship
sees every kind of corner early. All the ships share the same networks.
Every 20 updates the network flies the 32 validation mazes (never used in training), and the best one is kept.

Outputs:
    models/pilot_ppo.pt              the best network - the one the app uses
    models/snapshots/upd_*.pt        the "brain" at early stages (for the learning stages in the app)
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
from .flights import fly, summary, validation_mazes
from .ppo import ActorCritic, RunningStat, gaussian_log_prob, ppo_update, save
from .ship import ShipEnv, training_pool


def plot_history(h, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].plot(h["steps"], h["ep_return"], color="#3d5a80")
    axes[0].set_title("Average flight return (training)")
    axes[1].plot(h["steps"], np.array(h["exit"]) * 100, color="#2a9d8f", label="reached the exit (%)")
    axes[1].plot(h["steps"], np.array(h["crash"]) * 100, color="#e63946", label="crashed (%)")
    axes[1].plot(h["steps"], np.array(h["std"]) * 100, color="#8d99ae", label="exploration noise (std x100)")
    axes[1].set_title("Training flights")
    axes[1].legend(fontsize=8)
    axes[2].plot(h["val_steps"], np.array(h["val_exit"]) * 100, "o-", ms=3, color="#2a9d8f", label="exit (%)")
    axes[2].plot(h["val_steps"], np.array(h["val_crash"]) * 100, "o-", ms=3, color="#e63946", label="crash (%)")
    ax2 = axes[2].twinx()
    ax2.plot(h["val_steps"], h["val_speed"], "s--", ms=3, color="#6a4c93", label="mean speed (m/s)")
    ax2.set_ylabel("m/s")
    axes[2].set_title("Validation: 32 mazes never seen in training")
    axes[2].legend(fontsize=8, loc="center left")
    ax2.legend(fontsize=8, loc="center right")
    for a in axes:
        a.set_xlabel("environment steps (millions)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    P = dict(C.PPO)
    ap = argparse.ArgumentParser(description="Train the maze pilot with PPO")
    ap.add_argument("--steps", type=int, default=P["total_steps"])
    ap.add_argument("--seed", type=int, default=P["seed"])
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    rng = np.random.default_rng(args.seed)
    N, T = P["n_envs"], P["n_steps"]
    n_updates = max(1, args.steps // (N * T))

    print(f"Generating {C.TRAIN_POOL:,} training mazes...", flush=True)
    t0 = time.time()
    training_pool(C.TRAIN_SIZES)
    val_mazes = validation_mazes()
    print(f"  done in {time.time() - t0:.0f} s")

    model = ActorCritic(P["hidden"], P["log_std_init"])
    opt = torch.optim.Adam(model.parameters(), lr=P["lr"], eps=1e-5)
    env = ShipEnv(N, seed=args.seed)
    obs = env.reset()
    obs_rms, ret_rms = RunningStat(C.OBS_SIZE), RunningStat(())
    ret_acc = np.zeros(N)
    eprets, exits, crashes = [], [], []
    h = {"steps": [], "ep_return": [], "exit": [], "crash": [], "std": [],
         "val_steps": [], "val_exit": [], "val_crash": [], "val_speed": [], "val_time": []}
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
                mean = model.actor(on)
                a = mean + model.log_std.exp() * torch.randn_like(mean)        # sample: exploration
                logp = gaussian_log_prob(a, mean, model.log_std)
                v = model.value(on)
            nobs, r, term, trunc, info = env.step(a.numpy())
            # normalize the reward by the standard deviation of the return (stabilizes the Critic's learning)
            ret_acc = ret_acc * P["gamma"] + r
            ret_rms.update(ret_acc)
            rs = np.clip(r / np.sqrt(ret_rms.var + 1e-8), -10, 10)
            if trunc.any():   # time ran out (not a crash, not the exit): add the value of the last state
                with torch.no_grad():
                    vb = model.value(model.norm(torch.as_tensor(nobs))).numpy()
                rs = rs + np.where(trunc, P["gamma"] * vb, 0)
            done = term | trunc
            ret_acc[done] = 0
            if done.any():
                eprets.extend(info["ep_return"][done].tolist())
                exits.extend(info["reached"][done].tolist())
                crashes.extend(info["crashed"][done].tolist())
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
        batch = tuple(torch.as_tensor(x.reshape((N * T,) + x.shape[2:])) for x in (O, A, LP, adv, ret))

        # ---------- 3. update ----------
        ppo_update(model, opt, batch, P)

        steps_m = u * N * T / 1e6
        if u in C.SNAPSHOT_UPDATES:
            save(model, C.MODELS_DIR / "snapshots" / f"upd_{u:04d}.pt", update=u, steps=u * N * T)
        if u % 5 == 0:
            h["steps"].append(steps_m)
            h["ep_return"].append(float(np.mean(eprets[-300:])) if eprets else 0.0)
            h["exit"].append(float(np.mean(exits[-300:])) if exits else 0.0)
            h["crash"].append(float(np.mean(crashes[-300:])) if crashes else 0.0)
            h["std"].append(float(model.log_std.exp().mean()))
        if u % P["eval_every"] == 0 or u == n_updates:
            with torch.no_grad():
                pilot = lambda o: model.actor(model.norm(torch.as_tensor(o))).numpy()
                res = summary(fly(pilot, val_mazes))
            # the best = most exits and fewest crashes; speed breaks ties
            score = res["exit"] - res["crash"] + 0.002 * res["speed"]
            h["val_steps"].append(steps_m)
            h["val_exit"].append(res["exit"])
            h["val_crash"].append(res["crash"])
            h["val_speed"].append(res["speed"])
            h["val_time"].append(res["time"])
            star = ""
            if score > best:
                best, star = score, "  * saved"
                save(model, C.MODEL_PATH, update=u, steps=u * N * T, val_exit=res["exit"], val_crash=res["crash"],
                     val_speed=res["speed"])
            t_txt = f"{res['time']:5.1f} s" if res["time"] is not None else "   -  "
            print(f"update {u:4d}/{n_updates} | {steps_m:5.2f}M steps | return {np.mean(eprets[-300:]):6.1f} | "
                  f"training: exit {np.mean(exits[-300:]) * 100:3.0f}% crash {np.mean(crashes[-300:]) * 100:3.0f}% | "
                  f"validation: exit {res['exit'] * 100:3.0f}% crash {res['crash'] * 100:3.0f}% "
                  f"speed {res['speed']:4.1f} m/s time {t_txt}{star} | {time.time() - t0:4.0f} s", flush=True)
            C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(C.REPORTS_DIR / "history.json", "w") as f:
                json.dump(h, f)
            plot_history(h, C.REPORTS_DIR / "training_curves.png")
    print(f"\nDone! The best network was saved to {C.MODEL_PATH}")
    print("Next step:  python -m src.evaluate   then   python -m app.server")


if __name__ == "__main__":
    main()
