"""
Train the driver with reinforcement learning (DQN).

    python -m src.train                  # full training (a few minutes)
    python -m src.train --steps 5000     # short training for a check

16 cars drive in parallel on a random track (which changes every 500 steps), and they all use the same network.
Every experience (state, action, reward, next state) is stored in memory, and the network learns from random samples of it.
Every 2,500 steps the network is tested on 10 fixed validation tracks, and the best one is kept.

Outputs:
    models/driver.pt               - the best network (by validation) - the one the app uses
    models/last.pt                 - the network at the end of training
    models/snapshots/step_*.pt     - the network at early stages (for comparison in the app)
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
from .agent import DQNAgent, ReplayBuffer, save_checkpoint
from .env import CarEnv
from .track import random_track
from .validation import make_tracks, validation_score


def plot_history(h, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    ax = axes[0, 0]
    ax.plot(h["step"], h["ep_return"], color="#4c6ef5")
    ax.set_title("Average episode reward (last 100 episodes)")
    ax = axes[0, 1]
    ax.plot(h["val_step"], h["val_score"], "o-", color="#2f9e44")
    ax.set_title("Validation: laps completed in 30 s (avg of 10 tracks)")
    ax = axes[1, 0]
    ax.plot(h["step"], h["loss"], color="#e8590c")
    ax.set_title("TD loss (Huber)")
    ax.set_yscale("log")
    ax = axes[1, 1]
    ax.plot(h["step"], h["epsilon"], color="#868e96", label="epsilon (exploration)")
    ax.plot(h["step"], h["crash_rate"], color="#e03131", label="crash rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Exploration & crashes")
    for a in axes.flat:
        a.set_xlabel("training step")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    T = dict(C.TRAIN)
    ap = argparse.ArgumentParser(description="Train a DQN driver")
    ap.add_argument("--steps", type=int, default=T["total_steps"])
    ap.add_argument("--seed", type=int, default=T["seed"])
    args = ap.parse_args()
    T["total_steps"], T["seed"] = args.steps, args.seed

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    rng = np.random.default_rng(T["seed"])
    agent = DQNAgent(T["hidden"], T["lr"], T["gamma"], T["tau"], T["grad_clip"], T["seed"])
    buffer = ReplayBuffer(T["buffer_size"], C.OBS_SIZE, rng)
    env = CarEnv(random_track(rng), n_cars=T["n_envs"], rng=rng)
    obs = env.reset()
    val_tracks = make_tracks(T["eval_tracks"], C.VALIDATION_SEED)
    greedy = lambda o: agent.act(o, 0.0)

    hist = {k: [] for k in ["step", "ep_return", "loss", "epsilon", "crash_rate", "val_step", "val_score"]}
    returns, crashes, losses = [], [], []
    best_score, updates, t0 = -1.0, 0, time.time()
    total = T["total_steps"]
    print(f"Starting training: {total:,} steps x {T['n_envs']} cars = {total * T['n_envs']:,} experiences")

    for step in range(1, total + 1):
        if step % T["new_track_every"] == 0:
            # episodes cut short by a track change count as "no crash"
            returns.extend(env.ep_return.tolist())
            crashes.extend([False] * env.n)
            env.set_track(random_track(rng))
            obs = env.observe()

        eps = max(T["eps_end"], T["eps_start"] - (T["eps_start"] - T["eps_end"]) * step
                  / (T["eps_decay_fraction"] * total))
        actions = agent.act(obs, eps)
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        # on "truncated" (time ran out) we don't mark an end - the future still exists
        buffer.add(obs, actions, rewards, next_obs, terminated.astype(np.float32))
        if env.done.any():
            returns.extend(info["ep_return"][env.done].tolist())
            crashes.extend(info["crashed"][env.done].tolist())
        obs = env.auto_reset()

        # a "snapshot" of the brain before this step's update (to compare learning stages)
        if step in T["snapshot_steps"]:
            save_checkpoint(agent.checkpoint(step=step, updates=updates),
                            C.MODELS_DIR / "snapshots" / f"step_{step:05d}.pt")

        if step >= T["learning_starts"]:
            loss, _ = agent.learn(buffer.sample(T["batch_size"]))
            losses.append(loss)
            updates += 1

        if step % T["eval_every"] == 0 or step == total:
            score = validation_score(greedy, val_tracks, T["eval_steps"])
            hist["val_step"].append(step)
            hist["val_score"].append(score)
            star = ""
            if score > best_score:
                best_score, star = score, "  * best network so far - saved"
                save_checkpoint(agent.checkpoint(step=step, val_score=score), C.MODEL_PATH)
            print(f"    validation: {score:.2f} laps in 30 seconds{star}")

        if step % 500 == 0:
            r = float(np.mean(returns[-100:])) if returns else 0.0
            cr = float(np.mean(crashes[-100:])) if crashes else 0.0
            l = float(np.mean(losses[-500:])) if losses else float("nan")
            for k, v in zip(["step", "ep_return", "loss", "epsilon", "crash_rate"], [step, r, l, eps, cr]):
                hist[k].append(v)
            if step % 1000 == 0:
                print(f"step {step:6,}/{total:,} | epsilon {eps:.2f} | average reward {r:7.1f} | "
                      f"crashes {cr*100:4.0f}% | loss {l:.4f} | {time.time() - t0:5.0f} s")

    save_checkpoint(agent.checkpoint(step=total), C.MODELS_DIR / "last.pt")
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.REPORTS_DIR / "history.json", "w") as f:
        json.dump(hist, f)
    plot_history(hist, C.REPORTS_DIR / "training_curves.png")
    print(f"\nDone! Best validation score: {best_score:.2f} laps in 30 seconds")
    print(f"Network saved to {C.MODEL_PATH}")
    print("Next step:  python -m src.evaluate   then   python -m app.server")


if __name__ == "__main__":
    main()
