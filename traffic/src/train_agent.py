"""
Train network 2: the agent that manages the traffic light (reinforcement learning, DQN).

    python -m src.train_agent                  # the agent sees the exact count from the simulation
    python -m src.train_agent --noise auto     # ...or a "noisy" count like the camera's (an experiment)

16 intersections work in parallel, each with a different demand scenario (which also changes during the episode).
Every 1,000 decisions the agent is tested on 6 fixed validation scenarios and the best one is kept.

Outputs:  models/traffic_dqn.pt,  reports/agent_history.json,  reports/agent_curves.png
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
from .agent import DQNAgent, ReplayBuffer
from .traffic import TrafficEnv, fixed_time, max_queue

VAL_SEEDS = (101, 102, 103, 104, 105, 106)


def run_episodes(policy, seeds, noise_sd=0.0, noise_seed=0):
    """
    Runs whole episodes (20 minutes) and returns the average waiting time per vehicle.
    policy(env, counts) -> actions. counts = the count the controller "sees" (possibly with noise).
    """
    env = TrafficEnv(len(seeds), seed=seeds[0])
    rng = np.random.default_rng(noise_seed)
    while not (env.t >= C.EPISODE_SECONDS).all():
        counts = env.counts().astype(float)
        if noise_sd > 0:
            counts = np.maximum(0, counts + rng.normal(0, noise_sd, counts.shape))
        env.step(policy(env, counts))
    return env.metrics()


def plot_history(h, base, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    a1.plot(h["step"], h["reward"], color="#3d5a80")
    a1.set_title("Average reward per decision (training)")
    a1.set_xlabel("decision step")
    a2.plot(h["val_step"], h["val_wait"], "o-", color="#2a9d8f", label="DQN agent")
    a2.axhline(base["fixed"], color="#e76f51", ls="--", label=f"fixed-time ({base['fixed']:.1f}s)")
    a2.axhline(base["max_queue"], color="#e9c46a", ls="--", label=f"max-queue rule ({base['max_queue']:.1f}s)")
    a2.set_title("Validation: average wait per vehicle (s) - lower is better")
    a2.set_xlabel("decision step")
    a2.legend()
    for a in (a1, a2):
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    T = dict(C.TRAIN)
    ap = argparse.ArgumentParser(description="Train the traffic light agent")
    ap.add_argument("--steps", type=int, default=T["total_decisions"])
    ap.add_argument("--seed", type=int, default=T["seed"])
    ap.add_argument("--noise", default="0", help="standard deviation of the counting noise, or auto (from the camera network's error)")
    args = ap.parse_args()
    if args.noise == "auto":
        path = C.REPORTS_DIR / "counter_metrics.json"
        noise = json.load(open(path))["residual_sd"] if path.exists() else 1.0
    else:
        noise = float(args.noise)
    total = args.steps
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    rng = np.random.default_rng(args.seed)
    agent = DQNAgent(T["hidden"], T["lr"], T["gamma"], T["tau"], args.seed)
    buf = ReplayBuffer(T["buffer_size"], rng)
    env = TrafficEnv(T["n_envs"], seed=args.seed)

    def noisy_obs():
        c = env.counts().astype(float)
        if noise > 0:
            c = np.maximum(0, c + rng.normal(0, noise, c.shape))
        return env.observe(c)

    greedy = lambda e, c: agent.act(e.observe(c))
    print("Measuring the comparison traffic lights on the validation scenarios...")
    base = {"fixed": float(run_episodes(lambda e, c: fixed_time(e), VAL_SEEDS, noise)["avg_wait"].mean()),
            "max_queue": float(run_episodes(lambda e, c: max_queue(e, c), VAL_SEEDS, noise)["avg_wait"].mean())}
    print(f"  fixed light: {base['fixed']:.1f} s | 'longest queue' light: {base['max_queue']:.1f} s (average wait per vehicle)")
    if noise:
        print(f"  training with counting noise: standard deviation {noise:.2f} cars")

    hist = {k: [] for k in ["step", "reward", "val_step", "val_wait"]}
    obs, rewards, best, t0 = noisy_obs(), [], float("inf"), time.time()
    for step in range(1, total + 1):
        eps = max(T["eps_end"], T["eps_start"] - (T["eps_start"] - T["eps_end"]) * step / (T["eps_decay_fraction"] * total))
        a = agent.act(obs, eps)
        r, done = env.step(a)
        nobs = noisy_obs()
        buf.add(obs, a, r, nobs)
        rewards.append(float(r.mean()))
        if done.any():
            env.reset(done)
            nobs = noisy_obs()
        obs = nobs
        if step >= T["learning_starts"]:
            agent.learn(buf.sample(T["batch_size"]))
        if step % 250 == 0:
            hist["step"].append(step)
            hist["reward"].append(float(np.mean(rewards[-250:])))
        if step % T["eval_every"] == 0 or step == total:
            w = float(run_episodes(greedy, VAL_SEEDS, noise)["avg_wait"].mean())
            hist["val_step"].append(step)
            hist["val_wait"].append(w)
            star = ""
            if w < best:
                best, star = w, "  * saved"
                C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(agent.checkpoint(step=step, val_wait=w, noise=float(noise), baselines=base), C.AGENT_PATH)
            print(f"step {step:6,}/{total:,} | epsilon {eps:.2f} | reward {np.mean(rewards[-1000:]):6.2f} | "
                  f"validation: average wait {w:5.1f} s{star} | {time.time() - t0:4.0f} s")
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.REPORTS_DIR / "agent_history.json", "w") as f:
        json.dump({**hist, "baselines": base}, f)
    plot_history(hist, base, C.REPORTS_DIR / "agent_curves.png")
    print(f"\nDone! Best average wait in validation: {best:.1f} s "
          f"(fixed light: {base['fixed']:.1f}, 'longest queue': {base['max_queue']:.1f})")
    print("Next step:  python -m src.evaluate")


if __name__ == "__main__":
    main()
