"""
Reinforcement learning for the runner.

    python -m src.train_dqn                  # DQN from scratch: learns only from its own trial and error
    python -m src.train_dqn --combo          # "combo" (DQfD): starts from the recordings, then keeps practicing
    python -m src.train_dqn --steps 5000     # a short run, to check that everything works

32 runners play in parallel. Each run starts at a random speed (as if 0 to 2 minutes had passed), so the
network also practices at top speed, and is cut after 2.5 minutes.
Every 2,500 steps the network plays the 16 validation tracks, and the best version is saved.
The learning rate goes down during training, which makes the late part of training steadier.

The combo version uses half the steps of plain DQN: one of the questions of the project is whether
starting from human play saves practice.

Outputs:  models/dqn.pt  or  models/combo.pt,  reports/<name>_history.json,  reports/<name>_curves.png
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .dqn import DemoBuffer, DQNAgent, ReplayBuffer
from .game import MIRROR_ACTION, RunnerGame, mirror_obs
from .nets import count_parameters, pick_device, save_model
from .players import ModelPlayer, play
from .recordings import build_dataset, load_recordings, pick_source, summary


def load_demos(source, rng):
    source = pick_source(source)
    recs = load_recordings(source)
    if not recs:
        raise SystemExit("The combo needs recordings. Play the app, or:  python -m src.make_bot_demos")
    if source == "bot":
        print("[!] Using the BOT's practice recordings as the demonstrations, not people's.")
    s = summary(recs)
    print(f"Demonstrations ({source}): {s['runs']} runs, {s['minutes']:.1f} minutes of play")
    D = build_dataset(recs)
    demo = {
        "obs": np.concatenate([D["obs"], mirror_obs(D["obs"])]),
        "next_obs": np.concatenate([D["next_obs"], mirror_obs(D["next_obs"])]),
        "actions": np.concatenate([D["actions"], MIRROR_ACTION[D["actions"]]]),
        "rewards": np.concatenate([D["rewards"], D["rewards"]]),
        "dones": np.concatenate([D["dones"], D["dones"]]),
    }
    return DemoBuffer(demo, rng), source, s


def validate(agent):
    dist, crashes, _ = play(ModelPlayer(net=agent.q, kind="q", device=agent.device), C.VAL_SEEDS, max_ticks=2400)
    return float(dist.mean()), np.bincount(crashes, minlength=4)[1:].tolist()


def plot_history(hist, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    st = [h["step"] for h in hist]
    ax[0].plot(st, [h["val_distance"] for h in hist], marker="o", color="#2e86ab")
    ax[0].set_title(f"{title}: distance on the 16 validation tracks (2 min)")
    ax[0].set_xlabel("training steps (x32 runners)")
    ax[0].set_ylabel("meters")
    ax[1].plot(st, [h.get("train_distance", 0) for h in hist], color="#e4572e")
    ax[1].set_title("Average distance of training runs (with exploration)")
    ax[1].set_xlabel("training steps (x32 runners)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train the runner with reinforcement learning")
    ap.add_argument("--combo", action="store_true", help="start from the recordings (DQfD)")
    ap.add_argument("--source", default="auto", choices=["auto", "human", "bot", "all"],
                    help="which recordings the combo learns from")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=C.DQN["seed"])
    args = ap.parse_args()

    P = dict(C.DQN)
    name, path = ("combo", C.COMBO_PATH) if args.combo else ("dqn", C.DQN_PATH)
    if args.combo:
        P.update(total_steps=C.COMBO["total_steps"], eps_start=C.COMBO["eps_start"])
    if args.steps:
        P["total_steps"] = args.steps
    total, E, B = P["total_steps"], P["n_envs"], P["batch_size"]
    rng = np.random.default_rng(args.seed)
    device = pick_device()
    demo, n_demo, extra = None, 0, {}
    if args.combo:
        demo, source, s = load_demos(args.source, rng)
        extra = {"source": source, "demo_minutes": s["minutes"]}
    agent = DQNAgent(device, P["lr"], P["gamma"], P["tau"], seed=args.seed,
                     class_weights=demo.class_weights if demo else None)
    print(f"Device: {device} | network parameters: {count_parameters(agent.q):,}")
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    hist, best = [], -1.0

    if args.combo:
        n_demo = int(B * C.COMBO["demo_fraction"])
        print(f"Step 1: learning from the recordings only ({C.COMBO['pretrain_updates']:,} updates)...")
        t0 = time.time()
        for u in range(1, C.COMBO["pretrain_updates"] + 1):
            td, ml, _ = agent.learn(None, demo.sample(B), C.COMBO["margin"], C.COMBO["margin_weight"])
            if u % 1000 == 0:
                print(f"  update {u:5,} | TD loss {td:.4f} | margin loss {ml:.3f} | {time.time() - t0:3.0f}s")
        d, cr = validate(agent)
        hist.append({"step": 0, "val_distance": d, "crashes": cr})
        best = d
        save_model(agent.q, path, "q", variant=name, steps=0, val_distance=d, **extra)
        print(f"  after the recordings only: {d:.0f} m on the validation tracks (crashes {cr})\n")
        print(f"Step 2: practice ({total:,} steps x {E} runners)")
    else:
        print(f"Training: {total:,} steps x {E} runners = {total * E / 1e6:.1f} million experiences")

    buf = ReplayBuffer(P["buffer_size"], rng)
    game = RunnerGame(E)
    game.reset(rng.integers(0, 2**31, E), rng.integers(0, 2400, E))
    ep_t = np.zeros(E, np.int64)
    obs = game.observe()
    finished, losses = [], []
    t0 = time.time()
    for step in range(1, total + 1):
        frac = min(1.0, step / (P["eps_fraction"] * total))
        eps = P["eps_start"] + frac * (P["eps_end"] - P["eps_start"])
        agent.set_lr(P["lr"] * (1 - 0.9 * step / total))          # learning rate goes down to 10% (steadier)
        actions = agent.act(obs, eps)
        crashed, pressed = game.step(actions)
        ep_t += 1
        next_obs = game.observe()
        rewards = C.R_ALIVE + C.R_ACTION * pressed + (C.R_CRASH - C.R_ALIVE) * crashed
        buf.add(obs, actions, rewards.astype(np.float32), next_obs, crashed.astype(np.float32))

        # finished runs: a crash, or the time limit (not a crash - so no "done" in the Bellman equation)
        ended = crashed | (ep_t >= C.TRAIN_EPISODE_TICKS)
        if ended.any():
            idx = np.flatnonzero(ended)
            finished += (game.pz[idx] / 100).tolist()
            game.reset(rng.integers(0, 2**31, len(idx)), rng.integers(0, 2400, len(idx)), idx)
            ep_t[idx] = 0
            next_obs = game.observe()
        obs = next_obs

        if len(buf) >= P["learning_starts"] and step % P["train_every"] == 0:
            if demo is not None:
                out = agent.learn(buf.sample(B - n_demo), demo.sample(n_demo),
                                  C.COMBO["margin"], C.COMBO["margin_weight"])
            else:
                out = agent.learn(buf.sample(B))
            losses.append(out[0])

        if step % P["eval_every"] == 0 or step == total:
            d, cr = validate(agent)
            h = {"step": step, "val_distance": d, "crashes": cr, "epsilon": eps,
                 "train_distance": float(np.mean(finished[-200:])) if finished else 0.0,
                 "td_loss": float(np.mean(losses[-500:])) if losses else 0.0}
            hist.append(h)
            mark = ""
            if d > best:
                best, mark = d, "  * saved"
                save_model(agent.q, path, "q", variant=name, steps=step, val_distance=d, **extra)
            print(f"step {step:6,}/{total:,} | epsilon {eps:.2f} | training runs {h['train_distance']:5.0f} m | "
                  f"loss {h['td_loss']:.4f} | validation {d:5.0f} m (crashes: hurdle/bar/train {cr}) | "
                  f"{time.time() - t0:4.0f}s{mark}")

    (C.REPORTS_DIR / f"{name}_history.json").write_text(json.dumps(hist, indent=1))
    plot_history(hist, "Combo" if args.combo else "DQN", C.REPORTS_DIR / f"{name}_curves.png")
    print(f"\nDone! Best validation distance: {best:.0f} m  (saved to {path})")
    print("Next step:  python -m src.evaluate")


if __name__ == "__main__":
    main()
