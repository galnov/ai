"""
Training the air hockey player with PPO and self-play.

    python -m src.train                    # full training (about 12 million decisions)
    python -m src.train --steps 3000000    # shorter

64 tables play at the same time. The network always plays the bottom side (player 0) - the top side sees the
table turned around, so for the network both sides look the same. Its opponent on each table is chosen again
after every point:
  * the hand-written bot (src/bot.py), at a random speed - most of the time at the start (90%), later 20%
  * an earlier version of the network itself, from a "pool" of up to 12 saved versions - so it keeps meeting
    different styles and can't win by learning one trick against one opponent
  * the current network (40% of the network opponents) - so it always plays someone as good as itself

Every 25 updates it plays 400 points against the full bot and against its first saved stage.

Outputs:
    models/hockey_ppo.pt             the best version against the bot in the second half of training - the web page's
                                     "Champion". (Self-play goes up and down a little, so the last one isn't always the best)
    models/hockey_last.pt            the network at the end of training
    models/snapshots/upd_*.pt        the network at early stages (the "learning stages" in the web page)
    reports/history.json, reports/training_curves.png
"""
import argparse
import copy
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config as C
from .bot import bot_actions
from .physics import HockeyEnv, play_points
from .ppo import ActorCritic, RunningStat, ppo_update, save

BOT, LATEST = -1, -2


def plot_history(h, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    axes[0].plot(h["steps"], h["ep_return"], color="#3d5a80")
    axes[0].set_title("Average reward per point (training)")
    axes[1].plot(h["steps"], np.array(h["vs_bot_train"]) * 100, color="#e76f51", label="vs the bot")
    axes[1].plot(h["steps"], np.array(h["vs_self_train"]) * 100, color="#2a9d8f", label="vs itself / earlier versions")
    axes[1].plot(h["steps"], np.array(h["bot_share"]) * 100, "--", color="#8d99ae", label="share of bot opponents")
    axes[1].set_title("Training: share of goals scored (%)")
    axes[1].legend()
    axes[2].plot(h["eval_steps"], np.array(h["eval_bot"]) * 100, "o-", ms=3, color="#e76f51", label="vs the full bot")
    axes[2].plot(h["eval_steps"], np.array(h["eval_first"]) * 100, "o-", ms=3, color="#264653",
                 label="vs its first stage")
    axes[2].axhline(50, color="#aaa", lw=0.8)
    axes[2].set_title("Evaluation: share of goals scored (%)")
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    for a in axes:
        a.set_xlabel("decisions (millions)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    P = dict(C.PPO)
    ap = argparse.ArgumentParser(description="Train an air hockey player with PPO + self-play")
    ap.add_argument("--steps", type=int, default=P["total_steps"])
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    # the CPU: the networks are small, and most of the time goes into the simulation (NumPy) anyway
    dev = torch.device("cpu")
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    N, T = P["n_envs"], P["n_steps"]
    n_updates = max(1, args.steps // (N * T))

    model = ActorCritic(P["hidden"], P["log_std_init"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=P["lr"], eps=1e-5)
    env = HockeyEnv(N, seed=args.seed)
    obs_rms, ret_rms = RunningStat(C.OBS_SIZE), RunningStat(())
    ret_acc, ep_acc = np.zeros(N), np.zeros(N)
    eprets = []
    pool = []                                   # frozen earlier versions (on the CPU)
    first_stage = None
    tally = {"bot": [0, 0], "self": [0, 0]}     # goals for / against in training, by opponent type
    h = {k: [] for k in ("steps", "ep_return", "vs_bot_train", "vs_self_train", "bot_share", "std",
                         "eval_steps", "eval_bot", "eval_first")}
    bot_skill = rng.uniform(0.6, 1.0, N)
    best = -1.0
    t0 = time.time()
    print(f"Starting on {dev}: {n_updates} updates x {N * T:,} decisions = {n_updates * N * T / 1e6:.1f} million")

    def bot_share(u):
        f = min(1.0, (u - 1) / max(1, 0.6 * n_updates))          # reaches the end value at 60% of training
        return C.BOT_SHARE_START + f * (C.BOT_SHARE_END - C.BOT_SHARE_START)

    def pick(k, u):
        """New opponents for k tables."""
        out = np.full(k, BOT)
        if not pool:
            return out
        r = rng.random(k)
        net = r >= bot_share(u)
        latest = rng.random(k) < C.LATEST_SHARE
        out[net & latest] = LATEST
        rest = net & ~latest
        out[rest] = rng.integers(0, len(pool), rest.sum())
        return out

    def sync_norm():
        model.obs_mean.copy_(torch.as_tensor(obs_rms.mean, dtype=torch.float32))
        model.obs_var.copy_(torch.as_tensor(obs_rms.var, dtype=torch.float32))

    opp = pick(N, 1)
    for u in range(1, n_updates + 1):
        for g in opt.param_groups:                       # learning rate decreases linearly
            g["lr"] = P["lr"] * (1 - (u - 1) / n_updates)
        latest_net = copy.deepcopy(model).cpu().eval()    # "the current network" as an opponent this round
        O = np.zeros((T, N, C.OBS_SIZE), np.float32)
        A = np.zeros((T, N, C.ACT_SIZE), np.float32)
        LP, V, R, D = (np.zeros((T, N), np.float32) for _ in range(4))
        # ---------- 1. collecting experience ----------
        for t in range(T):
            o = env.observe()
            me, them = o[:, 0].astype(np.float32), o[:, 1].astype(np.float32)
            obs_rms.update(me)
            sync_norm()
            with torch.no_grad():
                on = model.norm(torch.as_tensor(me, device=dev))
                dist = model.dist(on)
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1)
                v = model.value(on)
            # the opponents' moves
            a_opp = np.zeros((N, 2), np.float32)
            for k in np.unique(opp):
                idx = np.flatnonzero(opp == k)
                if k == BOT:
                    a_opp[idx] = bot_actions(them[idx]) * bot_skill[idx, None]
                else:
                    a_opp[idx] = (latest_net if k == LATEST else pool[k]).play(them[idx])
            a_np = a.cpu().numpy()
            r, goal, timeout, _ = env.step(np.stack([a_np, a_opp], 1))
            r0 = r[:, 0]
            # normalize the reward by the standard deviation of the return (stabilizes the Critic's learning)
            ret_acc = ret_acc * P["gamma"] + r0
            ret_rms.update(ret_acc)
            rs = np.clip(r0 / np.sqrt(ret_rms.var + 1e-8), -10, 10)
            ep_acc += r0
            done = (goal != 0) | timeout                 # every point is an episode
            for i in np.flatnonzero(goal != 0):
                kind = "bot" if opp[i] == BOT else "self"
                tally[kind][0 if goal[i] > 0 else 1] += 1
            if done.any():
                eprets.extend(ep_acc[done].tolist())
                ep_acc[done] = 0
                ret_acc[done] = 0
                opp[done] = pick(int(done.sum()), u)
                bot_skill[done] = rng.uniform(0.6, 1.0, int(done.sum()))
            O[t], A[t], LP[t], V[t], R[t], D[t] = (on.cpu().numpy(), a_np, logp.cpu().numpy(), v.cpu().numpy(),
                                                   rs, done)

        # ---------- 2. GAE ----------
        with torch.no_grad():
            last_v = model.value(model.norm(torch.as_tensor(env.observe()[:, 0], dtype=torch.float32,
                                                            device=dev))).cpu().numpy()
        adv = np.zeros((T, N), np.float32)
        gae = 0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            nonterm = 1 - D[t]
            delta = R[t] + P["gamma"] * nv * nonterm - V[t]
            gae = delta + P["gamma"] * P["lam"] * nonterm * gae
            adv[t] = gae
        ret = adv + V
        batch = tuple(torch.as_tensor(x.reshape((N * T,) + x.shape[2:]), device=dev)
                      for x in (O, A, LP, adv, ret))

        # ---------- 3. update ----------
        ppo_update(model, opt, batch, P)

        steps_m = u * N * T / 1e6
        if u in C.SNAPSHOT_UPDATES:
            save(model, C.SNAPSHOT_DIR / f"upd_{u:04d}.pt", update=u, steps=u * N * T)
        if u % C.POOL_EVERY == 0 or u == 1:
            pool.append(copy.deepcopy(model).cpu().eval())
            if first_stage is None:
                first_stage = pool[0]
            if len(pool) > C.POOL_SIZE:                  # keep the oldest one and the recent ones
                pool.pop(1)
                opp[opp >= 1] = np.maximum(opp[opp >= 1] - 1, 0)
        if u % 5 == 0:
            share = lambda k: tally[k][0] / max(1, sum(tally[k]))
            h["steps"].append(steps_m)
            h["ep_return"].append(float(np.mean(eprets[-2000:])) if eprets else 0.0)
            h["vs_bot_train"].append(share("bot"))
            h["vs_self_train"].append(share("self"))
            h["bot_share"].append(bot_share(u))
            h["std"].append(float(model.log_std.exp().mean()))
            tally = {"bot": [0, 0], "self": [0, 0]}
        if u % C.EVAL_EVERY == 0 or u == n_updates:
            me_play = copy.deepcopy(model).cpu().eval().play
            gb = play_points(me_play, bot_actions)
            gf = play_points(me_play, first_stage.play)
            sb, sf = gb[0] / max(1, gb[0] + gb[1]), gf[0] / max(1, gf[0] + gf[1])
            h["eval_steps"].append(steps_m)
            h["eval_bot"].append(sb)
            h["eval_first"].append(sf)
            save(model, C.LAST_PATH, update=u, steps=u * N * T, vs_bot=sb)
            star = ""
            second_half = u >= n_updates // 2
            if not second_half or sb > best:             # in the first half: always the latest
                if second_half:
                    best, star = sb, "  * best"
                save(model, C.MODEL_PATH, update=u, steps=u * N * T, vs_bot=sb)
            rpp = float(np.mean(eprets[-2000:])) if eprets else 0.0
            print(f"update {u:4d}/{n_updates} | {steps_m:5.2f}M | reward/point {rpp:6.2f} | "
                  f"vs bot {gb[0]}:{gb[1]} ({gb[2]} replays) | vs first stage {gf[0]}:{gf[1]} | "
                  f"bot opponents {bot_share(u) * 100:.0f}% | std {float(model.log_std.exp().mean()):.2f} | "
                  f"{time.time() - t0:4.0f} s{star}")
            C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(C.REPORTS_DIR / "history.json", "w") as f:
                json.dump(h, f)
            plot_history(h, C.REPORTS_DIR / "training_curves.png")
    print(f"\nDone! The best network ({best * 100:.0f}% of the goals against the bot) was saved to {C.MODEL_PATH}")
    print("Next steps:  python -m src.evaluate   then   python -m src.export_web")


if __name__ == "__main__":
    main()
