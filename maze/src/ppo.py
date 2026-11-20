"""
PPO (Proximal Policy Optimization) with continuous actions - the same algorithm as in the drone project.

Two networks:
  * Actor (policy): takes the observation (93 numbers) and returns a *normal distribution* for each of the
    3 controls (pitch, yaw, throttle): a mean from the network, and a standard deviation that is a learned
    parameter. In training we sample from the distribution (exploration). When flying for real we use the mean.
  * Critic (value): estimates V(s) - how much reward to expect from this state onward.

    Actor:   93 -> 128 (tanh) -> 128 (tanh) -> 3 means
    Critic:  93 -> 128 (tanh) -> 128 (tanh) -> 1 value

A training round:
  1. Collect 128 steps from each of the 64 ships (8,192 samples) with the current policy
  2. Compute the "advantage" with GAE: how much better each action turned out than the Critic expected
  3. Update 10 times on this data, but "clip": the new policy can't move more than 20% away from the old
     one in the probability of any action. This is what makes PPO stable.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from . import config as C

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def layer(n_in, n_out, gain):
    lin = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


def gaussian_log_prob(a, mean, log_std):
    """log of the normal distribution's density, summed over the 3 controls."""
    return (-0.5 * ((a - mean) / log_std.exp()) ** 2 - log_std - LOG_SQRT_2PI).sum(-1)


class ActorCritic(nn.Module):
    def __init__(self, hidden=128, log_std_init=-0.5):
        super().__init__()
        h, g = hidden, math.sqrt(2)
        self.actor = nn.Sequential(layer(C.OBS_SIZE, h, g), nn.Tanh(), layer(h, h, g), nn.Tanh(),
                                   layer(h, C.ACT_SIZE, 0.01))
        self.critic = nn.Sequential(layer(C.OBS_SIZE, h, g), nn.Tanh(), layer(h, h, g), nn.Tanh(),
                                    layer(h, 1, 1.0))
        self.log_std = nn.Parameter(torch.full((C.ACT_SIZE,), float(log_std_init)))
        # observation normalization (running mean and variance) - saved together with the network
        self.register_buffer("obs_mean", torch.zeros(C.OBS_SIZE))
        self.register_buffer("obs_var", torch.ones(C.OBS_SIZE))

    def norm(self, obs):
        return torch.clamp((obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -10, 10)

    def value(self, obs_n):
        return self.critic(obs_n).squeeze(-1)


class RunningStat:
    """Running mean and variance (Welford/Chan algorithm)."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        bm, bv, n = x.mean(0), x.var(0), len(x)
        d = bm - self.mean
        tot = self.count + n
        self.mean = self.mean + d * n / tot
        self.var = (self.var * self.count + bv * n + d ** 2 * self.count * n / tot) / tot
        self.count = tot


def ppo_update(model, opt, batch, P):
    """PPO update: 10 passes over the data, in minibatches."""
    obs, act, old_logp, adv, ret = batch
    B = len(ret)
    stats = []
    for _ in range(P["epochs"]):
        perm = torch.randperm(B)
        for i in range(0, B, P["minibatch"]):
            j = perm[i:i + P["minibatch"]]
            a_adv = adv[j]
            a_adv = (a_adv - a_adv.mean()) / (a_adv.std() + 1e-8)
            logp = gaussian_log_prob(act[j], model.actor(obs[j]), model.log_std)
            ratio = torch.exp(logp - old_logp[j])
            # the "clipped" PPO objective
            pg_loss = -torch.min(ratio * a_adv, torch.clamp(ratio, 1 - P["clip"], 1 + P["clip"]) * a_adv).mean()
            v_loss = ((model.value(obs[j]) - ret[j]) ** 2).mean()
            loss = pg_loss + P["vf_coef"] * v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), P["max_grad_norm"])
            opt.step()
            stats.append((pg_loss.item(), v_loss.item()))
    return np.mean(stats, axis=0)


def save(model, path, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "hidden": C.PPO["hidden"], **extra}, path)


class Pilot:
    """A trained network - for evaluation and the app. Action = the mean of the distribution (no randomness)."""

    def __init__(self, path=C.MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"No trained network found at {path}. Run:  python -m src.train")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.model = ActorCritic(ck["hidden"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.ckpt = ck

    @torch.no_grad()
    def __call__(self, obs):
        o = self.model.norm(torch.as_tensor(np.asarray(obs), dtype=torch.float32))
        return self.model.actor(o).numpy()
