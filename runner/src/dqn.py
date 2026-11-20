"""
Reinforcement learning: Deep Q-Network (DQN), and its version that also learns from human runs (DQfD).

Q(s, a) = "how much reward will I collect from now on if I press key a in situation s".
The runner presses the key with the highest Q. Q is learned from the Bellman equation:

    Q(s, a)  ≈  r + γ · Q_target(s', argmax_a' Q(s', a'))

    r = +0.1 for every tick alive, -2 for a crash, -0.01 for a key press.

Additions that make learning stable (as in the car project):
  * Replay buffer   - learn from random samples of past experience, not only the latest
  * Target network  - a slowly updated copy, used on the right side of the equation
  * Double DQN      - the main network picks a', the target network rates it
  * Huber loss      - less sensitive to large errors than squared error

DQfD (Deep Q-learning from Demonstrations, Hester et al. 2018) adds the human runs:
  * Before playing at all, the network learns from the recordings only (pre-training).
  * After that, a quarter of every batch comes from the recordings.
  * On recorded ticks there is an extra "margin" loss: the action the human chose must be worth at least
    `margin` more than any other action. That pushes the network to start out playing like the humans,
    and the Bellman equation can later correct the humans' mistakes.
  * As in imitation learning, rare keys get a larger weight in the margin loss (most ticks are "run").
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .nets import RunnerNet


class ReplayBuffer:
    """A circular memory of transitions (s, a, r, s', done)."""

    def __init__(self, capacity, rng):
        self.capacity, self.rng = capacity, rng
        self.obs = np.zeros((capacity, C.OBS_SIZE), np.float32)
        self.next_obs = np.zeros((capacity, C.OBS_SIZE), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.ptr = self.size = 0

    def add(self, obs, actions, rewards, next_obs, dones):
        idx = (self.ptr + np.arange(len(obs))) % self.capacity
        self.obs[idx], self.actions[idx], self.rewards[idx] = obs, actions, rewards
        self.next_obs[idx], self.dones[idx] = next_obs, dones
        self.ptr = (self.ptr + len(obs)) % self.capacity
        self.size = min(self.size + len(obs), self.capacity)

    def sample(self, n):
        j = self.rng.integers(0, self.size, n)
        return self.obs[j], self.actions[j], self.rewards[j], self.next_obs[j], self.dones[j]

    def __len__(self):
        return self.size


class DemoBuffer:
    """The human runs as fixed transitions (they never change)."""

    def __init__(self, data, rng):
        self.d, self.rng = data, rng
        # most recorded ticks are "run" (no key): weight the rare keys up, exactly as in imitation learning,
        # otherwise the margin loss mostly teaches "never press anything"
        freq = np.bincount(data["actions"], minlength=C.N_ACTIONS) / len(data["actions"])
        w = 1.0 / np.sqrt(np.maximum(freq, 1e-4))
        self.class_weights = (w / (w * freq).sum()).astype(np.float32)

    def sample(self, n):
        j = self.rng.integers(0, len(self.d["actions"]), n)
        return (self.d["obs"][j], self.d["actions"][j], self.d["rewards"][j],
                self.d["next_obs"][j], self.d["dones"][j])

    def __len__(self):
        return len(self.d["actions"])


class DQNAgent:
    def __init__(self, device, lr, gamma, tau, seed=0, grad_clip=10.0, class_weights=None):
        torch.manual_seed(seed)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.q = RunnerNet().to(device)
        self.q_target = copy.deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.tau, self.grad_clip = gamma, tau, grad_clip
        self.class_weights = None if class_weights is None else torch.as_tensor(class_weights, device=device)
        # random actions during exploration: mostly "run", like real play (pressing keys all the time
        # mostly gets the runner killed, which teaches little)
        self.explore_p = np.array([0.6, 0.1, 0.1, 0.1, 0.1])

    @torch.no_grad()
    def act(self, obs, epsilon):
        q = self.q(torch.as_tensor(obs, device=self.device))
        actions = q.argmax(1).cpu().numpy()
        explore = self.rng.random(len(actions)) < epsilon
        actions[explore] = self.rng.choice(C.N_ACTIONS, explore.sum(), p=self.explore_p)
        return actions

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def learn(self, batch, demo_batch=None, margin=0.8, margin_weight=1.0):
        """One update. batch / demo_batch = (obs, actions, rewards, next_obs, dones) as NumPy arrays."""
        parts = [b for b in (batch, demo_batch) if b is not None]
        obs, actions, rewards, next_obs, dones = (
            torch.as_tensor(np.concatenate([p[i] for p in parts]), device=self.device) for i in range(5))
        with torch.no_grad():
            next_a = self.q(next_obs).argmax(1, keepdim=True)                    # Double DQN: choose...
            next_q = self.q_target(next_obs).gather(1, next_a).squeeze(1)        # ...and rate
            target = rewards + self.gamma * (1 - dones) * next_q                 # Bellman equation
        q_all = self.q(obs)
        q_pred = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_loss = F.smooth_l1_loss(q_pred, target)
        loss, m_loss = td_loss, torch.zeros((), device=self.device)

        if demo_batch is not None:
            # margin loss on the human ticks: max_a [Q(s,a) + margin·(a != human's a)] - Q(s, human's a)
            k = len(demo_batch[1])
            qd, ad = q_all[-k:], actions[-k:]
            bonus = torch.full_like(qd, margin)
            bonus.scatter_(1, ad.unsqueeze(1), 0.0)
            per_tick = (qd + bonus).max(1).values - qd.gather(1, ad.unsqueeze(1)).squeeze(1)
            w = self.class_weights[ad] if self.class_weights is not None else torch.ones_like(per_tick)
            m_loss = (w * per_tick).mean()
            loss = loss + margin_weight * m_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        with torch.no_grad():                                                     # soft target update
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.mul_(1 - self.tau).add_(p, alpha=self.tau)
        return td_loss.item(), m_loss.item(), q_pred.mean().item()
