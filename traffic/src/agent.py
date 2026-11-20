"""
Network 2: the agent that manages the traffic light - Deep Q-Network.

Input (6 numbers): how many cars in the two lanes that have green, how many in the two lanes that have red,
                 whether the light is mid-transition, and how long it has been green.
Output (2): a Q value for each action - "keep" or "switch".

Q(s,a) = an estimate of the future reward (minus the accumulated waiting) if a is chosen in state s.
Learned with the Bellman equation:  Q(s,a) <- r + γ·Q_target(s', argmax Q(s',·))
with a replay buffer, a target network with soft updates, and Double DQN (as in the car project).
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class QNetwork(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C.OBS_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, C.N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, rng):
        self.capacity, self.rng = capacity, rng
        self.obs = np.zeros((capacity, C.OBS_SIZE), np.float32)
        self.next_obs = np.zeros((capacity, C.OBS_SIZE), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.ptr = self.size = 0

    def add(self, obs, actions, rewards, next_obs):
        n = len(obs)
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx] = obs, actions, rewards, next_obs
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch):
        j = self.rng.integers(0, self.size, batch)
        return (torch.from_numpy(self.obs[j]), torch.from_numpy(self.actions[j]),
                torch.from_numpy(self.rewards[j]), torch.from_numpy(self.next_obs[j]))


class DQNAgent:
    def __init__(self, hidden=64, lr=5e-4, gamma=0.95, tau=0.01, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.q = QNetwork(hidden)
        self.q_target = copy.deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.tau, self.hidden = gamma, tau, hidden

    @torch.no_grad()
    def act(self, obs, epsilon=0.0):
        a = self.q(torch.as_tensor(obs)).argmax(dim=1).numpy()
        if epsilon > 0:
            r = self.rng.random(len(a)) < epsilon
            a[r] = self.rng.integers(0, C.N_ACTIONS, r.sum())
        return a

    def learn(self, batch):
        obs, actions, rewards, next_obs = batch
        with torch.no_grad():
            next_a = self.q(next_obs).argmax(dim=1, keepdim=True)
            target = rewards + self.gamma * self.q_target(next_obs).gather(1, next_a).squeeze(1)
            # (a continuing task: there is no "end" - the intersection keeps working)
        pred = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(pred, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.mul_(1 - self.tau).add_(p, alpha=self.tau)
        return loss.item()

    def checkpoint(self, **extra):
        return {"state_dict": copy.deepcopy(self.q.state_dict()), "hidden": self.hidden, **extra}


class Controller:
    """The trained agent - for evaluation and the app."""

    def __init__(self, path=C.AGENT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"No trained agent found at {path}. Run:  python -m src.train_agent")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.net = QNetwork(ck["hidden"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.ckpt = ck

    @torch.no_grad()
    def act(self, obs):
        q = self.net(torch.as_tensor(obs, dtype=torch.float32))
        return q.argmax(dim=1).numpy(), q.numpy()
