"""
The agent: Deep Q-Network (DQN).

The network receives what the car "sees" (10 numbers) and returns a Q value for each of the 6 actions:
an estimate of "how much reward will I get from now on if I choose this action". The agent picks the action with the highest Q.

How do we learn Q? The Bellman equation:
    Q(s, a)  ≈  r + γ · max Q(s', a')
"The value of what I did now = the immediate reward + the (discounted) value of the state I reached".
We train the network to reduce the difference between the two sides of the equation.

Additions that make learning stable:
  * Replay Buffer  - learn from a random sample of past experiences, not just the latest ones
  * Target Network - a copy of the network that updates slowly, used to compute the right side of the equation
  * Double DQN     - the main network chooses a', the target network evaluates it (prevents over-estimation)
  * Huber Loss     - less sensitive to large errors than MSE
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class QNetwork(nn.Module):
    def __init__(self, obs_size=C.OBS_SIZE, n_actions=C.N_ACTIONS, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    @torch.no_grad()
    def forward_with_activations(self, x):
        """For the app: also returns the hidden-layer values (to draw the "brain")."""
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.out(h2), h1, h2


class ReplayBuffer:
    """A circular memory of transitions (s, a, r, s', done)."""

    def __init__(self, capacity, obs_size, rng):
        self.capacity = capacity
        self.rng = rng
        self.obs = np.zeros((capacity, obs_size), np.float32)
        self.next_obs = np.zeros((capacity, obs_size), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, actions, rewards, next_obs, dones):
        n = len(obs)
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.dones[idx] = dones
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        j = self.rng.integers(0, self.size, batch_size)
        return (torch.from_numpy(self.obs[j]), torch.from_numpy(self.actions[j]),
                torch.from_numpy(self.rewards[j]), torch.from_numpy(self.next_obs[j]),
                torch.from_numpy(self.dones[j]))

    def __len__(self):
        return self.size


class DQNAgent:
    def __init__(self, hidden=128, lr=3e-4, gamma=0.99, tau=0.005, grad_clip=10.0, seed=0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.q = QNetwork(hidden=hidden)
        self.q_target = copy.deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.tau, self.grad_clip, self.hidden = gamma, tau, grad_clip, hidden

    @torch.no_grad()
    def act(self, obs, epsilon=0.0):
        """epsilon-greedy: with probability epsilon a random action (exploration), otherwise the best action."""
        q = self.q(torch.as_tensor(obs, dtype=torch.float32))
        actions = q.argmax(dim=1).numpy()
        if epsilon > 0:
            explore = self.rng.random(len(actions)) < epsilon
            actions[explore] = self.rng.integers(0, C.N_ACTIONS, explore.sum())
        return actions

    def learn(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        with torch.no_grad():
            next_a = self.q(next_obs).argmax(dim=1, keepdim=True)                  # Double DQN: selection
            next_q = self.q_target(next_obs).gather(1, next_a).squeeze(1)          # ...and evaluation
            target = rewards + self.gamma * (1 - dones) * next_q                   # Bellman equation
        q_pred = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_pred, target)                                   # Huber

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optimizer.step()

        # soft update: the target network slowly moves toward the main network
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.mul_(1 - self.tau).add_(p, alpha=self.tau)
        return loss.item(), q_pred.mean().item()

    # ------------------------------------------------------------ saving and loading
    def checkpoint(self, **extra):
        return {"state_dict": copy.deepcopy(self.q.state_dict()), "hidden": self.hidden,
                "obs_size": C.OBS_SIZE, "n_actions": C.N_ACTIONS, **extra}


def save_checkpoint(ckpt, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_network(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = QNetwork(ckpt["obs_size"], ckpt["n_actions"], ckpt["hidden"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


class Driver:
    """A trained driver - used by the app and the evaluation."""

    def __init__(self, path):
        self.net, self.ckpt = load_network(path)

    def act(self, obs):
        q, h1, h2 = self.net.forward_with_activations(torch.as_tensor(obs, dtype=torch.float32))
        return q.argmax(dim=1).numpy(), q.numpy(), h1.numpy(), h2.numpy()

    def weights(self):
        """The weights, for drawing the connections in the network."""
        return [self.net.fc1.weight.detach().numpy(), self.net.fc2.weight.detach().numpy(),
                self.net.out.weight.detach().numpy()]
