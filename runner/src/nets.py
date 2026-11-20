"""
The neural network - the same architecture for all three AI players:
    imitation (behavioral cloning)  - the 5 outputs are "how likely a human would press each key" (logits)
    DQN and combo                   - the 5 outputs are Q values: "how much future reward if I press this key"

Input: the observation from game.observe() - 297 numbers:
    the track ahead: 3 obstacle types x 32 distance bins (1.5 m each) x 3 lanes = 288 numbers (0 or 1)
    the runner: 9 numbers (lane, lane change, jump, slide, speed)

    297 -> 256 (ReLU) -> 256 (ReLU) -> 5

A fully connected network: every neuron in the first layer looks at the whole track at once, so it can
directly learn combinations such as "a train in my lane in 20 m, and the left lane is free".

Why not convolutions? We tried a 1D convolutional network along the track (the same filter at every
distance). In tests during development it learned imitation well, but with DQN it learned much more slowly
than this network (about 150 m after 32,000 training steps, against about 800 m). With only 32 x 3 cells
the fully connected network is small and fast enough, so we kept the simpler choice.
It has about 143,000 parameters and trains well on a laptop CPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class RunnerNet(nn.Module):
    def __init__(self, n_actions=C.N_ACTIONS, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(C.OBS_SIZE, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)
        with torch.no_grad():                       # small outputs at the start
            self.out.weight.mul_(0.1)
            self.out.bias.zero_()

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return self.out(h)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters())


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")      # Apple's "mps" is slower than the CPU for a network this small


def save_model(net, path, kind, **extra):
    """kind: 'imitation' (outputs are logits) or 'q' (outputs are Q values)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"state_dict": state, "kind": kind, **extra}, path)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net = RunnerNet()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
