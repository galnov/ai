"""
PPO with a convolutional network: it plays from the screen's pixels.

The network (shared "body", two "heads"):

    2 screens 48x48 ──► conv 6x6, stride 3, 16 filters ──► 15x15x16
                    ──► conv 3x3, stride 2, 32 filters ──► 7x7x32
                    ──► dense 256 ──┬──► policy head: 3 numbers (stay / left / right) -> probabilities (softmax)
                                    └──► value head: 1 number - how many points to expect from here (the Critic)

Why convolutions? A ball in the top-left corner and a ball in the bottom-right corner are the same pattern,
just in a different place. A filter slides over the whole screen, so what it learns in one place works
everywhere - with far fewer weights than a dense layer on 4,608 pixels.

A training round:
  1. 64 boards play 128 decisions each (8,192 samples) - sampling actions from the probabilities (exploration)
  2. GAE: how much better each action was than the Critic expected. A lost ball ends the "episode" here
     (it is the moment the agent should learn to avoid)
  3. 4 passes over the data in minibatches of 1,024 with the clipped PPO objective (10%), plus a small bonus
     for entropy (keeps it exploring)
"""
import numpy as np
import torch
import torch.nn as nn

from . import config as C


def init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class PixelNet(nn.Module):
    def __init__(self):
        super().__init__()
        g = np.sqrt(2)
        self.body = nn.Sequential(
            init(nn.Conv2d(C.FRAMES, 16, 6, stride=3), g), nn.ReLU(),        # 48 -> 15
            init(nn.Conv2d(16, 32, 3, stride=2), g), nn.ReLU(),              # 15 -> 7
            nn.Flatten(),
            init(nn.Linear(32 * 7 * 7, 256), g), nn.ReLU(),
        )
        self.pi = init(nn.Linear(256, len(C.ACTIONS)), 0.01)
        self.v = init(nn.Linear(256, 1), 1.0)

    def forward(self, obs_u8):
        """obs (B, FRAMES, 48, 48) uint8 -> logits (B, 3), value (B,)"""
        h = self.body(obs_u8.float() / 255.0)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def play(self, obs_u8, device="cpu"):
        """The most likely action (no randomness), for NumPy observations."""
        logits, _ = self(torch.as_tensor(np.asarray(obs_u8), device=device))
        return logits.argmax(-1).cpu().numpy()


def ppo_update(model, opt, batch, P):
    obs, act, old_logp, adv, ret = batch
    B = len(ret)
    stats = []
    for _ in range(P["epochs"]):
        perm = torch.randperm(B, device=obs.device)
        for i in range(0, B, P["minibatch"]):
            j = perm[i:i + P["minibatch"]]
            a_adv = (adv[j] - adv[j].mean()) / (adv[j].std() + 1e-8)
            logits, v = model(obs[j])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act[j])
            ratio = torch.exp(logp - old_logp[j])
            pg = -torch.min(ratio * a_adv, torch.clamp(ratio, 1 - P["clip"], 1 + P["clip"]) * a_adv).mean()
            ent = dist.entropy().mean()
            v_loss = ((v - ret[j]) ** 2).mean()
            loss = pg - P["ent_coef"] * ent + P["vf_coef"] * v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), P["max_grad_norm"])
            opt.step()
            stats.append((pg.item(), v_loss.item(), ent.item()))
    return np.mean(stats, axis=0)


def save(model, path, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, **extra}, path)


def load(path=C.MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"No trained network at {path}. Run:  python -m src.train")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = PixelNet()
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
