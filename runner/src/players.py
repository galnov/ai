"""
The players: everything that can play the game, behind one interface.

    player.act(game) -> (actions (n,), scores (n, 5) or None)

scores are shown in the app as "what the AI is thinking":
    imitation  - probabilities: how likely a human would press each key here
    DQN, combo - Q values: how much future reward each key is worth
"""
import numpy as np

from . import config as C
from .bot import RuleBot
from .game import RunnerGame

MODEL_FILES = {"combo": C.COMBO_PATH, "dqn": C.DQN_PATH, "imitation": C.BC_PATH}
LABELS = {"combo": "Combo (imitation + RL)", "dqn": "DQN (RL from scratch)",
          "imitation": "Imitation (learned from humans)", "bot": "Rule-based bot"}


class ModelPlayer:
    def __init__(self, path=None, net=None, kind="q", device=None):
        """Either a saved model (path), or a network that is still training (net + kind)."""
        import torch
        from .nets import load_model
        self.torch = torch
        if net is None:
            net, self.ckpt = load_model(path)
        else:
            self.ckpt = {"kind": kind}
        self.net, self.kind = net, self.ckpt["kind"]
        self.device = device or next(net.parameters()).device

    def act(self, game):
        torch = self.torch
        with torch.no_grad():
            out = self.net(torch.as_tensor(game.observe(), device=self.device))
            scores = torch.softmax(out, dim=1) if self.kind == "imitation" else out
        scores = scores.cpu().numpy()
        return scores.argmax(axis=1), scores


class BotPlayer:
    kind = "bot"

    def __init__(self, n=1):
        self.bot = RuleBot(n, skill=1.0)

    def act(self, game):
        if self.bot.n != game.n:
            self.bot = RuleBot(game.n, skill=1.0)
        return self.bot.act(game), None


def available():
    """The players that exist right now (the networks only after training)."""
    names = [k for k, p in MODEL_FILES.items() if p.exists()]
    return names + ["bot"]


def make_player(name):
    if name == "bot":
        return BotPlayer()
    return ModelPlayer(MODEL_FILES[name])


def play(player, seeds, max_ticks=C.EVAL_TICKS, t_offs=None):
    """Plays one game per seed, all in parallel. Returns distances (m), crash types and ticks survived."""
    g = RunnerGame(len(seeds))
    g.reset(seeds, t_offs)
    ticks = np.full(len(seeds), max_ticks)
    for t in range(max_ticks):
        if not g.alive.any():
            break
        was = g.alive.copy()
        g.step(player.act(g)[0])
        ticks[was & ~g.alive] = t + 1
    return g.distance_m(), g.crash_type.copy(), ticks
