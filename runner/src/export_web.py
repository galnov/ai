"""
Export the trained networks for the static web page (docs/), so it can be published on GitHub Pages.

    python -m src.export_web

GitHub Pages only serves files - it can't run Python or PyTorch. In the web version everything runs in the
visitor's browser, in JavaScript: the game (docs/game.js, the twin of src/game.py), what the network sees
(the 297 numbers of game.observe), the rule-based bot, and the networks themselves (docs/mlp.js).
This script writes what the page needs into docs/model/:

    weights.bin     all the numbers of the networks (float32)
    model.json      the layers of every network and its kind ("imitation": the outputs are logits,
                    "q": the outputs are Q values)

All three networks are exported, if they exist: combo, DQN and imitation.
Without docs/model the page still works, with the rule-based bot as the AI.
"""
import json

import numpy as np

from . import config as C
from .players import LABELS, MODEL_FILES

DOCS_DIR = C.ROOT / "docs"
OUT_DIR = DOCS_DIR / "model"


class Packer:
    """Collects arrays into one float32 buffer and remembers where each one starts."""

    def __init__(self):
        self.chunks, self.offset = [], 0

    def add(self, arr):
        a = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        self.chunks.append(a)
        start, self.offset = self.offset, self.offset + a.size
        return start

    def dense(self, w, b, act):
        w = np.asarray(w)
        return {"in": int(w.shape[1]), "out": int(w.shape[0]), "act": act, "w": self.add(w), "b": self.add(b)}

    def weights(self):
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)


def pack_net(packer, state, net_id, kind, extra=None):
    """state: the RunnerNet's state_dict as NumPy arrays (fc1, fc2, out)."""
    layers = [packer.dense(state["fc1.weight"], state["fc1.bias"], "relu"),
              packer.dense(state["fc2.weight"], state["fc2.bias"], "relu"),
              packer.dense(state["out.weight"], state["out.bias"], "none")]
    return {"id": net_id, "label": LABELS[net_id], "kind": kind, "layers": layers, **(extra or {})}


def write(networks, packer):
    weights = packer.weights()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.astype("<f4").tofile(OUT_DIR / "weights.bin")
    doc = {"networks": networks, "n_weights": int(weights.size), "obs_size": C.OBS_SIZE, "actions": C.ACTION_NAMES}
    (OUT_DIR / "model.json").write_text(json.dumps(doc, indent=1))
    return weights.size


def main():
    import torch                                            # PyTorch is only needed here
    packer, networks = Packer(), []
    for net_id, path in MODEL_FILES.items():
        if not path.exists():
            continue
        ck = torch.load(path, map_location="cpu", weights_only=True)
        state = {k: v.detach().cpu().numpy() for k, v in ck["state_dict"].items()}
        networks.append(pack_net(packer, state, net_id, ck["kind"]))
        print(f"  {LABELS[net_id]}: {path.name}")
    if not networks:
        raise SystemExit("No trained networks in models/. Train first (python -m src.train_bc / src.train_dqn).")
    n = write(networks, packer)
    print(f"Exported {len(networks)} network(s) ({n:,} numbers, {n * 4 / 1e6:.1f} MB) to {OUT_DIR}")
    print("\nTry it locally:   python -m http.server 8000 --directory docs   then open http://localhost:8000")


if __name__ == "__main__":
    main()
