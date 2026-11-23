"""
Export the trained networks for the static web page (docs/), so it can be published on GitHub Pages.

    python -m src.export_web

GitHub Pages only serves files - it can't run Python or PyTorch. In the web version everything runs in the
visitor's browser, in JavaScript: the table's physics (docs/physics.js, a copy of src/physics.py), the bot
(a copy of src/bot.py) and the networks (docs/mlp.js). This script writes what the page needs into docs/model/:

    weights.bin     all the numbers of the networks (float32)
    model.json      the layers of every network, the observation normalization, and the labels

Exported: the Actor of the final network and of every snapshot (the learning stages).
Without docs/model the page still works, with the bot as the only opponent.
"""
import json

import numpy as np

from . import config as C

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


def pack_actor(packer, state, net_id, label, steps):
    layers = [packer.dense(state["actor.0.weight"], state["actor.0.bias"], "tanh"),
              packer.dense(state["actor.2.weight"], state["actor.2.bias"], "tanh"),
              packer.dense(state["actor.4.weight"], state["actor.4.bias"], "none")]
    norm = {"mean": packer.add(state["obs_mean"]), "var": packer.add(state["obs_var"])}
    return {"id": net_id, "label": label, "steps": int(steps), "layers": layers, "norm": norm}


def write(networks, packer):
    weights = packer.weights()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.astype("<f4").tofile(OUT_DIR / "weights.bin")
    doc = {"networks": networks, "n_weights": int(weights.size), "obs_size": C.OBS_SIZE}
    (OUT_DIR / "model.json").write_text(json.dumps(doc, indent=1))
    return weights.size


def load_state(path):
    import torch                                            # PyTorch is only needed here
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.detach().cpu().numpy() for k, v in ck["state_dict"].items()}, ck


def main():
    if not C.MODEL_PATH.exists():
        raise SystemExit(f"No trained network at {C.MODEL_PATH}. Train first:  python -m src.train")
    packer, networks = Packer(), []
    names = ["Beginner", "Rookie", "Club player", "Expert", "Pro"]
    for k, path in enumerate(sorted(C.SNAPSHOT_DIR.glob("upd_*.pt"))):          # the learning stages
        s, ck = load_state(path)
        steps = ck.get("steps", 0)
        name = names[k] if k < len(names) else f"Stage {k + 1}"
        networks.append(pack_actor(packer, s, path.stem, f"{name} (after {steps / 1e6:.1f}M moves)", steps))
    s, ck = load_state(C.MODEL_PATH)
    networks.append(pack_actor(packer, s, "final", "Champion: fully trained", ck.get("steps", 0)))
    n = write(networks, packer)
    print(f"Exported {len(networks)} networks ({n:,} numbers, {n * 4 / 1e3:.0f} KB) to {OUT_DIR}")
    print("\nTry it locally:   python -m http.server 8000 --directory docs   then open http://localhost:8000")


if __name__ == "__main__":
    main()
