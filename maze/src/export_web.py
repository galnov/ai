"""
Export the trained networks for the static web page (docs/), so it can be published on GitHub Pages.

    python -m src.export_web

GitHub Pages only serves files - it can't run Python or PyTorch. So in the web version everything runs in
the visitor's browser, in JavaScript: the maze generator, the ship's physics and sensors, the autopilot,
and the network itself (docs/mlp.js). This script writes what the page needs into docs/model/:

    weights.bin     all the numbers of the networks (float32)
    model.json      the layers of every network, the observation normalization, and the labels

Only the Actor is exported (93 -> 128 -> 128 -> 3): flying uses the mean of its distribution, as in the app.
The snapshots from early in training are exported too, for the "learning stages" in the pilot menu.
Without docs/model the page still works, with the autopilot only.
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


def pack_actor(packer, state, net_id, label):
    """state: a PPO checkpoint's state_dict as NumPy arrays."""
    layers = [packer.dense(state["actor.0.weight"], state["actor.0.bias"], "tanh"),
              packer.dense(state["actor.2.weight"], state["actor.2.bias"], "tanh"),
              packer.dense(state["actor.4.weight"], state["actor.4.bias"], "none")]
    norm = {"mean": packer.add(state["obs_mean"]), "var": packer.add(state["obs_var"])}
    return {"id": net_id, "label": label, "layers": layers, "norm": norm}


def write(networks, packer, meta=None):
    weights = packer.weights()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.astype("<f4").tofile(OUT_DIR / "weights.bin")
    doc = {**(meta or {}), "networks": networks, "n_weights": int(weights.size),
           "obs_size": C.OBS_SIZE, "v_max": C.V_MAX}
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
    state, ck = load_state(C.MODEL_PATH)
    networks.append(pack_actor(packer, state, "ppo", f"PPO network ({ck.get('steps', 0) / 1e6:.1f}M steps)"))
    for path in sorted((C.MODELS_DIR / "snapshots").glob("upd_*.pt")):
        s, sck = load_state(path)
        networks.append(pack_actor(packer, s, f"stage:{path.stem}",
                                   f"Learning stage: after {sck.get('steps', 0) / 1e3:,.0f}K steps"))
    n = write(networks, packer)
    print(f"Exported {len(networks)} networks ({n:,} numbers, {n * 4 / 1e6:.2f} MB) to {OUT_DIR}")
    print("\nTry it locally:   python -m http.server 8000 --directory docs   then open http://localhost:8000")


if __name__ == "__main__":
    main()
