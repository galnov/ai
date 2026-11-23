"""
Export the trained networks for the static web page (docs/), so it can be published on GitHub Pages.

    python -m src.export_web

In the web version everything runs in the visitor's browser: the game (docs/game.js, a copy of src/game.py)
and the network (docs/cnn.js). This script writes docs/model/:

    weights.bin     all the numbers of the networks, as float16 (half the download; plenty of precision to play)
    model.json      the layers of every network and where its numbers are in weights.bin

Exported: the final network and every snapshot (the learning stages), the policy part only (the value head
isn't needed to play). Conv weights are in PyTorch's order: [out][in][row][column].
"""
import json

import numpy as np

from . import config as C

OUT_DIR = C.ROOT / "docs" / "model"


class Packer:
    """Collects arrays into one float32 buffer and remembers where each one starts."""

    def __init__(self):
        self.chunks, self.offset = [], 0

    def add(self, arr):
        a = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        self.chunks.append(a)
        start, self.offset = self.offset, self.offset + a.size
        return start

    def weights(self):
        return np.concatenate(self.chunks)


def pack(packer, s, net_id, label, steps):
    w1, w2 = np.asarray(s["body.0.weight"]), np.asarray(s["body.2.weight"])
    layers = [
        {"type": "conv", "in": int(w1.shape[1]), "out": int(w1.shape[0]), "k": int(w1.shape[2]), "stride": 3,
         "w": packer.add(w1), "b": packer.add(s["body.0.bias"])},
        {"type": "conv", "in": int(w2.shape[1]), "out": int(w2.shape[0]), "k": int(w2.shape[2]), "stride": 2,
         "w": packer.add(w2), "b": packer.add(s["body.2.bias"])},
        {"type": "dense", "in": int(np.asarray(s["body.5.weight"]).shape[1]), "out": int(np.asarray(s["body.5.weight"]).shape[0]),
         "act": "relu", "w": packer.add(s["body.5.weight"]), "b": packer.add(s["body.5.bias"])},
        {"type": "dense", "in": int(np.asarray(s["pi.weight"]).shape[1]), "out": int(np.asarray(s["pi.weight"]).shape[0]),
         "act": "none", "w": packer.add(s["pi.weight"]), "b": packer.add(s["pi.bias"])},
    ]
    return {"id": net_id, "label": label, "steps": int(steps), "layers": layers}


def write(networks, packer):
    w = packer.weights()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w.astype("<f2").tofile(OUT_DIR / "weights.bin")
    meta = {"networks": networks, "n_weights": int(w.size), "dtype": "f16", "screen": C.SCREEN, "frames": C.FRAMES,
            "actions": C.ACTIONS}
    (OUT_DIR / "model.json").write_text(json.dumps(meta, indent=1))
    return w.size


def load_state(path):
    import torch                                             # PyTorch is only needed here
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v.numpy() for k, v in ck["state_dict"].items()}, ck


STAGE_NAMES = ["Beginner", "Learner", "Improving", "Advanced", "Expert"]


def main():
    if not C.MODEL_PATH.exists():
        raise SystemExit(f"No trained network at {C.MODEL_PATH}. Train first:  python -m src.train")
    packer, nets = Packer(), []
    for k, path in enumerate(sorted(C.SNAPSHOT_DIR.glob("steps_*.pt"))):
        s, ck = load_state(path)
        name = STAGE_NAMES[k] if k < len(STAGE_NAMES) else f"Stage {k + 1}"
        nets.append(pack(packer, s, path.stem, f"{name} (after {ck['steps'] / 1e6:.1f}M moves)", ck["steps"]))
    s, ck = load_state(C.MODEL_PATH)
    nets.append(pack(packer, s, "final", "Fully trained", ck["steps"]))
    n = write(nets, packer)
    print(f"Exported {len(nets)} networks ({n:,} numbers, {n * 2 / 1e6:.1f} MB) to {OUT_DIR}")
    print("\nTry it locally:   python -m http.server 8000 --directory docs   then open http://localhost:8000")


if __name__ == "__main__":
    main()
