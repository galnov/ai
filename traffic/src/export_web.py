"""
Export the trained networks for the static web page (docs/), so it can be published on GitHub Pages.

    python -m src.export_web

GitHub Pages only serves files - it can't run Python or PyTorch. In the web version everything runs in the
visitor's browser, in JavaScript: the traffic simulation (docs/sim.js, a copy of src/traffic.py and
app/world.py), the cameras (docs/camera.js, a copy of src/camera.py), and both networks (docs/mlp.js for the
traffic-light agent, docs/counter.js for the camera network). This script writes what the page needs:

    docs/model/weights.bin   all the numbers of both networks (float32)
    docs/model/model.json    their layers, and the validation results from training
    docs/scene.png           the background of the intersection the cameras see (Scene(104), as in the app)
    docs/scene.json          the sun's direction (for the cars' shadows) and the street lamps

BatchNorm in the camera network is "folded" into the convolution before it. In evaluation mode BatchNorm
is just a fixed scale and shift per channel:   y = gamma * (x - mean) / sqrt(var + eps) + beta
so it can be merged into the convolution's weights and bias:   W' = W * s,  b' = beta - mean * s,
where s = gamma / sqrt(var + eps). The result is exactly the same network, with fewer steps to compute.

Without docs/model the page still works: the AI light uses the simple rule and a perfect sensor.
"""
import json

import numpy as np

from . import config as C

DOCS_DIR = C.ROOT / "docs"
OUT_DIR = DOCS_DIR / "model"
BN_EPS = 1e-5
SCENE_SEED = 104                 # the same intersection as in the app (app/world.py)
# the camera network's layers (src/counter.py): index in model.features -> dilation; 2 and 5 are max-pooling
CONV_BLOCKS = {0: 1, 1: 1, 3: 1, 4: 1, 6: 1, 7: 1, 8: 2, 9: 4}


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


def pack_agent(packer, state, ckpt):
    """The DQN agent (src/agent.py QNetwork): 6 -> 64 -> 64 -> 2."""
    layers = [packer.dense(state["net.0.weight"], state["net.0.bias"], "relu"),
              packer.dense(state["net.2.weight"], state["net.2.bias"], "relu"),
              packer.dense(state["net.4.weight"], state["net.4.bias"], "none")]
    meta = {"val_wait": ckpt.get("val_wait"), "baselines": ckpt.get("baselines")}
    return {"id": "agent", "label": "DQN agent", "layers": layers}, meta


def pack_counter(packer, state):
    """The camera network (src/counter.py CounterCNN), with BatchNorm folded into the convolutions."""
    layers = []
    for i in range(10):
        if i in (2, 5):
            layers.append({"type": "maxpool"})
            continue
        w = state[f"features.{i}.0.weight"]                       # (out, in, 3, 3), no bias
        g, beta = state[f"features.{i}.1.weight"], state[f"features.{i}.1.bias"]
        mean, var = state[f"features.{i}.1.running_mean"], state[f"features.{i}.1.running_var"]
        s = g / np.sqrt(var + BN_EPS)
        layers.append({"type": "conv", "in": int(w.shape[1]), "out": int(w.shape[0]), "k": 3, "dil": CONV_BLOCKS[i],
                       "act": "relu", "w": packer.add(w * s[:, None, None, None]), "b": packer.add(beta - mean * s)})
    hw, hb = state["head.weight"], state["head.bias"]              # (1, C, 1, 1)
    layers.append({"type": "conv", "in": int(hw.shape[1]), "out": 1, "k": 1, "dil": 1, "act": "softplus",
                   "w": packer.add(hw), "b": packer.add(hb)})
    return layers


def write_scene():
    """The background the cameras see, drawn once in Python (the browser adds the cars)."""
    from .camera import Scene
    scene = Scene(SCENE_SEED)
    DOCS_DIR.mkdir(exist_ok=True)
    scene.base.save(DOCS_DIR / "scene.png")
    (DOCS_DIR / "scene.json").write_text(json.dumps({"seed": SCENE_SEED, "sun": float(scene.sun),
                                                     "lamps": [[float(x), float(y)] for x, y in scene.lamps]}))


def write(networks, packer, extra):
    weights = packer.weights()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.astype("<f4").tofile(OUT_DIR / "weights.bin")
    doc = {**extra, "networks": networks, "n_weights": int(weights.size)}
    (OUT_DIR / "model.json").write_text(json.dumps(doc, indent=1))
    return weights.size


def main():
    import torch                                            # PyTorch is only needed here
    write_scene()
    print(f"Wrote the camera background to {DOCS_DIR / 'scene.png'}")
    packer, networks, extra = Packer(), [], {}
    if C.AGENT_PATH.exists():
        ck = torch.load(C.AGENT_PATH, map_location="cpu", weights_only=True)
        net, extra["agent"] = pack_agent(packer, {k: v.numpy() for k, v in ck["state_dict"].items()}, ck)
        networks.append(net)
        print(f"  the traffic-light agent: {C.AGENT_PATH.name}")
    if C.COUNTER_PATH.exists():
        ck = torch.load(C.COUNTER_PATH, map_location="cpu", weights_only=True)
        extra["counter"] = {"layers": pack_counter(packer, {k: v.numpy() for k, v in ck["model_state"].items()}),
                            "val_mae": ck.get("val_mae")}
        print(f"  the camera network: {C.COUNTER_PATH.name}")
    if not networks and "counter" not in extra:
        raise SystemExit("No trained networks in models/. Train first (python -m src.train_agent, src.train_counter).")
    n = write(networks, packer, extra)
    print(f"Exported {n:,} numbers ({n * 4 / 1e6:.1f} MB) to {OUT_DIR}")
    print("\nTry it locally:   python -m http.server 8000 --directory docs   then open http://localhost:8000")


if __name__ == "__main__":
    main()
