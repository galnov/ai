"""
Export the trained network for the static web page (docs/), so it can be published on GitHub Pages.

    python -m src.export_web

GitHub Pages only serves files - it can't run Python, PyTorch or OpenCV. In the web version everything runs
in the visitor's browser, in JavaScript: finding the faces (docs/faces.js), cropping them, the CNN and
Grad-CAM (docs/emotion-cnn.js). Nothing is sent anywhere: the camera images stay on the visitor's computer.
This script writes what the page needs into docs/model/:

    weights.bin     all the network's numbers (float32), about 4.8 MB (width 64) or 1.2 MB (width 32)
    model.json      the layers, the emotions, and the accuracy from training

BatchNorm is "folded" into the layer before it. In evaluation mode BatchNorm is just a fixed scale and
shift per channel:   y = gamma * (x - mean) / sqrt(var + eps) + beta
so it can be merged into the convolution's (or dense layer's) weights and bias:   W' = W * s,
b' = beta - mean * s (+ b * s), where s = gamma / sqrt(var + eps). The result is exactly the same network,
with fewer steps to compute.
"""
import json

import numpy as np

from .config import EMOTION_COLOR, EMOTION_EMOJI, EMOTION_NAMES, MODEL_PATH, ROOT

DOCS_DIR = ROOT / "docs"
OUT_DIR = DOCS_DIR / "model"
BN_EPS = 1e-5


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


def fold_bn(state, prefix):
    g, b = state[f"{prefix}.weight"], state[f"{prefix}.bias"]
    mean, var = state[f"{prefix}.running_mean"], state[f"{prefix}.running_var"]
    s = g / np.sqrt(var + BN_EPS)
    return s, b - mean * s


def pack(state, packer):
    """state: the EmotionCNN's state_dict as NumPy arrays. Returns the layer list for model.json."""
    layers = []
    for blk in range(1, 5):
        for k in (1, 2):
            w = state[f"block{blk}.conv{k}.weight"]                          # (out, in, 3, 3), no bias
            s, bias = fold_bn(state, f"block{blk}.bn{k}")
            layers.append({"type": "conv", "in": int(w.shape[1]), "out": int(w.shape[0]), "k": 3, "dil": 1,
                           "act": "relu", "w": packer.add(w * s[:, None, None, None]), "b": packer.add(bias)})
        layers.append({"type": "maxpool", **({"cam": True} if blk == 4 else {})})   # Grad-CAM looks before the last pooling
    w1, b1 = state["classifier.1.weight"], state["classifier.1.bias"]       # (256, 8w)
    s, shift = fold_bn(state, "classifier.2")
    dense1 = {"in": int(w1.shape[1]), "out": int(w1.shape[0]), "w": packer.add(w1 * s[:, None]), "b": packer.add(b1 * s + shift)}
    w2, b2 = state["classifier.5.weight"], state["classifier.5.bias"]
    dense2 = {"in": int(w2.shape[1]), "out": int(w2.shape[0]), "w": packer.add(w2), "b": packer.add(b2)}
    return layers, dense1, dense2


def write(layers, dense1, dense2, packer, meta, mean=0.5, std=0.5):
    """mean, std: the input normalization (MEAN, STD in src/dataset.py)."""
    weights = packer.weights()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.astype("<f4").tofile(OUT_DIR / "weights.bin")
    doc = {**meta, "mean": float(mean), "std": float(std), "layers": layers, "dense1": dense1, "dense2": dense2,
           "n_weights": int(weights.size)}
    (OUT_DIR / "model.json").write_text(json.dumps(doc, indent=1))
    return weights.size


def main():
    from .dataset import MEAN, STD                         # PyTorch is only needed here
    from .utils import load_model
    model, ckpt = load_model(MODEL_PATH, "cpu")
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    emotions = list(ckpt["emotions"])
    packer = Packer()
    layers, d1, d2 = pack(state, packer)
    meta = {
        "emotions": emotions,
        "categories": [{"key": e, "name": EMOTION_NAMES[e], "emoji": EMOTION_EMOJI[e], "color": EMOTION_COLOR[e]} for e in emotions],
        "val_acc": ckpt.get("val_acc"), "train_samples": ckpt.get("train_samples"), "epoch": ckpt.get("epoch"),
        "width": ckpt["width"],
    }
    n = write(layers, d1, d2, packer, meta, MEAN, STD)
    print(f"Exported {n:,} weights ({n * 4 / 1e6:.1f} MB) to {OUT_DIR}")
    print("\nTry it locally:   python -m http.server 8000 --directory docs   then open http://localhost:8000")
    print("(the camera works on http://localhost and on https:// addresses such as GitHub Pages)")


if __name__ == "__main__":
    main()
