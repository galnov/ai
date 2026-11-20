"""
Export the trained network for the static web page (docs/), so it can be published on GitHub Pages.

    python -m src.export_web

GitHub Pages only serves files - it can't run Python or PyTorch. So the network runs in the visitor's
browser, in plain JavaScript (docs/cnn.js). This script writes what the page needs into docs/model/:

    weights.bin     all the network's numbers (float32), about 2.4 MB
    model.json      the layer list, the categories, and the accuracy from training
    samples.json    12 real drawings per category from the dataset (if it was downloaded)

BatchNorm is "folded" into the layer before it. In evaluation mode BatchNorm is just a fixed scale and
shift per channel:   y = gamma * (x - mean) / sqrt(var + eps) + beta
so it can be merged into the convolution's weights and bias:   W' = W * s,  b' = beta - mean * s,
where s = gamma / sqrt(var + eps). The result is exactly the same network, with fewer steps to compute.
"""
import base64
import json

import numpy as np

from .config import CATEGORIES, MODEL_PATH, RAW_DIR, ROOT, file_name

DOCS_DIR = ROOT / "docs"
OUT_DIR = DOCS_DIR / "model"
BN_EPS = 1e-5
SAMPLES_PER_CLASS = 12


def fold_bn(state, prefix_bn):
    g, b = state[f"{prefix_bn}.weight"], state[f"{prefix_bn}.bias"]
    mean, var = state[f"{prefix_bn}.running_mean"], state[f"{prefix_bn}.running_var"]
    s = g / np.sqrt(var + BN_EPS)
    return s, b - mean * s


def pack(state):
    """
    state: the model's state_dict as NumPy arrays.
    Returns (list of layers for model.json, float32 array of all weights).
    """
    layers, chunks, offset = [], [], 0

    def add(arr):
        nonlocal offset
        a = np.ascontiguousarray(arr, dtype=np.float32).ravel()
        chunks.append(a)
        start = offset
        offset += a.size
        return start

    for blk in range(3):                                   # features.0 .. features.2
        for conv_i, bn_i in ((0, 1), (3, 4)):
            w = state[f"features.{blk}.{conv_i}.weight"]   # (out, in, 3, 3), no bias
            s, bias = fold_bn(state, f"features.{blk}.{bn_i}")
            w = w * s[:, None, None, None]
            layers.append({"type": "conv", "in": int(w.shape[1]), "out": int(w.shape[0]),
                           "w": add(w), "b": add(bias)})
        layers.append({"type": "maxpool"})
    w1, b1 = state["classifier.1.weight"], state["classifier.1.bias"]      # (256, 1152)
    s, shift = fold_bn(state, "classifier.2")
    layers.append({"type": "dense", "in": int(w1.shape[1]), "out": int(w1.shape[0]), "relu": True,
                   "w": add(w1 * s[:, None]), "b": add(b1 * s + shift)})
    w2, b2 = state["classifier.5.weight"], state["classifier.5.bias"]
    layers.append({"type": "dense", "in": int(w2.shape[1]), "out": int(w2.shape[0]), "relu": False,
                   "w": add(w2), "b": add(b2)})
    return layers, np.concatenate(chunks)


def export_samples(classes):
    if not (RAW_DIR.is_dir() and any(RAW_DIR.glob("*.npy"))):
        return None
    rng = np.random.default_rng(0)
    out = {}
    for key in classes:
        path = RAW_DIR / f"{file_name(key)}.npy"
        if not path.exists():
            continue
        arr = np.load(path, mmap_mode="r")
        pick = rng.choice(len(arr), size=min(SAMPLES_PER_CLASS, len(arr)), replace=False)
        out[key] = [base64.b64encode(np.asarray(arr[i], np.uint8).tobytes()).decode() for i in pick]
    return out


def write(layers, weights, meta, samples):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.astype("<f4").tofile(OUT_DIR / "weights.bin")
    (OUT_DIR / "model.json").write_text(json.dumps({**meta, "layers": layers, "n_weights": int(weights.size)}, indent=1))
    if samples:
        (OUT_DIR / "samples.json").write_text(json.dumps(samples, separators=(",", ":")))
    elif (OUT_DIR / "samples.json").exists():
        (OUT_DIR / "samples.json").unlink()


def main():
    from .utils import load_model                          # PyTorch is only needed here
    model, ckpt = load_model(MODEL_PATH, "cpu")
    state = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    classes = list(ckpt["classes"])
    layers, weights = pack(state)
    names = {k: (name, em) for k, name, em in CATEGORIES}
    meta = {
        "classes": classes,
        "categories": [{"key": k, "name": names.get(k, (k, ""))[0], "emoji": names.get(k, ("", "❓"))[1]} for k in classes],
        "val_acc": ckpt.get("val_acc"), "val_top3": ckpt.get("val_top3"), "quick": bool(ckpt.get("quick", False)),
        "input": {"size": 28, "scale": 1 / 255},
    }
    samples = export_samples(classes)
    write(layers, weights, meta, samples)
    size = (OUT_DIR / "weights.bin").stat().st_size / 1e6
    print(f"Exported {weights.size:,} weights ({size:.1f} MB) and {len(layers)} layers to {OUT_DIR}")
    print("Sample drawings: " + (f"{sum(len(v) for v in samples.values())} saved" if samples else
                                 "none (download the dataset first to include them)"))
    print("\nTry it locally:   python -m http.server 8000 --directory docs   then open http://localhost:8000")
    print("Publish: push the project to GitHub, then Settings -> Pages -> Deploy from a branch -> main, /docs")


if __name__ == "__main__":
    main()
