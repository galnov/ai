"""
Export the trained drivers for the static web page (docs/), so it can be published on GitHub Pages.

    python -m src.export_web

GitHub Pages only serves files - it can't run Python or PyTorch. In the web version everything runs in the
visitor's browser, in JavaScript: building tracks (from a drawing or at random), the sensors and the physics
(docs/backend.js, a copy of src/track.py, src/env.py and app/sim.py), and the networks (docs/mlp.js).
This script writes what the page needs into docs/:

    model/weights.bin   all the numbers of the networks (float32)
    model/model.json    the layers of every driver (the trained one and the learning stages), and for each
                        one the neurons drawn in "the brain" panel (picked here, exactly like the server does)
    tracks.json         the preset tracks from tracks/

Without docs/model the page still works: you can draw tracks and drive yourself.
"""
import json

import numpy as np

from . import config as C
from .track import Track

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


def pack_driver(packer, state, brain, net_id, label):
    """state: the QNetwork's state_dict as NumPy arrays (fc1, fc2, out). brain: from app.sim.pick_neurons."""
    layers = [packer.dense(state["fc1.weight"], state["fc1.bias"], "relu"),
              packer.dense(state["fc2.weight"], state["fc2.bias"], "relu"),
              packer.dense(state["out.weight"], state["out.bias"], "none")]
    brain = {k: (np.asarray(v).tolist() if k.endswith("_idx") else v) for k, v in brain.items()}
    return {"id": net_id, "label": label, "layers": layers, "brain": brain}


def write(networks, packer):
    weights = packer.weights()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.astype("<f4").tofile(OUT_DIR / "weights.bin")
    doc = {"networks": networks, "n_weights": int(weights.size), "obs_size": C.OBS_SIZE, "actions": C.ACTION_NAMES}
    (OUT_DIR / "model.json").write_text(json.dumps(doc, indent=1))
    return weights.size


def write_tracks():
    """The preset tracks, for the page's track menu."""
    out = []
    for p in sorted(C.TRACKS_DIR.glob("*.json")):
        t = Track.load(p)
        out.append({"id": p.stem, "name": t.name, "width": t.width, "center": np.round(t.center, 2).tolist()})
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "tracks.json").write_text(json.dumps(out, separators=(",", ":")))
    return len(out)


def export(drivers):
    """drivers: [(id, label, driver)], where driver has .net (with fc1/fc2/out) - like src.agent.Driver."""
    from app.sim import pick_neurons
    packer, networks = Packer(), []
    for net_id, label, d in drivers:
        state = {k: v.detach().cpu().numpy() for k, v in d.net.state_dict().items()}
        networks.append(pack_driver(packer, state, pick_neurons(d), net_id, label))
    return write(networks, packer), len(networks)


def main():
    print(f"Wrote {write_tracks()} preset tracks to {DOCS_DIR / 'tracks.json'}")
    if not C.MODEL_PATH.exists():
        raise SystemExit(f"No trained driver at {C.MODEL_PATH}. Train first:  python -m src.train")
    from .agent import Driver                              # PyTorch is only needed here
    drivers = []
    for p in sorted((C.MODELS_DIR / "snapshots").glob("step_*.pt")):        # the learning stages, like the server
        d = Driver(p)
        drivers.append((p.stem, f"After {d.ckpt.get('updates', 0):,} updates", d))
    drivers.append(("best", "Trained driver", Driver(C.MODEL_PATH)))
    n, k = export(drivers)
    print(f"Exported {k} drivers ({n:,} numbers, {n * 4 / 1e3:.0f} KB) to {OUT_DIR}")
    print("\nTry it locally:   python -m http.server 8000 --directory docs   then open http://localhost:8000")


if __name__ == "__main__":
    main()
