"""
The web app: draw a track, watch the trained driver, race against it, and compare the learning stages.

    python -m app.server
    then open in the browser:  http://localhost:5000
"""
import argparse

import numpy as np
from flask import Flask, jsonify, render_template, request

from src import config as C
from src.track import TrackError, Track, from_drawing, random_track

from .sim import Sim

CAR_COLORS = ["#ffd23f", "#3bceac", "#ff6b6b", "#7aa2ff", "#c792ea", "#ff9f43", "#5ee27a", "#f78fb3"]


def load_drivers():
    """Loads the trained driver and the learning stages (if they exist)."""
    from src.agent import Driver  # requires PyTorch
    drivers, meta = {}, []
    snaps = sorted((C.MODELS_DIR / "snapshots").glob("step_*.pt"))
    for p in snaps:
        d = Driver(p)
        drivers[p.stem] = d
        meta.append({"name": p.stem, "label": f"After {d.ckpt.get('updates', 0):,} updates", "best": False})
    if C.MODEL_PATH.exists():
        d = Driver(C.MODEL_PATH)
        drivers["best"] = d
        meta.append({"name": "best", "label": "Trained driver", "best": True,
                     "val_score": d.ckpt.get("val_score"), "step": d.ckpt.get("step")})
    return drivers, meta


def track_json(t: Track):
    r = lambda a: np.round(a, 1).tolist()
    return {"name": t.name, "width": t.width, "length": round(t.length),
            "center": r(t.center), "left": r(t.left), "right": r(t.right)}


def create_app(drivers=None, meta=None):
    app = Flask(__name__)
    error = None
    if drivers is None:
        try:
            drivers, meta = load_drivers()
            if "best" not in drivers:
                error = "There is no trained driver yet. Run first:  python -m src.train"
        except Exception as e:
            drivers, meta, error = {}, [], f"Could not load the drivers: {e}"
    sim = Sim(drivers)
    rng = np.random.default_rng()
    presets = {p.stem: Track.load(p) for p in sorted(C.TRACKS_DIR.glob("*.json"))}
    first = next(iter(presets.values()), None) or random_track(rng)
    sim.set_track(first)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/info")
    def info():
        brains = {name: {k: v for k, v in b.items() if not k.endswith("_idx")} for name, b in sim.brain_cache.items()}
        return jsonify({
            "error": error,
            "drivers": meta,
            "brains": brains,
            "presets": [{"id": k, "name": t.name} for k, t in presets.items()],
            "track": track_json(sim.track),
            "world": [C.WORLD_W, C.WORLD_H],
            "car": [C.CAR_LENGTH, C.CAR_WIDTH],
            "ray_angles": C.RAY_ANGLES.round(4).tolist(),
            "ray_length": C.RAY_LENGTH,
            "max_speed": C.MAX_SPEED,
            "dt": C.DT,
            "actions": C.ACTION_NAMES,
            "colors": CAR_COLORS,
        })

    @app.post("/api/track")
    def set_track():
        data = request.get_json(silent=True) or {}
        try:
            if data.get("preset") in presets:
                t = presets[data["preset"]]
            elif data.get("points"):
                t = from_drawing(np.array(data["points"], dtype=float), float(data.get("width", 95)))
            else:
                t = random_track(rng)
        except TrackError as e:
            return jsonify({"error": str(e)}), 400
        sim.set_track(t)
        return jsonify({"track": track_json(t)})

    @app.post("/api/start")
    def start():
        """mode: watch | race | stages | solo"""
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "watch")
        cars = []
        if mode == "stages":
            for i, m in enumerate(meta):
                cars.append({"kind": "ai", "driver": m["name"], "label": m["label"],
                             "color": CAR_COLORS[(i + 2) % len(CAR_COLORS)] if not m["best"] else CAR_COLORS[0]})
        else:
            n_ai = int(data.get("n_ai", 1)) if "best" in drivers else 0
            for i in range(max(0, min(n_ai, 8))):
                cars.append({"kind": "ai", "driver": "best", "label": f"AI {i + 1}" if n_ai > 1 else "AI",
                             "color": CAR_COLORS[i % len(CAR_COLORS)]})
            if mode in ("race", "solo"):
                cars.append({"kind": "human", "driver": None, "label": "You", "color": "#ffffff"})
        sim.start(cars, spread=(mode == "watch"))
        return jsonify({"cars": [{k: c[k] for k in ("kind", "label", "color", "driver")} for c in cars]})

    @app.post("/api/step")
    def step():
        data = request.get_json(silent=True) or {}
        n = int(np.clip(int(data.get("n", 1)), 0, 12))
        return jsonify(sim.step(n, data.get("keys", {})))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    app = create_app()
    print(f"\n  Open in the browser:  http://localhost:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
