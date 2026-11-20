"""
The web app: a control center for a smart intersection.

    python -m app.server
    then open in the browser:  http://localhost:5000
"""
import argparse

from flask import Flask, jsonify, render_template, request

from src import config as C

from .world import PRESETS, World


def load_models():
    controller = counter = None
    errors = []
    try:
        from src.agent import Controller
        controller = Controller()
    except Exception as e:
        errors.append(str(e))
    try:
        from src.vision import Counter
        counter = Counter()
    except Exception as e:
        errors.append(str(e))
    return controller, counter, errors


def create_app(controller=None, counter=None, errors=None):
    app = Flask(__name__)
    if controller is None and counter is None and errors is None:
        controller, counter, errors = load_models()
    world = World(controller, counter)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/info")
    def info():
        ac = getattr(controller, "ckpt", {}) or {}
        cc = getattr(counter, "ckpt", {}) or {}
        return jsonify({
            "errors": errors or [],
            "has_agent": controller is not None,
            "has_counter": counter is not None,
            "agent": {"val_wait": ac.get("val_wait"), "baselines": ac.get("baselines")},
            "counter": {"val_mae": cc.get("val_mae")},
            "presets": [{"id": k, "name": v[0]} for k, v in PRESETS.items()],
            "conditions": [{"id": c, "name": C.CONDITION_NAMES[c]} for c in C.CONDITIONS],
            "lanes": [C.LANE_NAMES[l] for l in C.LANES],
            "dt": C.DT,
            "view_range": C.VIEW_RANGE,
            "vehicle_lengths": [v[1] for v in C.VEHICLES],
            "settings": {"preset": world.preset, "custom": world.custom, "left_mode": world.left_mode,
                         "sensor": world.sensor, "condition": world.condition},
        })

    @app.post("/api/config")
    def config():
        d = request.get_json(silent=True) or {}
        with world.lock:
            if d.get("preset") in PRESETS:
                world.preset = d["preset"]
            if isinstance(d.get("custom"), list) and len(d["custom"]) == 4:
                world.custom = [min(0.4, max(0.0, float(v))) for v in d["custom"]]
            if d.get("left_mode") in ("fixed", "max_queue"):
                world.left_mode = d["left_mode"]
            if d.get("sensor") in ("camera", "perfect"):
                world.sensor = d["sensor"] if (d["sensor"] == "perfect" or counter is not None) else "perfect"
            if d.get("condition") in C.CONDITIONS:
                world.condition = d["condition"]
        return jsonify({"ok": True, "sensor": world.sensor})

    @app.post("/api/reset")
    def reset():
        world.reset()
        return jsonify(world.state())

    @app.post("/api/step")
    def step():
        d = request.get_json(silent=True) or {}
        n = max(0, min(int(d.get("n", 1)), 40))
        st = world.step(n)
        if world.camera and d.get("cam_seq") != world.cam_seq:
            st["camera"] = world.camera
        return jsonify(st)

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
