"""
The web app: a drone that learned to fly.

    python -m app.server
    then open in the browser:  http://localhost:5000
"""
import argparse

from flask import Flask, jsonify, render_template, request

from src import config as C

from .world import World


def load_policies():
    """The trained network + "snapshots" from early in training (for the learning stages)."""
    from src.ppo import DronePolicy
    policies, meta, value_fn = {}, [], None
    for p in sorted((C.MODELS_DIR / "snapshots").glob("upd_*.pt")):
        pol = DronePolicy(p)
        policies[p.stem] = pol
        meta.append({"name": p.stem, "label": f"After {pol.ckpt.get('steps', 0) // 1000:,}K steps"})
    if C.MODEL_PATH.exists():
        pol = DronePolicy(C.MODEL_PATH)
        policies["best"] = pol
        meta.append({"name": "best", "label": "Trained network"})
        value_fn = pol.value
    return policies, meta, value_fn


def create_app(policies=None, meta=None, value_fn=None):
    app = Flask(__name__)
    error = None
    if policies is None:
        try:
            policies, meta, value_fn = load_policies()
            if "best" not in policies:
                error = "There is no trained network yet, so the drone is flown by a PID controller. To train:  python -m src.train"
        except Exception as e:
            policies, meta, value_fn, error = {}, [], None, f"Could not load the network: {e}"
    world = World(policies, meta)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/info")
    def info():
        return jsonify({"error": error, "has_model": "best" in policies, "stages": len(meta) > 1,
                        "has_value": value_fn is not None, "world": [C.WORLD_W, C.WORLD_H],
                        "dt": C.CONTROL_DT, "arm": C.ARM})

    @app.post("/api/mode")
    def mode():
        world.set_mode((request.get_json(silent=True) or {}).get("mode", "free"))
        return jsonify(world.state())

    @app.post("/api/config")
    def config():
        d = request.get_json(silent=True) or {}
        world.configure(d.get("wind"), d.get("gust"), d.get("payload"), d.get("compare_pid"))
        return jsonify({"ok": True})

    @app.post("/api/click")
    def click():
        d = request.get_json(silent=True) or {}
        world.click(float(d.get("x", 10)), float(d.get("y", 6)))
        return jsonify({"ok": True})

    @app.post("/api/shape")
    def shape():
        world.set_shape((request.get_json(silent=True) or {}).get("name", "circle"))
        return jsonify({"ok": True})

    @app.post("/api/clear_route")
    def clear_route():
        world.clear_route()
        return jsonify({"ok": True})

    @app.post("/api/step")
    def step():
        n = max(0, min(int((request.get_json(silent=True) or {}).get("n", 1)), 25))
        return jsonify(world.step(n))

    @app.get("/api/value")
    def value():
        if value_fn is None:
            return jsonify({"error": "No trained network"}), 404
        return jsonify(world.value_map(value_fn))

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
