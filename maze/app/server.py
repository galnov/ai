"""
The web app: watch the pilot fly through a 3D maze from the cockpit - or fly it yourself.

    python -m app.server
    then open in the browser:  http://localhost:5000

The AI's flights are computed here, in the same Python simulation the network was trained in, and sent to
the browser as a list of frames (position, orientation, speed, depth image) that it plays back.
When you fly yourself, the browser runs its own copy of the ship's physics (app/static/ship.js).
"""
import argparse
import threading
from collections import OrderedDict

from flask import Flask, jsonify, render_template, request

from src import config as C
from src.autopilot import autopilot
from src.flights import fly
from src.maze import Maze


def load_pilots():
    """The trained network, the snapshots from training, and the autopilot."""
    pilots, labels, message = OrderedDict(), OrderedDict(), None
    try:
        from src.ppo import Pilot
        if C.MODEL_PATH.exists():
            p = Pilot(C.MODEL_PATH)
            pilots["ppo"] = p
            labels["ppo"] = f"PPO network ({p.ckpt.get('steps', 0) / 1e6:.1f}M steps)"
            for path in sorted((C.MODELS_DIR / "snapshots").glob("upd_*.pt")):
                s = Pilot(path)
                key = f"stage:{path.stem}"
                pilots[key] = s
                labels[key] = f"Learning stage: after {s.ckpt.get('steps', 0) / 1e3:,.0f}K steps"
        else:
            message = ("There is no trained network yet, so the ship is flown by the hand-written autopilot. "
                       "To train:  python -m src.train")
    except Exception as e:  # e.g. PyTorch is not installed
        message = f"Could not load the network ({e}). The ship is flown by the autopilot."
    pilots["autopilot"] = autopilot
    labels["autopilot"] = "Autopilot (hand-written rules)"
    return pilots, labels, message


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    pilots, labels, message = load_pilots()
    cache, lock = OrderedDict(), threading.Lock()

    def get_maze(seed, rooms):
        rooms = max(2, min(7, int(rooms)))
        return Maze(int(seed) & 0x7FFFFFFF, rooms)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/info")
    def info():
        return jsonify({"pilots": [{"id": k, "label": v} for k, v in labels.items()], "message": message,
                        "physics": {"cell": C.CELL, "radius": C.SHIP_RADIUS, "v_min": C.V_MIN, "v_max": C.V_MAX,
                                    "accel_k": C.ACCEL_K, "pitch_rate": C.PITCH_RATE, "yaw_rate": C.YAW_RATE,
                                    "dt": C.DT, "substeps": C.SUBSTEPS,
                                    "rays_x": C.RAYS_X, "rays_y": C.RAYS_Y, "fov": C.FOV, "ray_range": C.RAY_RANGE}})

    @app.get("/api/maze")
    def maze():
        return jsonify(get_maze(request.args.get("seed", 1), request.args.get("rooms", C.MAZE_ROOMS)).to_json())

    @app.post("/api/fly")
    def fly_maze():
        d = request.get_json(silent=True) or {}
        pid = d.get("pilot") if d.get("pilot") in pilots else next(iter(pilots))
        key = (int(d.get("seed", 1)) & 0x7FFFFFFF, int(d.get("rooms", C.MAZE_ROOMS)), pid)
        with lock:
            if key in cache:
                return jsonify(cache[key])
        m = get_maze(key[0], key[1])
        res = fly(pilots[pid], [m], record=True)
        out = {"pilot": pid, "label": labels[pid], "frames": res["frames"], "actions": res["actions"],
               "end": res["frames"][-1].get("end", "timeout"), "time": float(res["time"][0]),
               "progress": float(res["progress"][0]), "speed": float(res["speed"][0]),
               "path_length": m.path_length()}
        with lock:
            cache[key] = out
            while len(cache) > 40:
                cache.popitem(last=False)
        return jsonify(out)

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
