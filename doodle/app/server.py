"""
The "Guess What I'm Drawing" web app.

    python -m app.server
    then open in the browser:  http://localhost:5000
"""
import argparse

import numpy as np
from flask import Flask, jsonify, render_template, request

from src.config import CATEGORIES, CUSTOM_DIR, IMG_SIZE, RAW_DIR, file_name
from src.render import strokes_to_image

TOP_K = 5


def create_app(predictor=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
    error = None
    if predictor is None:
        try:
            from src.predictor import DoodlePredictor
            predictor = DoodlePredictor()
        except Exception as e:
            error = str(e)
            print("[!] Could not load the model:", e)
    keys = [c[0] for c in CATEGORIES]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/info")
    def info():
        ck = getattr(predictor, "ckpt", {}) or {}
        return jsonify({
            "ready": predictor is not None,
            "error": error,
            "val_acc": ck.get("val_acc"),
            "val_top3": ck.get("val_top3"),
            "quick": ck.get("quick", False),
            "categories": [{"key": k, "name": name, "emoji": em} for k, name, em in CATEGORIES],
            "has_samples": RAW_DIR.is_dir() and any(RAW_DIR.glob("*.npy")),
        })

    def parse_strokes(data):
        strokes = data.get("strokes") or []
        clean = []
        for s in strokes[:200]:
            if isinstance(s, list) and len(s) == 2 and len(s[0]) == len(s[1]) and len(s[0]) > 0:
                clean.append([[float(v) for v in s[0][:2000]], [float(v) for v in s[1][:2000]]])
        return clean

    @app.post("/api/predict")
    def predict():
        if predictor is None:
            return jsonify({"error": error or "Model not loaded"}), 503
        strokes = parse_strokes(request.get_json(silent=True) or {})
        img, probs = predictor.predict_strokes(strokes)
        if probs is None:
            return jsonify({"top": [], "image": img.ravel().tolist()})
        order = np.argsort(-probs)[:TOP_K]
        return jsonify({
            "top": [{"key": keys[i], "p": round(float(probs[i]), 4)} for i in order],
            "image": img.ravel().tolist(),
        })

    @app.post("/api/save")
    def save():
        """Save a drawing as a training example (collecting your own data)."""
        data = request.get_json(silent=True) or {}
        label = data.get("label")
        if label not in keys:
            return jsonify({"error": "Unknown category"}), 400
        img = strokes_to_image(parse_strokes(data))
        if img.max() == 0:
            return jsonify({"error": "The drawing is empty"}), 400
        CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
        path = CUSTOM_DIR / f"{file_name(label)}.npy"
        arr = np.load(path) if path.exists() else np.zeros((0, IMG_SIZE * IMG_SIZE), np.uint8)
        arr = np.vstack([arr, img.reshape(1, -1)])
        np.save(path, arr)
        total = sum(len(np.load(p, mmap_mode="r")) for p in CUSTOM_DIR.glob("*.npy"))
        return jsonify({"saved": int(len(arr)), "total": int(total)})

    @app.get("/api/samples/<int:idx>")
    def samples(idx):
        """Real drawings from the dataset - how other people drew this category."""
        if not 0 <= idx < len(keys):
            return jsonify({"error": "Category does not exist"}), 404
        path = RAW_DIR / f"{file_name(keys[idx])}.npy"
        if not path.exists():
            return jsonify({"images": []})
        arr = np.load(path, mmap_mode="r")
        pick = np.random.default_rng().choice(len(arr), size=min(12, len(arr)), replace=False)
        return jsonify({"images": [np.asarray(arr[i]).tolist() for i in pick]})

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
