"""
The web app: a Flask server that receives webcam frames from the browser and returns the detected emotions.

    python -m app.server
    then open in the browser:  http://localhost:5000
"""
import argparse
import time
import traceback

from flask import Flask, jsonify, render_template, request

from src.config import EMOTION_COLOR, EMOTION_EMOJI, EMOTION_NAMES, EMOTIONS
from src.vision import decode_data_url, limit_size


def create_app(predictor=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024
    state = {"predictor": predictor, "error": None}

    if predictor is None:
        try:
            from src.predictor import EmotionPredictor
            state["predictor"] = EmotionPredictor()
            print(f"Model loaded (device: {state['predictor'].device})")
        except Exception as e:  # model not trained yet / missing library
            state["error"] = str(e)
            print("[!] Could not load the model:", e)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/info")
    def info():
        p = state["predictor"]
        return jsonify({
            "ready": p is not None,
            "error": state["error"],
            "val_acc": getattr(p, "val_acc", None),
            "train_samples": getattr(p, "train_samples", None),
            "epoch": getattr(p, "epoch", None),
            "device": str(getattr(p, "device", "")),
            "emotions": [{"key": e, "name": EMOTION_NAMES[e], "emoji": EMOTION_EMOJI[e],
                          "color": EMOTION_COLOR[e]} for e in EMOTIONS],
        })

    @app.post("/api/predict")
    def predict():
        p = state["predictor"]
        if p is None:
            return jsonify({"error": state["error"] or "Model not loaded"}), 503
        data = request.get_json(silent=True) or {}
        if "image" not in data:
            return jsonify({"error": "Missing image"}), 400
        t0 = time.perf_counter()
        try:
            img = limit_size(decode_data_url(data["image"]))
            faces = p.analyze(img, with_cam=bool(data.get("gradcam")))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception:
            traceback.print_exc()
            return jsonify({"error": "Error while processing the image"}), 500
        return jsonify({
            "faces": faces,
            "width": img.shape[1],
            "height": img.shape[0],
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    app = create_app()
    print(f"\n  Open in the browser:  http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
