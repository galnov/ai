"""
Image processing with OpenCV (no PyTorch): face detection, cropping, heat maps.
"""
import base64
import os
from pathlib import Path

import cv2
import numpy as np

from .config import IMG_SIZE, MODELS_DIR

_CASCADE = None


def _cascade_candidates():
    name = "haarcascade_frontalface_default.xml"
    if os.environ.get("FACE_CASCADE"):          # option to override with your own file
        yield Path(os.environ["FACE_CASCADE"])
    yield MODELS_DIR / name                      # local copy in the models folder
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if data_dir:
        yield Path(data_dir) / name              # the file that ships with opencv-python 4.x


def face_detector():
    """A ready-made face detector (Haar Cascade) that ships with OpenCV."""
    global _CASCADE
    if _CASCADE is None:
        for path in _cascade_candidates():
            if path.exists():
                cascade = cv2.CascadeClassifier(str(path))
                if not cascade.empty():
                    _CASCADE = cascade
                    break
        else:
            raise RuntimeError(
                "Face detector file haarcascade_frontalface_default.xml not found.\n"
                "Install OpenCV version 4:  pip install \"opencv-python>=4.8,<5\"")
    return _CASCADE


def detect_faces(gray, max_faces=4):
    """Returns a list of rectangles (x, y, w, h), largest first."""
    h, w = gray.shape[:2]
    min_side = max(40, int(min(h, w) * 0.08))
    faces = face_detector().detectMultiScale(
        cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side))
    faces = sorted((tuple(int(v) for v in f) for f in faces), key=lambda b: b[2] * b[3], reverse=True)
    return faces[:max_faces]


def expand_box(box, img_w, img_h, margin=0.08):
    """Slightly enlarges the face box (FER images include a small margin too)."""
    x, y, w, h = box
    dx, dy = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(img_w, x + w + dx), min(img_h, y + h + dy)
    return x0, y0, x1 - x0, y1 - y0


def crop_face(gray, box):
    x, y, w, h = box
    face = gray[y:y + h, x:x + w]
    return cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def decode_data_url(data_url):
    """'data:image/jpeg;base64,....' -> an OpenCV BGR image."""
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the image")
    return img


def limit_size(img, max_side=1280):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def heatmap_png_b64(cam, size):
    """
    Converts a Grad-CAM map (0..1) into a transparent, colored PNG of size size=(w, h),
    so the browser can draw it on top of the face.
    """
    w, h = size
    cam = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    color = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (np.power(cam, 1.3) * 190).astype(np.uint8)
    bgra = np.dstack([color, alpha])
    ok, png = cv2.imencode(".png", bgra)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
