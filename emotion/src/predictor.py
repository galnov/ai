"""
Using the trained network on a new image: face detection -> crop -> network -> probabilities (+ Grad-CAM).
"""
import threading

import cv2
import numpy as np
import torch

from .config import MODEL_PATH
from .dataset import MEAN, STD
from .gradcam import GradCAM
from .utils import get_device, load_model
from .vision import crop_face, detect_faces, expand_box, face_detector, heatmap_png_b64


class EmotionPredictor:
    def __init__(self, model_path=MODEL_PATH, device=None):
        face_detector()  # check up front that the face detector is available
        self.device = device or get_device()
        self.model, ckpt = load_model(model_path, self.device)
        self.emotions = ckpt["emotions"]
        self.val_acc = ckpt.get("val_acc")
        self.train_samples = ckpt.get("train_samples")
        self.epoch = ckpt.get("epoch")
        self.gradcam = GradCAM(self.model, self.model.cam_layer)
        self._lock = threading.Lock()  # the web server handles several requests in parallel

    def _to_tensor(self, face48):
        x = torch.from_numpy(face48.astype(np.float32) / 255.0)
        x = (x - MEAN) / STD
        return x.view(1, 1, *face48.shape).to(self.device)

    def predict_face(self, face48, with_cam=False):
        x = self._to_tensor(face48)
        if with_cam:
            cam, probs = self.gradcam(x)
            return probs, cam
        with torch.no_grad():
            logits = self.model(x) + self.model(torch.flip(x, dims=[3]))  # TTA: image + mirror
            probs = torch.softmax(logits / 2, dim=1)[0].cpu().numpy()
        return probs, None

    def analyze(self, bgr, with_cam=False, max_faces=4):
        """Returns a list of detected faces, each with its position and a probability per emotion."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        results = []
        with self._lock:
            for box in detect_faces(gray, max_faces):
                crop_box = expand_box(box, W, H)
                probs, cam = self.predict_face(crop_face(gray, crop_box), with_cam)
                top = int(np.argmax(probs))
                item = {
                    "box": list(box),
                    "crop_box": list(crop_box),
                    "probs": {e: round(float(p), 4) for e, p in zip(self.emotions, probs)},
                    "top": self.emotions[top],
                    "confidence": round(float(probs[top]), 4),
                }
                if cam is not None:
                    item["heatmap"] = heatmap_png_b64(cam, (crop_box[2], crop_box[3]))
                results.append(item)
        return results
