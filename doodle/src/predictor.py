"""
Using the trained network: a drawing (list of strokes) -> a 28x28 image -> a probability for each category.
"""
import threading

import numpy as np
import torch

from .config import MODEL_PATH
from .render import strokes_to_image
from .utils import load_model, to_input


class DoodlePredictor:
    def __init__(self, path=MODEL_PATH):
        self.device = torch.device("cpu")   # for a single drawing the CPU is fast enough
        self.model, self.ckpt = load_model(path, self.device)
        self.classes = self.ckpt["classes"]
        self._lock = threading.Lock()

    @torch.no_grad()
    def predict_image(self, img):
        x = to_input(torch.from_numpy(img[None]), self.device)
        with self._lock:
            logits = self.model(x) + self.model(torch.flip(x, dims=[3]))
        return torch.softmax(logits / 2, dim=1)[0].numpy()

    def predict_strokes(self, strokes):
        img = strokes_to_image(strokes)
        if img.max() == 0:
            return img, None
        return img, self.predict_image(img)
