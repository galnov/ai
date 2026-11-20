import random

import numpy as np
import torch

from .config import EMOTIONS, IMG_SIZE, MODEL_PATH
from .model import EmotionCNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")  # Mac computers with Apple silicon
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(model, width, path=MODEL_PATH, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "width": width,
        "emotions": list(EMOTIONS),
        "img_size": IMG_SIZE,
        **extra,
    }, path)


def load_model(path=MODEL_PATH, device="cpu"):
    if not path.exists():
        raise FileNotFoundError(f"No trained model found at {path}. Run first: python -m src.train")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = EmotionCNN(num_classes=len(ckpt["emotions"]), width=ckpt["width"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
