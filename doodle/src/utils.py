import torch

from .config import CLASS_NAMES, MODEL_PATH
from .model import DoodleCNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")   # Mac computers with Apple silicon
    return torch.device("cpu")


def to_input(x_uint8, device):
    """uint8 (B,28,28) -> float (B,1,28,28) in the range 0..1. Must be identical in training and inference."""
    return x_uint8.to(device).float().div_(255.0).unsqueeze(1)


def save_checkpoint(model, path, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": list(CLASS_NAMES), **extra}, path)


def load_model(path=MODEL_PATH, device="cpu"):
    if not path.exists():
        raise FileNotFoundError(f"No trained model found at {path}. Run first:  python -m src.train")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = DoodleCNN(num_classes=len(ckpt["classes"]), width=ckpt["width"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt
