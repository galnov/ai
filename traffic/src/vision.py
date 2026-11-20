"""
Using the trained camera network: images -> count + density maps.
"""
import numpy as np
import torch

from . import config as C
from .counter import CounterCNN


class Counter:
    def __init__(self, path=C.COUNTER_PATH, device="cpu"):
        if not path.exists():
            raise FileNotFoundError(f"The camera network was not found at {path}. Run:  python -m src.train_counter")
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.model = CounterCNN(ckpt["width"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(device).eval()
        self.device = device
        self.ckpt = ckpt

    @torch.no_grad()
    def count(self, images):
        """images: uint8 (N,H,W) -> (counts (N,), density maps (N,h,w))."""
        x = torch.from_numpy(np.ascontiguousarray(images)).to(self.device).float().div_(255.0).unsqueeze(1)
        counts, dens = self.model(x)
        return counts.cpu().numpy(), dens[:, 0].cpu().numpy()
