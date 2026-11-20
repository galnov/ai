"""
The neural network: a CNN that recognizes drawings.

Input:  a grayscale drawing 1x28x28 (values 0..1, white lines on black)
Output: a score for each of the 42 categories (softmax -> probabilities)

  Block 1:  28x28 -> 14x14   (32 channels)   lines and edges
  Block 2:  14x14 ->  7x7    (64 channels)   corners, circles, arcs
  Block 3:   7x7  ->  3x3    (128 channels)  parts: ears, wheels, leaves...
  Fully connected layer (256) -> output (42)
Each block: Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU -> MaxPool -> Dropout
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out, drop):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout2d(drop),
    )


class DoodleCNN(nn.Module):
    def __init__(self, num_classes, width=32, dropout=0.4):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_block(1, w, 0.05),
            conv_block(w, w * 2, 0.10),
            conv_block(w * 2, w * 4, 0.15),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 4 * 3 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def augment(x, max_rot=12, max_shift=0.12, max_zoom=0.12):
    """
    Data augmentation on the whole batch at once: random rotation, shift, zoom and horizontal flip.
    A drawing of a cat is still a cat even if it's a bit crooked, smaller, or facing the other way.
    x: (B,1,28,28)
    """
    b, dev = x.size(0), x.device                           # compute on the same device as the images
    rnd = lambda: torch.rand(b, device=dev) * 2 - 1
    ang = rnd() * math.radians(max_rot)
    zoom = 1 + rnd() * max_zoom
    flip = (torch.rand(b, device=dev) < 0.5).float() * 2 - 1   # -1 = horizontal flip
    cos, sin = torch.cos(ang) / zoom, torch.sin(ang) / zoom
    theta = torch.stack([
        torch.stack([cos * flip, -sin, rnd() * max_shift], dim=1),
        torch.stack([sin * flip, cos, rnd() * max_shift], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
