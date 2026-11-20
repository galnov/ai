"""
Network 1: counting cars in a camera image (supervised learning, regression).

The network doesn't count directly. It produces a "density map": for every region of the image, how many cars are in it.
The sum of the map is the count. This way the network learns *where* the cars are (you can see this in the heat map in the app),
even though in training it only gets the final number.

Input:   a grayscale image 1 x 112 x 64 (values 0..1)
Output:  a density map 1 x 28 x 16, whose sum = the number of cars in the incoming lane

  Block 1: 112x64 -> 56x32   (32 channels)    edges, stripes, spots of light
  Block 2: 56x32  -> 28x16   (64 channels)    shapes of cars and headlights
  Block 3: 28x16             (128 channels)   context: where the incoming lane is, what's near and what's far
  "Dilated" convolutions - they see a large area without shrinking the map
  Head: a 1x1 convolution -> a non-negative density map
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(c_in, c_out, dilation=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=dilation, dilation=dilation, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class CounterCNN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            cbr(1, w), cbr(w, w), nn.MaxPool2d(2),
            cbr(w, 2 * w), cbr(2 * w, 2 * w), nn.MaxPool2d(2),
            cbr(2 * w, 4 * w), cbr(4 * w, 4 * w),
            cbr(4 * w, 4 * w, dilation=2), cbr(4 * w, 4 * w, dilation=4),
            nn.Dropout2d(0.1),
        )
        self.head = nn.Conv2d(4 * w, 1, 1)

    def forward(self, x):
        """Returns (count (B,), density map (B,1,28,16))."""
        density = F.softplus(self.head(self.features(x)))
        return density.sum(dim=(1, 2, 3)), density


def photometric_augment(x):
    """
    Data augmentation: changes in brightness, contrast and gamma, and noise - different cameras and lighting.
    (No horizontal flip: the incoming lane is always on the left.)
    """
    b = x.size(0)
    dev = x.device
    gamma = torch.exp(torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.35, 0.35))
    x = x.clamp(1e-4, 1) ** gamma
    contrast = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.75, 1.25)
    bright = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.08, 0.08)
    mean = x.mean(dim=(2, 3), keepdim=True)
    x = (x - mean) * contrast + mean + bright
    x = x + torch.randn_like(x) * torch.empty(b, 1, 1, 1, device=dev).uniform_(0, 0.03)
    return x.clamp(0, 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
