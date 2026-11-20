"""
Neural network definition - a CNN (convolutional neural network) that classifies emotions.

Input:   a grayscale face image, 1x48x48
Output:  7 values (logits) - one per emotion. softmax turns them into probabilities.

Structure:
  4 convolution blocks. Each block:
     Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout
  48x48 -> 24x24 -> 12x12 -> 6x6 -> 3x3   (resolution goes down, number of channels goes up)
  Global Average Pooling -> fully connected layer (256) -> Dropout -> output layer (7)

The first blocks learn simple features (edges, lines); the deeper ones learn
complex features (mouth shape, raised eyebrows, wide-open eyes).
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return self.drop(self.pool(x))


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, width=64, dropout=0.4):
        """
        width - number of channels in the first block (64 = more accurate, 32 = ~4x faster).
        """
        super().__init__()
        w = width
        self.block1 = ConvBlock(1, w, 0.10)
        self.block2 = ConvBlock(w, w * 2, 0.15)
        self.block3 = ConvBlock(w * 2, w * 4, 0.20)
        self.block4 = ConvBlock(w * 4, w * 8, 0.25)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(self.gap(x))

    @property
    def cam_layer(self):
        """The layer Grad-CAM looks at: the last convolution (6x6), before pooling."""
        return self.block4.act2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = EmotionCNN()
    out = m(torch.randn(2, 1, 48, 48))
    print(m)
    print("output shape:", tuple(out.shape))
    print(f"parameters: {count_parameters(m):,}")
