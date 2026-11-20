"""
PyTorch Dataset + data augmentation + class weights.
"""
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import EMOTIONS, IMG_SIZE
from .data_sources import class_counts, load_splits

# normalize to [-1, 1]. Must be identical in training and inference (predictor.py)
MEAN, STD = 0.5, 0.5


def train_transform():
    """
    Data augmentation: in every epoch the network sees a slightly different version of each image,
    so it learns to generalize instead of "memorizing" the training set.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),            # mirrored face - same emotion
        transforms.RandomRotation(10),                # slight head tilt
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # different lighting
        transforms.ToTensor(),
        transforms.Normalize([MEAN], [STD]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),  # partial occlusion (hand, glasses)
    ])


def eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([MEAN], [STD]),
    ])


class EmotionDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source, label = self.samples[idx]
        if isinstance(source, np.ndarray):
            img = Image.fromarray(source)  # 2-D uint8 array -> grayscale image
        else:
            img = Image.open(source).convert("L")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return self.transform(img), label


def compute_class_weights(train_samples):
    """
    FER-2013 is imbalanced ('disgust' is very rare). A higher weight for rare classes
    makes the network "pay attention" to them. We use the square root so we don't overdo it.
    """
    counts = np.array(class_counts(train_samples), dtype=np.float64)
    counts = np.maximum(counts, 1)
    weights = np.sqrt(counts.sum() / (len(EMOTIONS) * counts))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_loaders(batch_size=64, workers=2, seed=42, limit=None):
    splits = load_splits(seed=seed)
    if limit:  # quick-test mode
        rng = np.random.default_rng(seed)
        for k in splits:
            idx = rng.permutation(len(splits[k]))[:limit]
            splits[k] = [splits[k][i] for i in idx]

    train_ds = EmotionDataset(splits["train"], train_transform())
    val_ds = EmotionDataset(splits["val"], eval_transform())
    test_ds = EmotionDataset(splits["test"], eval_transform())

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin, drop_last=True,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=workers, pin_memory=pin)
    return splits, train_loader, val_loader, test_loader
