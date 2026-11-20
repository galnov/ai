"""
Find and load the data from disk (no PyTorch).

Three sources are supported:
  1. data/fer2013.csv  (or data/fer2013/fer2013.csv)  - the original FER-2013 file
  2. data/fer2013/train , data/fer2013/test             - the folder version of FER-2013 from Kaggle
  3. data/custom/<emotion>/*.png                         - images you collected yourself (collect_data.py)

Each sample is a pair (source, label):
  source = path to an image file  or  a 48x48 numpy array
  label  = class index according to config.EMOTIONS
"""
import csv
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .config import CSV_LABEL_TO_EMOTION, DATA_DIR, EMOTIONS, IMG_SIZE

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


def _folder_samples(root: Path):
    """Reads a folder structured as root/<emotion>/<image>."""
    samples = []
    for emotion_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = emotion_dir.name.lower()
        if name not in EMOTION_TO_IDX:
            print(f"  [!] Skipping unknown folder: {emotion_dir}")
            continue
        label = EMOTION_TO_IDX[name]
        for f in sorted(emotion_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                samples.append((f, label))
    return samples


def _csv_samples(csv_path: Path):
    """Reads fer2013.csv: columns emotion, pixels, Usage."""
    splits = {"train": [], "val": [], "test": []}
    usage_map = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE)
            emotion = CSV_LABEL_TO_EMOTION[int(row["emotion"])]
            split = usage_map.get(row.get("Usage", "Training"), "train")
            splits[split].append((pixels, EMOTION_TO_IDX[emotion]))
    return splits


def stratified_split(samples, val_fraction, seed):
    """Splits a list into two parts while keeping the class proportions."""
    rng = random.Random(seed)
    by_label = {}
    for s in samples:
        by_label.setdefault(s[1], []).append(s)
    a, b = [], []
    for items in by_label.values():
        rng.shuffle(items)
        n_val = int(round(len(items) * val_fraction))
        b.extend(items[:n_val])
        a.extend(items[n_val:])
    rng.shuffle(a)
    rng.shuffle(b)
    return a, b


def find_fer_source(data_dir: Path = DATA_DIR):
    for p in [data_dir / "fer2013.csv", data_dir / "fer2013" / "fer2013.csv"]:
        if p.exists():
            return "csv", p
    for root in [data_dir / "fer2013", data_dir]:
        if (root / "train").is_dir() and (root / "test").is_dir():
            return "folders", root
    return None, None


def load_splits(data_dir: Path = DATA_DIR, val_fraction: float = 0.1, seed: int = 42,
                use_custom: bool = True):
    """Returns a dict {'train': [...], 'val': [...], 'test': [...]}."""
    kind, path = find_fer_source(data_dir)
    if kind == "csv":
        print(f"Loading FER-2013 from CSV file: {path}")
        splits = _csv_samples(path)
    elif kind == "folders":
        print(f"Loading FER-2013 from folders: {path}")
        train_all = _folder_samples(path / "train")
        train, val = stratified_split(train_all, val_fraction, seed)
        splits = {"train": train, "val": val, "test": _folder_samples(path / "test")}
    else:
        splits = {"train": [], "val": [], "test": []}

    custom_dir = data_dir / "custom"
    if use_custom and custom_dir.is_dir():
        custom = _folder_samples(custom_dir)
        if custom:
            c_train, c_val = stratified_split(custom, 0.15, seed)
            splits["train"] += c_train
            splits["val"] += c_val
            print(f"Added {len(custom)} self-collected images (data/custom)")

    if not splits["train"]:
        raise FileNotFoundError(
            "No training data found!\n"
            "Download FER-2013 (see data/README.md) and put it in data/fer2013/train and data/fer2013/test,\n"
            "or the file data/fer2013.csv."
        )
    if not splits["test"]:
        print("  [!] No test set - using the validation set for testing too")
        splits["test"] = splits["val"]
    return splits


def class_counts(samples):
    c = Counter(label for _, label in samples)
    return [c.get(i, 0) for i in range(len(EMOTIONS))]


def summarize(splits):
    print(f"{'':10s}" + "".join(f"{e[:7]:>9s}" for e in EMOTIONS) + f"{'total':>9s}")
    for name, samples in splits.items():
        counts = class_counts(samples)
        print(f"{name:10s}" + "".join(f"{c:9d}" for c in counts) + f"{sum(counts):9d}")
