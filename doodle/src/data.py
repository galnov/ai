"""
Loading the downloaded data and splitting it into training / validation / test.
"""
import numpy as np

from .config import (CATEGORIES, CUSTOM_DIR, IMG_SIZE, PER_CLASS, RAW_DIR, TEST_PER_CLASS,
                     VAL_PER_CLASS, file_name)


def load_splits(per_class=PER_CLASS, seed=42, use_custom=True):
    """
    Returns {'train': (X, y), 'val': (X, y), 'test': (X, y)}
    X: uint8 with shape (N, 28, 28), y: int64.
    From each category: 1,000 for validation, 1,000 for testing, and the rest for training.
    """
    rng = np.random.default_rng(seed)
    parts = {"train": ([], []), "val": ([], []), "test": ([], [])}
    missing = []
    for label, (name, _, _) in enumerate(CATEGORIES):
        path = RAW_DIR / f"{file_name(name)}.npy"
        if not path.exists():
            missing.append(name)
            continue
        arr = np.load(path)[:per_class]
        arr = arr[rng.permutation(len(arr))]
        n_val = min(VAL_PER_CLASS, len(arr) // 10)
        n_test = min(TEST_PER_CLASS, len(arr) // 10)
        chunks = {"val": arr[:n_val], "test": arr[n_val:n_val + n_test], "train": arr[n_val + n_test:]}
        for split, x in chunks.items():
            parts[split][0].append(x)
            parts[split][1].append(np.full(len(x), label, dtype=np.int64))

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} categories are missing ({', '.join(missing[:5])}...).\n"
            "Run first:  python -m src.download_data")

    if use_custom and CUSTOM_DIR.is_dir():
        added = 0
        for label, (name, _, _) in enumerate(CATEGORIES):
            path = CUSTOM_DIR / f"{file_name(name)}.npy"
            if path.exists():
                x = np.load(path).reshape(-1, IMG_SIZE * IMG_SIZE)
                parts["train"][0].append(x)
                parts["train"][1].append(np.full(len(x), label, dtype=np.int64))
                added += len(x)
        if added:
            print(f"Added {added} drawings saved from the app (data/custom) to training")

    out = {}
    for split, (xs, ys) in parts.items():
        X = np.concatenate(xs).reshape(-1, IMG_SIZE, IMG_SIZE)
        y = np.concatenate(ys)
        order = rng.permutation(len(y))
        out[split] = (X[order], y[order])
    return out


def summarize(splits):
    for k, (X, y) in splits.items():
        print(f"  {k:6s} {len(y):8,} drawings")
