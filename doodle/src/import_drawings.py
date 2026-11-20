"""
Add drawings saved on the web page (the GitHub Pages version) to the training data.

    python -m src.import_drawings my_drawings.json

On the web page, "Save as a training example" keeps drawings in the visitor's browser, and
"Download my drawings" saves them to my_drawings.json. This script appends them to data/custom/,
exactly like the save button of the local app does - the next training run uses them.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from .config import CATEGORIES, CUSTOM_DIR, IMG_SIZE, file_name


def main():
    ap = argparse.ArgumentParser(description="Import drawings downloaded from the web page")
    ap.add_argument("file", type=Path)
    args = ap.parse_args()
    data = json.loads(args.file.read_text())
    if data.get("format") != "doodle-drawings-v1":
        raise SystemExit("This doesn't look like a file from 'Download my drawings'.")
    keys = {c[0] for c in CATEGORIES}
    by_label, skipped = {}, 0
    for d in data["drawings"]:
        px = np.frombuffer(base64.b64decode(d["pixels"]), np.uint8)
        if d.get("label") not in keys or px.size != IMG_SIZE * IMG_SIZE or px.max() == 0:
            skipped += 1
            continue
        by_label.setdefault(d["label"], []).append(px)
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    for label, imgs in sorted(by_label.items()):
        path = CUSTOM_DIR / f"{file_name(label)}.npy"
        old = np.load(path) if path.exists() else np.zeros((0, IMG_SIZE * IMG_SIZE), np.uint8)
        np.save(path, np.vstack([old, np.stack(imgs)]))
        print(f"  {label:14s} +{len(imgs)}  (now {len(old) + len(imgs)})")
    print(f"Imported {sum(len(v) for v in by_label.values())} drawings into {CUSTOM_DIR}"
          + (f" ({skipped} skipped)" if skipped else ""))
    print("They will be used in the next training run:  python -m src.train")


if __name__ == "__main__":
    main()
