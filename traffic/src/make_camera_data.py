"""
Collecting data for the camera network: run the simulation, "photograph" the intersection and save every image
with its label: how many cars are really in the lane.

    python -m src.make_camera_data                 # 8,000 shots x 4 cameras = 32,000 images
    python -m src.make_camera_data --frames 2000   # faster, for a check

To make the data varied:
  * intersections with random demand, and traffic lights that behave differently (fixed, smart, random) - short and long queues
  * 40 "intersections" with different buildings, trees and asphalt colors
  * day (40%), dusk, night and rain (20% each)
  * small camera movements, noise and compression

The test set is shot at 10 intersections that don't appear in training at all, to check that the network learned "to count
cars" and not "to remember the background".

Output: data/camera_dataset.npz  (+ reports/camera_examples.png)
"""
import argparse
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .camera import Scene, lane_counts, render_cameras, vehicle_list
from .traffic import TrafficEnv, fixed_time, max_queue

TRAIN_SCENES = list(range(40))
TEST_SCENES = list(range(100, 110))
COND_P = [0.4, 0.2, 0.2, 0.2]

_scenes = {}


def collect_states(n_frames, seed):
    """Runs intersections with different controllers and saves "state snapshots" (lists of vehicles)."""
    rng = np.random.default_rng(seed)
    E = 16
    env = TrafficEnv(E, seed=seed)
    # for each intersection: a controller type. Some are bad on purpose, so there are long queues too
    kind = rng.integers(0, 3, E)
    green = rng.uniform(15, 60, E)
    states = []
    while len(states) < n_frames:
        a = np.where(kind == 0, (env.green_time >= green).astype(int),
                     np.where(kind == 1, max_queue(env), (rng.random(E) < 0.15).astype(int)))
        _, done = env.step(a)
        for e in np.flatnonzero(rng.random(E) < 0.08):
            states.append(vehicle_list(env, e))
        if done.any():
            env.reset(done)
            kind[done] = rng.integers(0, 3, done.sum())
            green[done] = rng.uniform(15, 60, done.sum())
    return states[:n_frames]


def _render_job(args):
    vehicles, scene_id, cond, seed = args
    if scene_id not in _scenes:
        _scenes[scene_id] = Scene(scene_id)
    rng = np.random.default_rng(seed)
    views = render_cameras(_scenes[scene_id], vehicles, C.CONDITIONS[cond], rng, jitter=1.0)
    return np.stack([v[1] for v in views]), lane_counts(vehicles)


def build(frames, scenes, seed, workers):
    rng = np.random.default_rng(seed)
    states = collect_states(frames, seed)
    jobs = [(st, int(rng.choice(scenes)), int(rng.choice(4, p=COND_P)), int(rng.integers(1 << 31)))
            for st in states]
    with Pool(workers) as pool:
        res = pool.map(_render_job, jobs, chunksize=16)
    X = np.concatenate([r[0] for r in res])                          # (frames*4, H, W)
    y = np.concatenate([r[1] for r in res]).astype(np.float32)
    cond = np.repeat([j[2] for j in jobs], C.N_LANES).astype(np.int8)
    lane = np.tile(np.arange(C.N_LANES), len(jobs)).astype(np.int8)
    return X, y, cond, lane


def save_examples(path):
    rng = np.random.default_rng(3)
    states = collect_states(40, 99)
    fig, axes = plt.subplots(2, 8, figsize=(16, 7))
    for i, ax in enumerate(axes.flat):
        st = states[i * 2 + 5]
        cond = C.CONDITIONS[i % 4]
        lane = i // 4 % 4
        view, _ = render_cameras(Scene(i), st, cond, rng, lanes=[lane])[0]
        ax.imshow(view)
        ax.set_title(f"{cond}, lane {C.LANES[lane]}: {lane_counts(st)[lane]} cars", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Create a dataset of labeled images from the cameras")
    ap.add_argument("--frames", type=int, default=C.CAMERA_FRAMES)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    args = ap.parse_args()
    t0 = time.time()
    n_test = max(200, args.frames // 10)
    n_val = max(200, args.frames // 10)
    print(f"Shooting {args.frames + n_test:,} states x 4 cameras ({args.workers} processes in parallel)...")
    Xtr, ytr, ctr, ltr = build(args.frames, TRAIN_SCENES, 1, args.workers)
    Xte, yte, cte, lte = build(n_test, TEST_SCENES, 2, args.workers)
    # validation: part of the training shots (same intersections, different states)
    idx = np.random.default_rng(0).permutation(len(ytr))
    va, tr = idx[:n_val * 4], idx[n_val * 4:]
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(C.CAMERA_DATA,
             X_train=Xtr[tr], y_train=ytr[tr], cond_train=ctr[tr], lane_train=ltr[tr],
             X_val=Xtr[va], y_val=ytr[va], cond_val=ctr[va], lane_val=ltr[va],
             X_test=Xte, y_test=yte, cond_test=cte, lane_test=lte)
    print(f"Saved {len(tr):,} training images, {len(va):,} validation, {len(yte):,} test  ({time.time() - t0:.0f} seconds)")
    print(f"Label distribution in training: mean {ytr.mean():.1f}, max {ytr.max():.0f}, "
          f"empty lanes {np.mean(ytr == 0) * 100:.0f}%")
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_examples(C.REPORTS_DIR / "camera_examples.png")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(ytr, bins=np.arange(-0.5, ytr.max() + 1.5), color="#3b6fb6")
    ax.set_xlabel("cars in lane (label)")
    ax.set_ylabel("images")
    ax.set_title("Label distribution (training set)")
    fig.tight_layout()
    fig.savefig(C.REPORTS_DIR / "camera_label_hist.png", dpi=100)
    plt.close(fig)
    print("Next step:  python -m src.train_counter")


if __name__ == "__main__":
    main()
