"""
Turning a user's drawing (a list of strokes) into a 28x28 image - exactly like the Quick Draw images.

This is the most important part for recognition quality: the network learns from images that went through
a specific processing, so the user's drawing must go through exactly the same processing. Google did this (according to the documentation and the code they published):
  1. "Simplification": move the drawing so its top-left corner is at (0,0), and scale it
     so the largest dimension is 255. Simplify each stroke with the Ramer-Douglas-Peucker algorithm (ε=2).
  2. Rendering: center on a 256x256 canvas, add padding (16) and draw with width 16 and round caps,
     then shrink to 28x28. White lines on a black background.

Input format: a list of strokes, each stroke is [[x1, x2, ...], [y1, y2, ...]]  (as in Quick Draw).
"""
import numpy as np
from PIL import Image, ImageDraw

from .config import IMG_SIZE

ORIGINAL_SIDE = 256.0
LINE_DIAMETER = 16.0
PADDING = 16.0
SUPERSAMPLE = 8          # draw at 8x the resolution and then shrink - for smooth edges (antialiasing)


def rdp(points, epsilon):
    """Ramer-Douglas-Peucker: keeps only the points that matter for the shape of the stroke."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    d = end - start
    norm = np.hypot(*d)
    if norm == 0:
        dists = np.hypot(*(points - start).T)
    else:
        dists = np.abs(d[0] * (points[:, 1] - start[1]) - d[1] * (points[:, 0] - start[0])) / norm
    i = int(np.argmax(dists))
    if dists[i] > epsilon:
        left = rdp(points[: i + 1], epsilon)
        right = rdp(points[i:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def simplify(strokes, epsilon=2.0):
    """Step 1: align to the corner, scale to 255, and RDP simplification. Returns a list of (n,2) arrays."""
    strokes = [np.asarray(s, dtype=np.float64).T for s in strokes if len(s) == 2 and len(s[0]) > 0]
    if not strokes:
        return []
    allp = np.vstack(strokes)
    mn = allp.min(axis=0)
    span = (allp.max(axis=0) - mn).max()
    scale = 255.0 / span if span > 0 else 1.0
    return [rdp((s - mn) * scale, epsilon) for s in strokes]


def rasterize(simplified, side=IMG_SIZE):
    """Step 2: render like Quick Draw. Returns a uint8 array of size (side, side)."""
    big = side * SUPERSAMPLE
    img = Image.new("L", (big, big), 0)
    if not simplified:
        return np.zeros((side, side), np.uint8)
    draw = ImageDraw.Draw(img)
    total_padding = PADDING * 2 + LINE_DIAMETER
    k = big / (ORIGINAL_SIDE + total_padding)      # from 256-units to pixels
    bbox = np.vstack(simplified).max(axis=0)
    offset = (ORIGINAL_SIDE - bbox) / 2.0           # center the drawing
    w = LINE_DIAMETER * k
    r = w / 2
    for s in simplified:
        pts = (s + offset + total_padding / 2) * k
        xy = [tuple(p) for p in pts]
        if len(xy) > 1:
            draw.line(xy, fill=255, width=max(1, int(round(w))))
        for x, y in xy:                              # round caps and joints
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
    small = img.resize((side, side), Image.BOX)      # shrink with averaging = smoothing
    return np.array(small, dtype=np.uint8)


def strokes_to_image(strokes):
    return rasterize(simplify(strokes))
