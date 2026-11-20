"""
A race track: a closed center line + a width.

From the center line we build two walls (right and left) - these are what the car's sensors "see".
A track can be created:
  * randomly    - random_track()   (for training: a new track every so often, so the agent learns to drive instead of memorizing a track)
  * from a drawing - from_drawing()  (the user draws in the app)
  * from a file - Track.load()
"""
import json

import numpy as np

from .config import TRACK_SPACING, TRACK_WIDTH_RANGE, WORLD_H, WORLD_W


# ---------------------------------------------------------------- geometry
def resample_closed(points, spacing=TRACK_SPACING):
    """Resamples a closed line so the distance between neighboring points is constant."""
    pts = np.asarray(points, dtype=np.float64)
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-6])[:-1]  # removes duplicate points
    pts = pts[keep]
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    n = max(16, int(round(total / spacing)))
    s = np.linspace(0, total, n, endpoint=False)
    x = np.interp(s, cum, closed[:, 0])
    y = np.interp(s, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


def smooth_closed(points, window=5, iters=1):
    """Circular moving average - smooths out hand "jitter" in a drawing."""
    pts = np.asarray(points, dtype=np.float64)
    k = np.ones(window) / window
    for _ in range(iters):
        pad = np.vstack([pts[-(window // 2):], pts, pts[:window // 2]])
        pts = np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(2)], axis=1)
    return pts


def curvature_radius(points, step=3):
    """Turning radius at each point (circle through 3 points). Small = sharp turn."""
    a = np.roll(points, step, axis=0)
    b = points
    c = np.roll(points, -step, axis=0)
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    return ab * bc * ca / (2 * cross + 1e-9)


def _segments_intersect(p, q):
    """Checks whether a closed line crosses itself (ignores neighboring segments)."""
    a, b = p, q
    n = len(a)
    d = b - a
    # all pairs i<j
    i, j = np.triu_indices(n, k=2)
    near = (j - i >= n - 1)          # the first and last segments are neighbors
    i, j = i[~near], j[~near]

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    denom = cross(d[i], d[j])
    ok = np.abs(denom) > 1e-12
    t = cross(a[j] - a[i], d[j]) / np.where(ok, denom, 1)
    u = cross(a[j] - a[i], d[i]) / np.where(ok, denom, 1)
    return bool(np.any(ok & (t > 0) & (t < 1) & (u > 0) & (u < 1)))


def self_intersects(points):
    return _segments_intersect(points, np.roll(points, -1, axis=0))


def parts_too_close(points, width):
    """
    Checks that two parts that are far apart along the track are not too close in space
    (otherwise their walls would overlap).
    """
    n = len(points)
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    gap = np.minimum(gap, n - gap) * TRACK_SPACING      # distance along the track
    far_along = gap > width * 2.0
    return bool(np.any(d[far_along] < width * 1.15))


def signed_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


# ---------------------------------------------------------------- the track
class Track:
    def __init__(self, center, width, name="track"):
        self.center = np.asarray(center, dtype=np.float64)
        self.width = float(width)
        self.name = name
        c = self.center
        nxt = np.roll(c, -1, axis=0)
        seg = nxt - c
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.cum = np.concatenate([[0], np.cumsum(self.seg_len)])  # distance along the track
        self.length = float(self.cum[-1])
        self.n = len(c)

        # the track direction at each point, and the normal (perpendicular) to it
        tangent = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        self.tangent = tangent
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        half = self.width / 2
        self.left = c + normal * half
        self.right = c - normal * half

        # all wall segments - for the sensors
        self.wall_a = np.vstack([self.left, self.right])
        self.wall_b = np.vstack([np.roll(self.left, -1, axis=0), np.roll(self.right, -1, axis=0)])

    def heading_at(self, idx):
        t = self.tangent[idx % self.n]
        return float(np.arctan2(t[1], t[0]))

    def reversed(self):
        return Track(self.center[::-1].copy(), self.width, self.name)

    def to_dict(self):
        return {"name": self.name, "width": self.width, "center": np.round(self.center, 2).tolist()}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["center"]), d["width"], d.get("name", "track"))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


class TrackError(ValueError):
    pass


def _fit_to_world(points, width, fill=False):
    """Fits the track inside the world (if it sticks out). fill=True - also scales up to fill the world."""
    margin = width / 2 + 12
    lo, hi = points.min(axis=0), points.max(axis=0)
    size = hi - lo
    avail = np.array([WORLD_W - 2 * margin, WORLD_H - 2 * margin])
    s = min(avail / np.maximum(size, 1))
    if not fill:
        s = min(1.0, s)
    pts = (points - (lo + hi) / 2) * s + np.array([WORLD_W / 2, WORLD_H / 2])
    return pts


def make_track(points, width, name="track", max_smooth=30, fill=False):
    """
    Builds a valid track from raw points: resampling, smoothing until there are no turns too
    sharp for this width, and checking that the track doesn't cross itself.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 8:
        raise TrackError("The track is too short - draw a bigger loop")
    if fill:
        pts = _fit_to_world(pts, width, fill=True)
    pts = resample_closed(pts)
    pts = smooth_closed(pts, window=5, iters=2)
    min_r = width / 2 * 1.25
    for _ in range(max_smooth):
        if curvature_radius(pts).min() >= min_r:
            break
        pts = resample_closed(smooth_closed(pts, window=7, iters=2))
    pts = _fit_to_world(resample_closed(pts), width, fill=fill)
    if curvature_radius(pts).min() < width / 2 * 1.05:
        raise TrackError("The track has a turn that is too sharp. Draw wider turns or make the road narrower")
    if self_intersects(pts):
        raise TrackError("The track crosses itself. Draw a single loop without crossings")
    if parts_too_close(pts, width):
        raise TrackError("Parts of the track are too close to each other. Move them apart or make the road narrower")
    track = Track(pts, width, name)
    if track.length < 1200:
        raise TrackError("The track is too short - draw a bigger loop")
    return track


def from_drawing(points, width=95.0):
    return make_track(points, width, name="Drawing")


def _fourier_points(rng, n_harmonics):
    """An ellipse whose radius "wobbles" by a sum of cosine waves (a Fourier series) - gentle turns."""
    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    r = np.ones_like(theta)
    for k in range(2, n_harmonics + 1):
        r += rng.uniform(0, 0.42 / k ** 0.6) * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    r = np.clip(r, 0.35, None)
    rx, ry = rng.uniform(430, 520), rng.uniform(270, 330)
    return np.stack([rx * r * np.cos(theta), ry * r * np.sin(theta)], axis=1)


def _polygon_points(rng):
    """A random polygon around a center - after smoothing it has sharp turns that require braking."""
    k = rng.integers(7, 14)
    theta = np.sort(rng.uniform(0, 2 * np.pi, k))
    r = rng.uniform(0.45, 1.0, k)
    rx, ry = rng.uniform(460, 540), rng.uniform(290, 340)
    return np.stack([rx * r * np.cos(theta), ry * r * np.sin(theta)], axis=1)


def random_track(rng, width=None, n_harmonics=5, kind=None):
    """
    A random training track. Two kinds:
      fourier - gentle, flowing turns
      polygon - sharp turns (the driver must learn to brake)
    Tries again until a valid track is produced.
    """
    for _ in range(100):
        w = width or rng.uniform(*TRACK_WIDTH_RANGE)
        k = kind or ("polygon" if rng.random() < 0.6 else "fourier")
        pts = _polygon_points(rng) if k == "polygon" else _fourier_points(rng, n_harmonics)
        rot = rng.uniform(-0.25, 0.25)
        pts = pts @ np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
        try:
            t = make_track(pts, w, name="Random", max_smooth=8, fill=True)
        except TrackError:
            continue
        return t.reversed() if rng.random() < 0.5 else t  # both clockwise and counter-clockwise
    raise RuntimeError("Could not create a random track")
