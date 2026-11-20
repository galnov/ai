"""
The intersection cameras: they produce images from the state of the simulation.

Each direction has a camera on a pole at the stop line, looking along the road toward the approaching cars
(like at real intersections). In the image the incoming lane is on the left, and the outgoing lane is on the right.

1. Draw the intersection from above: roads, crosswalks, buildings and cars.
2. For each camera, "tilt" the area it sees with a perspective warp: nearby cars are large,
   and distant ones are small and squeezed together.
3. Add shooting conditions: day, dusk, night (headlights only) or rain, plus camera noise and compression.

The simulation knows exactly how many cars are in each lane, so every image is labeled automatically.
This way we produce tens of thousands of labeled images without labeling a single one by hand.

World coordinates (meters): the center of the intersection is at (0,0), x points east, y points south. Traffic drives on the right.
"""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from . import config as C

STOP = 9.0                    # the stop line: 9 m from the center of the intersection
LANE_OFF = 1.75               # center of the incoming lane: 1.75 m from the center line
ROAD_HALF = 4.5
EXTENT = STOP + C.VIEW_RANGE + 12.0
PPM = 3.0                     # pixels per meter in the top-down drawing
TOP = int(2 * EXTENT * PPM)
WIDTH = {"car": 1.8, "truck": 2.5, "bus": 2.5}
KIND_NAMES = [v[0] for v in C.VEHICLES]
LENGTHS = [v[1] for v in C.VEHICLES]
CAR_COLORS = [(235, 235, 235), (200, 202, 205), (40, 40, 45), (25, 30, 60), (150, 20, 25),
              (30, 60, 140), (110, 112, 118), (60, 90, 70), (210, 180, 120), (230, 120, 30)]
BUS_COLORS = [(240, 200, 40), (235, 235, 235), (40, 120, 200)]
TRUCK_COLORS = [(235, 235, 235), (180, 40, 30), (60, 110, 160)]
GROUND = (118, 128, 104)

# the field of view of each camera, in local coordinates (u sideways, w distance from the center of the intersection):
# near - narrow (6.5 m to each side), far - wide (15 m to each side) = perspective
FAR_W = STOP + C.VIEW_RANGE + 3
VIEW_LOCAL = [(-15.0, FAR_W), (15.0, FAR_W), (6.5, STOP - 3.5), (-6.5, STOP - 3.5)]


def px(x, y):
    return (x + EXTENT) * PPM, (y + EXTENT) * PPM


def box(p, q):
    """A rectangle in pixels with ordered corners (PIL requires x0<=x1, y0<=y1)."""
    return [min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1])]


def local_to_world(lane, u, w):
    """The camera of lane `lane`: u to the right in the image, w away from the intersection."""
    if lane == 0:
        return u, -w            # from the north: looks north
    if lane == 1:
        return -u, w            # from the south: looks south
    if lane == 2:
        return w, u             # from the east: looks east
    return -w, -u               # from the west: looks west


def car_rect(lane, s, length, width):
    """The vehicle's rectangle (x0,y0,x1,y1) in meters, and the driving direction (dx,dy)."""
    a, b = STOP + s, STOP + s + length          # front / rear, distance from the center of the intersection
    w = width / 2
    if lane == 0:
        return (-LANE_OFF - w, -b, -LANE_OFF + w, -a), (0, 1)
    if lane == 1:
        return (LANE_OFF - w, a, LANE_OFF + w, b), (0, -1)
    if lane == 2:
        return (a, -LANE_OFF - w, b, -LANE_OFF + w), (-1, 0)
    return (-b, LANE_OFF - w, -a, LANE_OFF + w), (1, 0)


def _homography(src, dst):
    """PIL PERSPECTIVE coefficients: map a point in the output (dst) to a point in the input (src)."""
    A, bvec = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v]); bvec.append(x)
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v]); bvec.append(y)
    return tuple(np.linalg.solve(np.array(A, float), np.array(bvec, float)))


def view_coeffs(lane, rng=None, jitter=0.0):
    """A camera's perspective: the image corners -> the corners of the field of view in the top-down drawing."""
    pts = []
    for u, w in VIEW_LOCAL:
        if rng is not None and jitter:
            u += rng.uniform(-jitter, jitter)
            w += rng.uniform(-jitter, jitter) * (3 if w > 50 else 0.3)
        pts.append(px(*local_to_world(lane, u, w)))
    return _homography(pts, [(0, 0), (C.VIEW_W, 0), (C.VIEW_W, C.VIEW_H), (0, C.VIEW_H)])


class Scene:
    """The fixed background of the intersection (buildings, trees, roads, markings) - built once and kept."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.sun = rng.uniform(0, 2 * np.pi)
        img = Image.new("RGB", (TOP, TOP), GROUND)
        d = ImageDraw.Draw(img)
        sw = ROAD_HALF + 3.0
        for qx in (-1, 1):                      # buildings in the four quadrants
            for qy in (-1, 1):
                for _ in range(int(rng.integers(6, 14))):
                    x0 = qx * rng.uniform(sw + 2, EXTENT - 10)
                    y0 = qy * rng.uniform(sw + 2, EXTENT - 10)
                    w, h = rng.uniform(8, 30, 2)
                    x1 = qx * min(abs(x0) + w, EXTENT)
                    y1 = qy * min(abs(y0) + h, EXTENT)
                    shade = int(rng.uniform(110, 210))
                    tint = rng.integers(-18, 18, 3)
                    d.rectangle(box(px(x0, y0), px(x1, y1)),
                                fill=tuple(int(np.clip(shade + t, 0, 255)) for t in tint))
        d.rectangle(box(px(-sw, -EXTENT), px(sw, EXTENT)), fill=(168, 168, 162))     # sidewalks
        d.rectangle(box(px(-EXTENT, -sw), px(EXTENT, sw)), fill=(168, 168, 162))
        asphalt = tuple(int(v) for v in rng.integers(50, 74) + rng.integers(-3, 4, 3))
        d.rectangle(box(px(-ROAD_HALF, -EXTENT), px(ROAD_HALF, EXTENT)), fill=asphalt)
        d.rectangle(box(px(-EXTENT, -ROAD_HALF), px(EXTENT, ROAD_HALF)), fill=asphalt)
        line = (225, 225, 215)
        for t in np.arange(STOP + 2, EXTENT, 7.0):                       # dashed center line
            for sg in (-1, 1):
                d.rectangle(box(px(-0.1, sg * t), px(0.1, sg * (t + 3.5))), fill=line)
                d.rectangle(box(px(sg * t, -0.1), px(sg * (t + 3.5), 0.1)), fill=line)
        for sg in (-1, 1):                                                 # solid line near the intersection
            d.rectangle(box(px(-0.1, sg * STOP), px(0.1, sg * (STOP + 15))), fill=line)
            d.rectangle(box(px(sg * STOP, -0.1), px(sg * (STOP + 15), 0.1)), fill=line)
        white = (235, 235, 230)                                           # stop lines
        d.rectangle(box(px(-ROAD_HALF, -STOP - 0.3), px(0, -STOP + 0.3)), fill=white)
        d.rectangle(box(px(0, STOP - 0.3), px(ROAD_HALF, STOP + 0.3)), fill=white)
        d.rectangle(box(px(STOP - 0.3, -ROAD_HALF), px(STOP + 0.3, 0)), fill=white)
        d.rectangle(box(px(-STOP - 0.3, 0), px(-STOP + 0.3, ROAD_HALF)), fill=white)
        for t in np.arange(-ROAD_HALF + 0.3, ROAD_HALF, 1.2):             # crosswalks
            for c in (-6.5, 6.5):
                d.rectangle(box(px(t, c - 1.5), px(t + 0.6, c + 1.5)), fill=white)
                d.rectangle(box(px(c - 1.5, t), px(c + 1.5, t + 0.6)), fill=white)
        for _ in range(60):                                               # trees on the sidewalks
            along = rng.uniform(STOP + 8, EXTENT) * rng.choice([-1, 1])
            side = rng.choice([-1, 1]) * (sw + rng.uniform(1, 4))
            x, y = (side, along) if rng.random() < 0.5 else (along, side)
            r = rng.uniform(1.8, 3.2) * PPM
            cx, cy = px(x, y)
            green = (int(rng.integers(40, 70)), int(rng.integers(85, 120)), int(rng.integers(40, 60)))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=green)
        self.base = img
        lamps = []
        for t in np.arange(STOP + 5, EXTENT, 25):
            for sg in (-1, 1):
                lamps += [(ROAD_HALF + 1.2, sg * t), (-ROAD_HALF - 1.2, sg * t),
                          (sg * t, ROAD_HALF + 1.2), (sg * t, -ROAD_HALF - 1.2)]
        self.lamps = lamps


def vehicle_list(env, e):
    """The list of vehicles of intersection e: (lane, s, kind, id)."""
    out = []
    for l in range(C.N_LANES):
        act = np.isfinite(env.s[e, l])
        for s, k, i in zip(env.s[e, l][act], env.kind[e, l][act], env.cid[e, l][act]):
            if s < EXTENT:
                out.append((l, float(s), int(k), int(i)))
    return out


def lane_counts(vehicles):
    c = np.zeros(C.N_LANES, np.int64)
    for l, s, _, _ in vehicles:
        if 0 <= s <= C.VIEW_RANGE:
            c[l] += 1
    return c


def draw_world(scene, vehicles, night):
    """The top-down drawing (and a light layer at night)."""
    img = scene.base.copy()
    d = ImageDraw.Draw(img)
    glow = Image.new("RGB", (TOP, TOP), (0, 0, 0)) if night else None
    g = ImageDraw.Draw(glow) if night else None
    sdx, sdy = np.cos(scene.sun) * 1.0, np.sin(scene.sun) * 1.0
    for lane, s, k, cid in vehicles:
        name = KIND_NAMES[k]
        L, W = LENGTHS[k], WIDTH[name]
        (x0, y0, x1, y1), (dx, dy) = car_rect(lane, s, L, W)
        pal = BUS_COLORS if name == "bus" else TRUCK_COLORS if name == "truck" else CAR_COLORS
        color = pal[(cid * 7919) % len(pal)]
        if not night:
            d.rectangle(box(px(x0 + sdx, y0 + sdy), px(x1 + sdx, y1 + sdy)), fill=(32, 34, 36))
        d.rectangle(box(px(x0, y0), px(x1, y1)), fill=color, outline=(18, 18, 18))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        fx, fy = cx + dx * (L / 2 - 1.2), cy + dy * (L / 2 - 1.2)        # windshield
        hw = W / 2 - 0.25
        if dx == 0:
            d.rectangle(box(px(fx - hw, fy - 0.4), px(fx + hw, fy + 0.4)), fill=(28, 38, 52))
        else:
            d.rectangle(box(px(fx - 0.4, fy - hw), px(fx + 0.4, fy + hw)), fill=(28, 38, 52))
        if night:
            hx, hy = cx + dx * L / 2, cy + dy * L / 2
            rx, ry = cx - dx * L / 2, cy - dy * L / 2
            for side in (-0.6, 0.6):
                ox, oy = (side, 0) if dx == 0 else (0, side)
                a, b = px(hx + ox + dx * 2.0, hy + oy + dy * 2.0)            # headlight beam
                g.ellipse([a - 3.5, b - 3.5, a + 3.5, b + 3.5], fill=(255, 240, 200))
                a, b = px(rx + ox, ry + oy)                                   # tail light
                g.ellipse([a - 1.8, b - 1.8, a + 1.8, b + 1.8], fill=(255, 30, 20))
    if night:
        for lx, ly in scene.lamps:
            a, b = px(lx, ly)
            g.ellipse([a - 7, b - 7, a + 7, b + 7], fill=(70, 58, 34))
    return img, glow


def postprocess(view, glow_view, condition, rng):
    arr = np.asarray(view).astype(np.float32)
    if condition == "day":
        arr = arr * rng.uniform(0.85, 1.15) + rng.uniform(-10, 10)
    elif condition == "dusk":
        arr = arr * np.array([0.78, 0.58, 0.48]) * rng.uniform(0.65, 0.95)
    elif condition == "night":
        arr = arr * np.array([0.13, 0.15, 0.22]) * rng.uniform(0.8, 1.3)
    elif condition == "rain":
        gray = arr.mean(axis=2, keepdims=True)
        arr = (0.5 * arr + 0.5 * gray) * rng.uniform(0.45, 0.65) + 10
    if glow_view is not None:
        gl = np.asarray(glow_view.filter(ImageFilter.GaussianBlur(1.6))).astype(np.float32)
        arr = arr + gl * {"dusk": 0.4, "night": 1.0, "rain": 0.65}[condition]
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if condition == "rain":
        out = out.filter(ImageFilter.GaussianBlur(0.8))
        dr = ImageDraw.Draw(out, "RGBA")
        for _ in range(int(rng.integers(120, 260))):
            x, y = rng.uniform(0, C.VIEW_W), rng.uniform(0, C.VIEW_H)
            dr.line([x, y, x - 2, y + rng.uniform(7, 14)], fill=(200, 210, 220, 70), width=1)
    noisy = np.asarray(out).astype(np.float32) + rng.normal(0, rng.uniform(2, 8), (C.VIEW_H, C.VIEW_W, 1))
    out = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.85, 1.15))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=int(rng.integers(35, 85)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def to_net(view):
    """The color image -> grayscale at the size the network receives."""
    return np.array(view.convert("L").resize((C.CAM_W, C.CAM_H), Image.BOX), dtype=np.uint8)


def render_cameras(scene, vehicles, condition="day", rng=None, jitter=0.0, lanes=range(C.N_LANES)):
    """
    Draws the camera images.
    Returns a list: for each lane (a PIL color image of size VIEW, a uint8 grayscale array for the network).
    """
    rng = rng or np.random.default_rng()
    night = condition != "day"
    img, glow = draw_world(scene, vehicles, night)
    out = []
    for lane in lanes:
        co = view_coeffs(lane, rng, jitter)
        warp = lambda im, fill: im.transform((C.VIEW_W, C.VIEW_H), Image.PERSPECTIVE, co,
                                             Image.BILINEAR, fillcolor=fill)
        view = postprocess(warp(img, GROUND), warp(glow, (0, 0, 0)) if night else None, condition, rng)
        out.append((view, to_net(view)))
    return out
