// =====================================================================================
//  The intersection cameras in JavaScript - a copy of src/camera.py.
//    1. Draw the intersection from above: the fixed background (docs/scene.png, drawn by Python's
//       Scene(104) and saved by  python -m src.export_web) plus the cars, pixel by pixel like PIL draws them
//    2. For each camera, a perspective warp (a homography) of the area it sees
//    3. Shooting conditions (day, dusk, night, rain), camera noise, contrast and JPEG compression
//    4. The image for the network: grayscale, 64 x 112 (each pixel = the average of 3 x 3)
// =====================================================================================
(function (root) {
  const STOP = 9.0, LANE_OFF = 1.75, VIEW_RANGE = 120.0;
  const EXTENT = STOP + VIEW_RANGE + 12.0, PPM = 3.0, TOP = Math.trunc(2 * EXTENT * PPM);
  const VIEW_W = 192, VIEW_H = 336, CAM_W = 64, CAM_H = 112;
  const WIDTH = { car: 1.8, truck: 2.5, bus: 2.5 };
  const KIND_NAMES = ["car", "truck", "bus"], LENGTHS = [4.5, 9.0, 12.0];
  const CAR_COLORS = [[235, 235, 235], [200, 202, 205], [40, 40, 45], [25, 30, 60], [150, 20, 25],
                      [30, 60, 140], [110, 112, 118], [60, 90, 70], [210, 180, 120], [230, 120, 30]];
  const BUS_COLORS = [[240, 200, 40], [235, 235, 235], [40, 120, 200]];
  const TRUCK_COLORS = [[235, 235, 235], [180, 40, 30], [60, 110, 160]];
  const GROUND = [118, 128, 104];
  const FAR_W = STOP + VIEW_RANGE + 3;
  const VIEW_LOCAL = [[-15.0, FAR_W], [15.0, FAR_W], [6.5, STOP - 3.5], [-6.5, STOP - 3.5]];

  const px = (x, y) => [(x + EXTENT) * PPM, (y + EXTENT) * PPM];
  const uniform = (a, b) => a + Math.random() * (b - a);
  function gauss() { let u = 0; while (u === 0) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); }

  function localToWorld(lane, u, w) {
    if (lane === 0) return [u, -w];         // from the north: looks north
    if (lane === 1) return [-u, w];         // from the south: looks south
    if (lane === 2) return [w, u];          // from the east: looks east
    return [-w, -u];                        // from the west: looks west
  }

  function carRect(lane, s, length, width) {
    const a = STOP + s, b = STOP + s + length, w = width / 2;
    if (lane === 0) return [[-LANE_OFF - w, -b, -LANE_OFF + w, -a], [0, 1]];
    if (lane === 1) return [[LANE_OFF - w, a, LANE_OFF + w, b], [0, -1]];
    if (lane === 2) return [[a, -LANE_OFF - w, b, -LANE_OFF + w], [-1, 0]];
    return [[-b, LANE_OFF - w, -a, LANE_OFF + w], [1, 0]];
  }

  // ---------------------------------------------------------------- drawing like PIL (whole pixels, no smoothing)
  class Raster {
    constructor(w, h, data) { this.w = w; this.h = h; this.d = data || new Float32Array(w * h * 3); }
    set(x, y, c) { if (x < 0 || y < 0 || x >= this.w || y >= this.h) return; const i = (y * this.w + x) * 3; this.d[i] = c[0]; this.d[i + 1] = c[1]; this.d[i + 2] = c[2]; }
    // PIL: a rectangle covers the whole pixels from trunc(x0) to trunc(x1), both included
    rect(p, q, fill, outline) {
      const x0 = Math.trunc(Math.min(p[0], q[0])), x1 = Math.trunc(Math.max(p[0], q[0]));
      const y0 = Math.trunc(Math.min(p[1], q[1])), y1 = Math.trunc(Math.max(p[1], q[1]));
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const edge = outline && (x === x0 || x === x1 || y === y0 || y === y1);
        this.set(x, y, edge ? outline : fill);
      }
    }
    ellipse(x0f, y0f, x1f, y1f, fill) {
      const x0 = Math.trunc(x0f), x1 = Math.trunc(x1f), y0 = Math.trunc(y0f), y1 = Math.trunc(y1f);
      const cx = (x0 + x1 + 1) / 2, cy = (y0 + y1 + 1) / 2, rx = (x1 - x0 + 1) / 2, ry = (y1 - y0 + 1) / 2;
      for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
        const dx = (x + 0.5 - cx) / rx, dy = (y + 0.5 - cy) / ry;
        if (dx * dx + dy * dy <= 1) this.set(x, y, fill);
      }
    }
  }

  // the top-down drawing (and a light layer at night) - draw_world in camera.py
  function drawWorld(scene, vehicles, night) {
    const img = new Raster(TOP, TOP, Float32Array.from(scene.base));
    const glow = night ? new Raster(TOP, TOP) : null;
    const sdx = Math.cos(scene.sun), sdy = Math.sin(scene.sun);
    for (const [lane, s, k, cid] of vehicles) {
      const name = KIND_NAMES[k], L = LENGTHS[k], W = WIDTH[name];
      if (s >= EXTENT) continue;
      const [[x0, y0, x1, y1], [dx, dy]] = carRect(lane, s, L, W);
      const pal = name === "bus" ? BUS_COLORS : name === "truck" ? TRUCK_COLORS : CAR_COLORS;
      const color = pal[(cid * 7919) % pal.length];
      if (!night) img.rect(px(x0 + sdx, y0 + sdy), px(x1 + sdx, y1 + sdy), [32, 34, 36]);      // the shadow
      img.rect(px(x0, y0), px(x1, y1), color, [18, 18, 18]);
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const fx = cx + dx * (L / 2 - 1.2), fy = cy + dy * (L / 2 - 1.2), hw = W / 2 - 0.25;      // the windshield
      if (dx === 0) img.rect(px(fx - hw, fy - 0.4), px(fx + hw, fy + 0.4), [28, 38, 52]);
      else img.rect(px(fx - 0.4, fy - hw), px(fx + 0.4, fy + hw), [28, 38, 52]);
      if (night) {
        const hx = cx + (dx * L) / 2, hy = cy + (dy * L) / 2, rx = cx - (dx * L) / 2, ry = cy - (dy * L) / 2;
        for (const side of [-0.6, 0.6]) {
          const [ox, oy] = dx === 0 ? [side, 0] : [0, side];
          let [a, b] = px(hx + ox + dx * 2.0, hy + oy + dy * 2.0);                                // headlight beam
          glow.ellipse(a - 3.5, b - 3.5, a + 3.5, b + 3.5, [255, 240, 200]);
          [a, b] = px(rx + ox, ry + oy);                                                          // tail light
          glow.ellipse(a - 1.8, b - 1.8, a + 1.8, b + 1.8, [255, 30, 20]);
        }
      }
    }
    if (night) for (const [lx, ly] of scene.lamps) { const [a, b] = px(lx, ly); glow.ellipse(a - 7, b - 7, a + 7, b + 7, [70, 58, 34]); }
    return [img, glow];
  }

  // ---------------------------------------------------------------- the perspective warp
  function solve(A, b) {                     // Gaussian elimination with pivoting
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    return M.map((r, i) => r[n] / r[i]);
  }

  // PIL PERSPECTIVE coefficients: a point in the output image -> a point in the top-down drawing
  function viewCoeffs(lane) {
    const src = VIEW_LOCAL.map(([u, w]) => px(...localToWorld(lane, u, w)));
    const dst = [[0, 0], [VIEW_W, 0], [VIEW_W, VIEW_H], [0, VIEW_H]];
    const A = [], b = [];
    src.forEach(([x, y], i) => {
      const [u, v] = dst[i];
      A.push([u, v, 1, 0, 0, 0, -x * u, -x * v]); b.push(x);
      A.push([0, 0, 0, u, v, 1, -y * u, -y * v]); b.push(y);
    });
    return solve(A, b);
  }

  // like PIL's transform(PERSPECTIVE, BILINEAR): sample at the center of each output pixel
  function warp(img, co, fill) {
    const out = new Raster(VIEW_W, VIEW_H), [a, b, c, d, e, f, g, h] = co, W = img.w, H = img.h, D = img.d;
    for (let y = 0; y < VIEW_H; y++) for (let x = 0; x < VIEW_W; x++) {
      const u = x + 0.5, v = y + 0.5, den = g * u + h * v + 1;
      const sx = (a * u + b * v + c) / den - 0.5, sy = (d * u + e * v + f) / den - 0.5;
      const o = (y * VIEW_W + x) * 3;
      if (sx < -0.5 || sy < -0.5 || sx > W - 0.5 || sy > H - 0.5) { out.d[o] = fill[0]; out.d[o + 1] = fill[1]; out.d[o + 2] = fill[2]; continue; }
      const x0 = Math.floor(sx), y0 = Math.floor(sy), tx = sx - x0, ty = sy - y0;
      const xa = Math.max(0, x0), xb = Math.min(W - 1, x0 + 1), ya = Math.max(0, y0), yb = Math.min(H - 1, y0 + 1);
      for (let k = 0; k < 3; k++) {
        const p00 = D[(ya * W + xa) * 3 + k], p10 = D[(ya * W + xb) * 3 + k], p01 = D[(yb * W + xa) * 3 + k], p11 = D[(yb * W + xb) * 3 + k];
        out.d[o + k] = Math.round((p00 * (1 - tx) + p10 * tx) * (1 - ty) + (p01 * (1 - tx) + p11 * tx) * ty);
      }
    }
    return out;
  }

  function blur(r, sigma) {                  // a Gaussian blur, in two passes (rows, then columns)
    const rad = Math.ceil(sigma * 3), k = [];
    let sum = 0;
    for (let i = -rad; i <= rad; i++) { const v = Math.exp(-(i * i) / (2 * sigma * sigma)); k.push(v); sum += v; }
    for (let i = 0; i < k.length; i++) k[i] /= sum;
    const { w, h } = r, tmp = new Float32Array(r.d.length), out = new Float32Array(r.d.length);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let i = -rad; i <= rad; i++) s += k[i + rad] * r.d[(y * w + Math.min(w - 1, Math.max(0, x + i))) * 3 + c];
      tmp[(y * w + x) * 3 + c] = s;
    }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let i = -rad; i <= rad; i++) s += k[i + rad] * tmp[(Math.min(h - 1, Math.max(0, y + i)) * w + x) * 3 + c];
      out[(y * w + x) * 3 + c] = s;
    }
    return new Raster(w, h, out);
  }

  const clip255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

  // ---------------------------------------------------------------- shooting conditions (postprocess in camera.py)
  async function postprocess(view, glowView, condition) {
    let d = view.d.slice();
    if (condition === "day") { const m = uniform(0.85, 1.15), b = uniform(-10, 10); for (let i = 0; i < d.length; i++) d[i] = d[i] * m + b; }
    else if (condition === "dusk") { const m = uniform(0.65, 0.95), t = [0.78, 0.58, 0.48]; for (let i = 0; i < d.length; i++) d[i] *= t[i % 3] * m; }
    else if (condition === "night") { const m = uniform(0.8, 1.3), t = [0.13, 0.15, 0.22]; for (let i = 0; i < d.length; i++) d[i] *= t[i % 3] * m; }
    else if (condition === "rain") {
      const m = uniform(0.45, 0.65);
      for (let i = 0; i < d.length; i += 3) {
        const g = (d[i] + d[i + 1] + d[i + 2]) / 3;
        for (let k = 0; k < 3; k++) d[i + k] = (0.5 * d[i + k] + 0.5 * g) * m + 10;
      }
    }
    if (glowView) {
      const gl = blur(glowView, 1.6).d, f = { dusk: 0.4, night: 1.0, rain: 0.65 }[condition];
      for (let i = 0; i < d.length; i++) d[i] += gl[i] * f;
    }
    for (let i = 0; i < d.length; i++) d[i] = Math.trunc(clip255(d[i]));         // like numpy's astype(uint8)
    let r = new Raster(VIEW_W, VIEW_H, d);
    if (condition === "rain") {
      r = blur(r, 0.8);
      for (let i = 0; i < r.d.length; i++) r.d[i] = Math.round(r.d[i]);
      const n = 120 + Math.floor(Math.random() * 140), al = 70 / 255;
      for (let k = 0; k < n; k++) {                                   // rain streaks
        const x = uniform(0, VIEW_W), y = uniform(0, VIEW_H), len = uniform(7, 14), steps = Math.ceil(len);
        for (let j = 0; j <= steps; j++) {
          const xi = Math.round(x - (2 * j) / steps), yi = Math.round(y + (len * j) / steps);
          if (xi < 0 || yi < 0 || xi >= VIEW_W || yi >= VIEW_H) continue;
          const o = (yi * VIEW_W + xi) * 3;
          [200, 210, 220].forEach((c, q) => (r.d[o + q] = Math.round(r.d[o + q] * (1 - al) + c * al)));
        }
      }
    }
    d = r.d;
    const sd = uniform(2, 8);                                          // camera noise (the same on all 3 colors)
    for (let i = 0; i < d.length; i += 3) { const nz = gauss() * sd; for (let k = 0; k < 3; k++) d[i + k] = Math.trunc(clip255(d[i + k] + nz)); }
    let lum = 0;                                                       // contrast (like PIL's ImageEnhance.Contrast)
    for (let i = 0; i < d.length; i += 3) lum += (d[i] * 19595 + d[i + 1] * 38470 + d[i + 2] * 7471 + 0x8000) >> 16;
    const mean = Math.trunc(lum / (VIEW_W * VIEW_H) + 0.5), cf = uniform(0.85, 1.15);
    for (let i = 0; i < d.length; i++) d[i] = clip255(Math.trunc(mean + cf * (d[i] - mean)));
    return jpeg(d, 35 + Math.floor(Math.random() * 50));
  }

  // JPEG compression with the browser's encoder, and decoding it back (like saving and reopening in PIL)
  async function jpeg(d, quality) {
    const cv = document.createElement("canvas");
    cv.width = VIEW_W; cv.height = VIEW_H;
    const ctx = cv.getContext("2d", { willReadFrequently: true }), im = ctx.createImageData(VIEW_W, VIEW_H);
    for (let i = 0, j = 0; i < d.length; i += 3, j += 4) { im.data[j] = d[i]; im.data[j + 1] = d[i + 1]; im.data[j + 2] = d[i + 2]; im.data[j + 3] = 255; }
    ctx.putImageData(im, 0, 0);
    const url = cv.toDataURL("image/jpeg", quality / 100);
    const img = new Image();
    img.src = url;
    await img.decode();
    ctx.drawImage(img, 0, 0);
    const back = ctx.getImageData(0, 0, VIEW_W, VIEW_H).data, rgb = new Float32Array(VIEW_W * VIEW_H * 3);
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) { rgb[i] = back[j]; rgb[i + 1] = back[j + 1]; rgb[i + 2] = back[j + 2]; }
    return { url, rgb };
  }

  // the color image -> grayscale at the size the network receives (PIL: convert("L"), resize BOX)
  function toNet(rgb) {
    const L = new Float32Array(VIEW_W * VIEW_H);
    for (let i = 0; i < L.length; i++) L[i] = (rgb[3 * i] * 19595 + rgb[3 * i + 1] * 38470 + rgb[3 * i + 2] * 7471 + 0x8000) >> 16;
    const out = new Uint8Array(CAM_W * CAM_H);
    for (let y = 0; y < CAM_H; y++) for (let x = 0; x < CAM_W; x++) {
      let s = 0;
      for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) s += L[(3 * y + j) * VIEW_W + 3 * x + i];
      out[y * CAM_W + x] = Math.round(s / 9);
    }
    return out;
  }

  const COEFFS = [0, 1, 2, 3].map(viewCoeffs);

  // all four cameras: [{url (the color JPEG for the screen), net (uint8 64 x 112 for the network)}]
  async function renderCameras(scene, vehicles, condition = "day") {
    const night = condition !== "day";
    const [img, glow] = drawWorld(scene, vehicles, night);
    const out = [];
    for (let lane = 0; lane < 4; lane++) {
      const view = warp(img, COEFFS[lane], GROUND), gv = night ? warp(glow, COEFFS[lane], [0, 0, 0]) : null;
      const { url, rgb } = await postprocess(view, gv, condition);
      out.push({ url, net: toNet(rgb) });
    }
    return out;
  }

  async function loadScene(dir = ".") {
    const meta = await (await fetch(`${dir}/scene.json`, { cache: "no-cache" })).json();
    const img = new Image();
    img.src = `${dir}/scene.png`;
    await img.decode();
    const cv = document.createElement("canvas");
    cv.width = TOP; cv.height = TOP;
    const ctx = cv.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    const px4 = ctx.getImageData(0, 0, TOP, TOP).data, base = new Float32Array(TOP * TOP * 3);
    for (let i = 0, j = 0; i < base.length; i += 3, j += 4) { base[i] = px4[j]; base[i + 1] = px4[j + 1]; base[i + 2] = px4[j + 2]; }
    return { base, sun: meta.sun, lamps: meta.lamps };
  }

  root.Cameras = { renderCameras, loadScene, drawWorld, warp, viewCoeffs, toNet, Raster, TOP, VIEW_W, VIEW_H, CAM_W, CAM_H, GROUND };
})(typeof window !== "undefined" ? window : globalThis);
