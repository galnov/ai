// =====================================================================================
//  Face detection in JavaScript: OpenCV's LBP cascade (docs/face_cascade.json), the same algorithm as
//  cv2.CascadeClassifier.detectMultiScale (Viola-Jones with "Local Binary Pattern" features).
//
//  The idea: slide a 24 x 24 window over the image, at many sizes. For each window, a "cascade" of 20 stages
//  decides: face or not. Every stage looks at a few features and can reject the window right away, so most
//  windows are rejected after a handful of checks and the detector is fast.
//  An LBP feature compares a 3 x 3 grid of blocks: the 8 outer blocks against the middle one (brighter or
//  not) gives an 8-bit code (0-255). Block sums are computed in O(1) with an "integral image".
//  At the end, windows that overlap are grouped, and a face needs at least minNeighbors windows.
//
//  (The Flask app uses OpenCV's Haar cascade. Newer OpenCV versions don't ship it the same way, so the web
//   version uses the LBP cascade, which comes with this page: same idea, different features.)
// =====================================================================================
(function (root) {
  // ---------------------------------------------------------------- images
  // like cv2.equalizeHist: spread the brightness values over the whole range
  function equalizeHist(g) {
    const hist = new Int32Array(256);
    for (let i = 0; i < g.length; i++) hist[g[i]]++;
    let i0 = 0;
    while (i0 < 255 && hist[i0] === 0) i0++;
    const total = g.length;
    if (hist[i0] === total) return Uint8Array.from(g, () => i0);
    const scale = 255 / (total - hist[i0]), lut = new Uint8Array(256);
    let sum = 0;
    for (let i = i0 + 1; i < 256; i++) { sum += hist[i]; lut[i] = Math.min(255, Math.max(0, Math.round(sum * scale))); }
    return Uint8Array.from(g, (v) => lut[v]);
  }

  // like cv2.resize(INTER_LINEAR)
  function resizeLinear(g, W, H, w, h) {
    const out = new Uint8Array(w * h), sx = W / w, sy = H / h;
    for (let y = 0; y < h; y++) {
      let fy = (y + 0.5) * sy - 0.5, y0 = Math.floor(fy);
      let ty = fy - y0;
      if (y0 < 0) { y0 = 0; ty = 0; }
      const y1 = Math.min(H - 1, y0 + 1);
      if (y0 >= H - 1) { y0 = H - 1; ty = 0; }
      for (let x = 0; x < w; x++) {
        let fx = (x + 0.5) * sx - 0.5, x0 = Math.floor(fx);
        let tx = fx - x0;
        if (x0 < 0) { x0 = 0; tx = 0; }
        const x1 = Math.min(W - 1, x0 + 1);
        if (x0 >= W - 1) { x0 = W - 1; tx = 0; }
        const v = (g[y0 * W + x0] * (1 - tx) + g[y0 * W + x1] * tx) * (1 - ty) + (g[y1 * W + x0] * (1 - tx) + g[y1 * W + x1] * tx) * ty;
        out[y * w + x] = Math.round(v);
      }
    }
    return out;
  }

  // like cv2.resize(INTER_AREA) when shrinking: every output pixel is the average of the area it covers
  function resizeArea(g, W, H, w, h) {
    const out = new Uint8Array(w * h), sx = W / w, sy = H / h;
    for (let y = 0; y < h; y++) {
      const ya = y * sy, yb = (y + 1) * sy;
      for (let x = 0; x < w; x++) {
        const xa = x * sx, xb = (x + 1) * sx;
        let s = 0, area = 0;
        for (let yy = Math.floor(ya); yy < Math.ceil(yb); yy++) {
          const wy = Math.min(yb, yy + 1) - Math.max(ya, yy);
          for (let xx = Math.floor(xa); xx < Math.ceil(xb); xx++) {
            const wx = Math.min(xb, xx + 1) - Math.max(xa, xx), wgt = wx * wy;
            s += g[Math.min(H - 1, yy) * W + Math.min(W - 1, xx)] * wgt; area += wgt;
          }
        }
        out[y * w + x] = Math.round(s / area);
      }
    }
    return out;
  }

  function integral(g, W, H) {
    const I = new Int32Array((W + 1) * (H + 1));
    for (let y = 0; y < H; y++) {
      let row = 0;
      for (let x = 0; x < W; x++) { row += g[y * W + x]; I[(y + 1) * (W + 1) + x + 1] = I[y * (W + 1) + x + 1] + row; }
    }
    return I;
  }

  // ---------------------------------------------------------------- the cascade
  class Cascade {
    constructor(json) {
      Object.assign(this, json);
    }

    // offsets of the 4 x 4 grid corners of every feature, for an integral image of row length S
    prepare(S) {
      if (this.S === S) return;
      this.S = S;
      this.ofs = this.features.map(([x, y, w, h]) => {
        const o = new Int32Array(16);
        for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) o[j * 4 + i] = (y + j * h) * S + (x + i * w);
        return o;
      });
    }

    // is the window with its top-left corner at integral-image index p a face?
    classify(I, p) {
      for (const st of this.stages) {
        let sum = 0;
        for (const [f, subset, left, right] of st.w) {
          const o = this.ofs[f];
          // the 16 grid corners, read once; block (i, j) = v[j*4+i] - v[j*4+i+1] - v[j*4+i+4] + v[j*4+i+5]
          const v0 = I[p + o[0]], v1 = I[p + o[1]], v2 = I[p + o[2]], v3 = I[p + o[3]];
          const v4 = I[p + o[4]], v5 = I[p + o[5]], v6 = I[p + o[6]], v7 = I[p + o[7]];
          const v8 = I[p + o[8]], v9 = I[p + o[9]], v10 = I[p + o[10]], v11 = I[p + o[11]];
          const v12 = I[p + o[12]], v13 = I[p + o[13]], v14 = I[p + o[14]], v15 = I[p + o[15]];
          const cval = v5 - v6 - v9 + v10;                                    // the middle block
          const code = (v0 - v1 - v4 + v5 >= cval ? 128 : 0) | (v1 - v2 - v5 + v6 >= cval ? 64 : 0) |
                       (v2 - v3 - v6 + v7 >= cval ? 32 : 0) | (v6 - v7 - v10 + v11 >= cval ? 16 : 0) |
                       (v10 - v11 - v14 + v15 >= cval ? 8 : 0) | (v9 - v10 - v13 + v14 >= cval ? 4 : 0) |
                       (v8 - v9 - v12 + v13 >= cval ? 2 : 0) | (v4 - v5 - v8 + v9 >= cval ? 1 : 0);
          sum += subset[code >> 5] & (1 << (code & 31)) ? left : right;
        }
        if (sum < st.t - 1e-5) return false;
      }
      return true;
    }

    // all windows that the cascade accepts, in the original image's coordinates (before grouping)
    windows(gray, W, H, scaleFactor = 1.1, minSize = 0) {
      const out = [], ww = this.width, wh = this.height;
      for (let factor = 1; ; factor *= scaleFactor) {
        const winW = Math.round(ww * factor), winH = Math.round(wh * factor);
        const sw = Math.round(W / factor), sh = Math.round(H / factor);
        if (sw - ww <= 0 || sh - wh <= 0) break;
        if (winW < minSize || winH < minSize) continue;
        const img = factor === 1 ? gray : resizeLinear(gray, W, H, sw, sh);
        const I = integral(img, sw, sh), S = sw + 1;
        this.prepare(S);
        const step = factor > 2 ? 1 : 2;
        for (let y = 0; y < sh - wh; y += step) for (let x = 0; x < sw - ww; x += step) {
          if (this.classify(I, y * S + x)) out.push([Math.round(x * factor), Math.round(y * factor), winW, winH]);
        }
      }
      return out;
    }
  }

  // like cv2.groupRectangles(rects, minNeighbors, 0.2)
  function groupRectangles(rects, minNeighbors, eps = 0.2) {
    if (minNeighbors <= 0 || !rects.length) return rects;
    const parent = rects.map((_, i) => i);
    const find = (i) => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    const similar = (a, b) => {
      const d = eps * (Math.min(a[2], b[2]) + Math.min(a[3], b[3])) * 0.5;
      return Math.abs(a[0] - b[0]) <= d && Math.abs(a[1] - b[1]) <= d &&
             Math.abs(a[0] + a[2] - b[0] - b[2]) <= d && Math.abs(a[1] + a[3] - b[1] - b[3]) <= d;
    };
    for (let i = 0; i < rects.length; i++) for (let j = i + 1; j < rects.length; j++) if (similar(rects[i], rects[j])) parent[find(i)] = find(j);
    const groups = new Map();
    rects.forEach((r, i) => {
      const k = find(i);
      if (!groups.has(k)) groups.set(k, [0, 0, 0, 0, 0]);
      const g = groups.get(k);
      for (let q = 0; q < 4; q++) g[q] += r[q];
      g[4] += 1;
    });
    const cls = [...groups.values()].map((g) => ({ r: g.slice(0, 4).map((v) => Math.round(v / g[4])), n: g[4] }));
    const out = [];
    for (const a of cls) {
      if (a.n <= minNeighbors) continue;
      const inside = cls.some((b) => {               // a small face inside a bigger one is dropped
        if (b === a || b.n <= minNeighbors) return false;
        const dx = Math.round(b.r[2] * eps), dy = Math.round(b.r[3] * eps);
        return a.r[0] >= b.r[0] - dx && a.r[1] >= b.r[1] - dy && a.r[0] + a.r[2] <= b.r[0] + b.r[2] + dx &&
               a.r[1] + a.r[3] <= b.r[1] + b.r[3] + dy && (b.n > Math.max(3, a.n) || a.n < 3);
      });
      if (!inside) out.push(a.r);
    }
    return out;
  }

  // like detect_faces in src/vision.py: (x, y, w, h) rectangles, largest first
  function detectFaces(cascade, gray, W, H, maxFaces = 4, { scaleFactor = 1.1, minNeighbors = 4 } = {}) {
    const minSide = Math.max(40, Math.floor(Math.min(W, H) * 0.08));
    const rects = groupRectangles(cascade.windows(equalizeHist(gray), W, H, scaleFactor, minSide), minNeighbors);
    return rects.sort((a, b) => b[2] * b[3] - a[2] * a[3]).slice(0, maxFaces);
  }

  root.Faces = { Cascade, detectFaces, groupRectangles, equalizeHist, resizeLinear, resizeArea, integral };
})(typeof self !== "undefined" ? self : globalThis);
