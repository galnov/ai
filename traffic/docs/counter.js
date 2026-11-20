// =====================================================================================
//  Network 1 in JavaScript: counting the cars in a camera image (src/counter.py).
//  Runs in a background thread (counter-worker.js), so the page stays smooth.
//
//  The layers come from docs/model/model.json ("counter"), the numbers from docs/model/weights.bin.
//  BatchNorm was folded into the convolutions by  python -m src.export_web  (in evaluation mode it is
//  just a fixed scale and shift per channel), so every block is: 3x3 convolution (+ bias) -> ReLU.
//    112x64 -> [conv, conv, maxpool] -> 56x32 -> [conv, conv, maxpool] -> 28x16 -> [conv, conv,
//    dilated conv (2), dilated conv (4)] -> 1x1 convolution -> softplus = the density map (28 x 16)
//  The count is the sum of the density map.
// =====================================================================================
(function (root) {
  // a 3x3 (or 1x1) convolution with "same" padding and dilation d, followed by an activation
  function conv(x, H, W, L, buf) {
    const { in: cin, out: cout, k, dil } = L;
    const w = buf.subarray(L.w, L.w + cout * cin * k * k), b = buf.subarray(L.b, L.b + cout);
    const pad = k === 3 ? dil : 0, PW = W + 2 * pad, PH = H + 2 * pad;
    // the input with a border of zeros, so the inner loop needs no bounds checks
    const xp = new Float32Array(cin * PH * PW);
    for (let c = 0; c < cin; c++) for (let y = 0; y < H; y++) xp.set(x.subarray((c * H + y) * W, (c * H + y + 1) * W), (c * PH + y + pad) * PW + pad);
    const out = new Float32Array(cout * H * W), N = H * W, KK = k * k;
    // four output channels at a time: every input value that is read is used four times
    for (let o0 = 0; o0 < cout; o0 += 4) {
      const n = Math.min(4, cout - o0);
      const p0 = out.subarray(o0 * N, (o0 + 1) * N), p1 = n > 1 ? out.subarray((o0 + 1) * N, (o0 + 2) * N) : new Float32Array(N);
      const p2 = n > 2 ? out.subarray((o0 + 2) * N, (o0 + 3) * N) : new Float32Array(N), p3 = n > 3 ? out.subarray((o0 + 3) * N, (o0 + 4) * N) : new Float32Array(N);
      p0.fill(b[o0]); if (n > 1) p1.fill(b[o0 + 1]); if (n > 2) p2.fill(b[o0 + 2]); if (n > 3) p3.fill(b[o0 + 3]);
      for (let c = 0; c < cin; c++) {
        for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
          const t = c * KK + ky * k + kx;
          const w0 = w[o0 * cin * KK + t], w1 = n > 1 ? w[(o0 + 1) * cin * KK + t] : 0;
          const w2 = n > 2 ? w[(o0 + 2) * cin * KK + t] : 0, w3 = n > 3 ? w[(o0 + 3) * cin * KK + t] : 0;
          const base = c * PH * PW + ky * dil * PW + kx * dil;
          for (let y = 0; y < H; y++) {
            const src = base + y * PW, dst = y * W;
            for (let xx = 0; xx < W; xx++) {
              const v = xp[src + xx], j = dst + xx;
              p0[j] += w0 * v; p1[j] += w1 * v; p2[j] += w2 * v; p3[j] += w3 * v;
            }
          }
        }
      }
    }
    for (let o = 0; o < cout; o++) {
      const plane = out.subarray(o * N, (o + 1) * N);
      if (L.act === "relu") for (let i = 0; i < plane.length; i++) plane[i] = plane[i] > 0 ? plane[i] : 0;
      else if (L.act === "softplus") for (let i = 0; i < plane.length; i++) { const v = plane[i]; plane[i] = v > 20 ? v : Math.log1p(Math.exp(v)); }
    }
    return out;
  }

  function maxpool(x, C, H, W) {
    const h = Math.floor(H / 2), w = Math.floor(W / 2), out = new Float32Array(C * h * w);
    for (let c = 0; c < C; c++) for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) {
      const i = (c * H + 2 * y) * W + 2 * xx;
      out[(c * h + y) * w + xx] = Math.max(x[i], x[i + 1], x[i + W], x[i + W + 1]);
    }
    return out;
  }

  // pixels: uint8 grayscale, 112 x 64. Returns { count, density (Float32Array 28 x 16) }
  function count(pixels, layers, buf, H = 112, W = 64) {
    let x = new Float32Array(pixels.length), C = 1;
    for (let i = 0; i < x.length; i++) x[i] = pixels[i] / 255;
    for (const L of layers) {
      if (L.type === "maxpool") { x = maxpool(x, C, H, W); H = Math.floor(H / 2); W = Math.floor(W / 2); }
      else { x = conv(x, H, W, L, buf); C = L.out; }
    }
    let s = 0;
    for (let i = 0; i < x.length; i++) s += x[i];
    return { count: s, density: x, h: H, w: W };
  }

  root.Counter = { count, conv, maxpool };
})(typeof self !== "undefined" ? self : globalThis);
