// =====================================================================================
//  The emotion network in JavaScript (src/model.py EmotionCNN), with Grad-CAM (src/gradcam.py).
//  Runs in a background thread (emotion-worker.js), so the page stays smooth.
//
//  The layers come from docs/model/model.json, the numbers from docs/model/weights.bin, both written by
//  python -m src.export_web. BatchNorm was folded into the layer before it (in evaluation mode it is just a
//  fixed scale and shift per channel), so the network is:
//    48x48 -> 4 blocks of [conv 3x3 -> ReLU -> conv 3x3 -> ReLU -> max-pool]  -> 512 x 3 x 3
//          -> global average -> dense (256) -> ReLU -> dense (7) = the logits, one per emotion
//  (Dropout does nothing when the network is used, so it is not here.)
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

  function dense(x, L, buf, relu) {
    const W = buf.subarray(L.w, L.w + L.in * L.out), b = buf.subarray(L.b, L.b + L.out), y = new Float32Array(L.out);
    for (let o = 0; o < L.out; o++) {
      let s = b[o];
      for (let i = 0; i < L.in; i++) s += W[o * L.in + i] * x[i];
      y[o] = relu && s < 0 ? 0 : s;
    }
    return y;
  }

  // x: Float32Array 48 x 48 (already normalized). Returns the logits, and what Grad-CAM needs.
  function forward(x, M, buf) {
    let H = 48, W = 48, C = 1, a = x, camAct = null;
    for (const L of M.layers) {
      if (L.type === "maxpool") {
        if (L.cam) camAct = { a, C, H, W };                 // the last convolution's output, before pooling (6 x 6)
        a = maxpool(a, C, H, W); H = Math.floor(H / 2); W = Math.floor(W / 2);
      } else { a = conv(a, H, W, L, buf); C = L.out; }
    }
    const g = new Float32Array(C);                           // global average pooling
    for (let c = 0; c < C; c++) { let s = 0; for (let i = 0; i < H * W; i++) s += a[c * H * W + i]; g[c] = s / (H * W); }
    const h = dense(g, M.dense1, buf, true);
    const logits = dense(h, M.dense2, buf, false);
    return { logits, h, camAct, pooledHW: [H, W] };
  }

  // Grad-CAM for class k: how much each channel of the last convolution pushes the score of k up
  // (the gradient, averaged over the 6 x 6 map), times the channel's map, summed, and only the positive part.
  function gradcam(f, k, M, buf) {
    const D1 = M.dense1, D2 = M.dense2;
    const W2 = buf.subarray(D2.w, D2.w + D2.in * D2.out), W1 = buf.subarray(D1.w, D1.w + D1.in * D1.out);
    const gh = new Float32Array(D1.out);                     // d logit_k / d h (through the ReLU)
    for (let j = 0; j < D1.out; j++) gh[j] = f.h[j] > 0 ? W2[k * D2.in + j] : 0;
    const gg = new Float32Array(D1.in);                      // d logit_k / d (the averaged features)
    for (let j = 0; j < D1.out; j++) if (gh[j]) for (let i = 0; i < D1.in; i++) gg[i] += gh[j] * W1[j * D1.in + i];
    // the average over the 3 x 3 pooled cells spreads gg/9 to each cell; max-pooling sends each cell's gradient
    // to the largest of its 2 x 2 values, so the total gradient on a channel's 6 x 6 map is gg, and its mean is gg/36
    const { a, C, H, W } = f.camAct, N = H * W, cam = new Float32Array(N);
    for (let c = 0; c < C; c++) {
      const wgt = gg[c] / N;
      if (wgt === 0) continue;
      for (let i = 0; i < N; i++) cam[i] += wgt * a[c * N + i];
    }
    for (let i = 0; i < N; i++) if (cam[i] < 0) cam[i] = 0;
    // up to 48 x 48 (bilinear, like F.interpolate with align_corners=False), then scaled to 0..1
    const S = 48, out = new Float32Array(S * S), sy = H / S, sx = W / S;
    for (let y = 0; y < S; y++) {
      const fy = Math.max(0, (y + 0.5) * sy - 0.5), y0 = Math.min(H - 1, Math.floor(fy)), y1 = Math.min(H - 1, y0 + 1), ty = fy - y0;
      for (let x = 0; x < S; x++) {
        const fx = Math.max(0, (x + 0.5) * sx - 0.5), x0 = Math.min(W - 1, Math.floor(fx)), x1 = Math.min(W - 1, x0 + 1), tx = fx - x0;
        out[y * S + x] = (cam[y0 * W + x0] * (1 - tx) + cam[y0 * W + x1] * tx) * (1 - ty) + (cam[y1 * W + x0] * (1 - tx) + cam[y1 * W + x1] * tx) * ty;
      }
    }
    let lo = Infinity, hi = -Infinity;
    for (const v of out) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (let i = 0; i < out.length; i++) out[i] = (out[i] - lo) / (hi - lo + 1e-8);
    return out;
  }

  const softmax = (z, T = 1) => { const m = Math.max(...z), e = Array.from(z, (v) => Math.exp((v - m) / T)), s = e.reduce((p, q) => p + q, 0); return e.map((v) => v / s); };

  // face48: uint8 48 x 48. Like EmotionPredictor.predict_face: with Grad-CAM, one pass and the CAM of the top
  // emotion; without, the image and its mirror image ("test-time augmentation"), averaged.
  function predictFace(face48, M, buf, withCam, tta = true) {
    const x = new Float32Array(48 * 48);
    for (let i = 0; i < x.length; i++) x[i] = (face48[i] / 255 - M.mean) / M.std;
    const f = forward(x, M, buf);
    if (withCam) {
      const probs = softmax(f.logits);
      let k = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[k]) k = i;
      return { probs, cam: gradcam(f, k, M, buf) };
    }
    if (!tta) return { probs: softmax(f.logits), cam: null };
    const xm = new Float32Array(48 * 48);
    for (let y = 0; y < 48; y++) for (let xx = 0; xx < 48; xx++) xm[y * 48 + xx] = x[y * 48 + 47 - xx];
    const g = forward(xm, M, buf);
    return { probs: softmax(f.logits.map((v, i) => v + g.logits[i]), 2), cam: null };
  }

  root.EmotionNet = { forward, gradcam, predictFace, conv, maxpool };
})(typeof self !== "undefined" ? self : globalThis);
