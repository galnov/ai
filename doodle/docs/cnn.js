// =====================================================================================
//  The trained CNN, running in the browser - written in plain JavaScript, no libraries.
//  The weights come from  python -m src.export_web  (docs/model/weights.bin + model.json).
//
//  The network (BatchNorm already folded into the layers by the exporter):
//    3 blocks of [conv 3x3 -> ReLU -> conv 3x3 -> ReLU -> max pool 2x2]    28 -> 14 -> 7 -> 3
//    dense 1152 -> 256 -> ReLU -> dense 256 -> 42 scores
//  Exactly the same computation as PyTorch in evaluation mode (dropout does nothing there).
// =====================================================================================
(function (root) {
  // a 3x3 convolution with padding 1, followed by ReLU.  x: [cin, H, W] -> [cout, H, W]
  function conv3x3(x, cin, H, W, w, b, cout) {
    const out = new Float32Array(cout * H * W);
    const HW = H * W;
    for (let o = 0; o < cout; o++) {
      const op = o * HW;
      out.fill(b[o], op, op + HW);
      for (let c = 0; c < cin; c++) {
        const ip = c * HW, wp = (o * cin + c) * 9;
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            const wv = w[wp + ky * 3 + kx];
            if (wv === 0) continue;
            const dy = ky - 1, dx = kx - 1;
            const y0 = Math.max(0, -dy), y1 = Math.min(H, H - dy);
            const x0 = Math.max(0, -dx), x1 = Math.min(W, W - dx);
            for (let y = y0; y < y1; y++) {
              const orow = op + y * W, irow = ip + (y + dy) * W + dx;
              for (let xx = x0; xx < x1; xx++) out[orow + xx] += wv * x[irow + xx];
            }
          }
        }
      }
    }
    for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
    return out;
  }

  // 2x2 max pooling (an odd size loses its last row/column, like PyTorch)
  function maxpool(x, C, H, W) {
    const h = Math.floor(H / 2), w = Math.floor(W / 2), out = new Float32Array(C * h * w);
    for (let c = 0; c < C; c++) for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) {
      const i = c * H * W + 2 * y * W + 2 * xx;
      out[(c * h + y) * w + xx] = Math.max(x[i], x[i + 1], x[i + W], x[i + W + 1]);
    }
    return out;
  }

  function dense(x, w, b, nin, nout, relu) {
    const out = new Float32Array(nout);
    for (let o = 0; o < nout; o++) {
      let s = b[o];
      const wp = o * nin;
      for (let i = 0; i < nin; i++) s += w[wp + i] * x[i];
      out[o] = relu && s < 0 ? 0 : s;
    }
    return out;
  }

  class CNN {
    constructor(meta, buffer) {
      this.meta = meta;
      const all = new Float32Array(buffer);
      if (all.length !== meta.n_weights) throw new Error(`weights.bin has ${all.length} numbers, expected ${meta.n_weights}`);
      this.layers = meta.layers.map((L) => {
        if (L.type === "conv") return { ...L, W: all.subarray(L.w, L.w + L.out * L.in * 9), B: all.subarray(L.b, L.b + L.out) };
        if (L.type === "dense") return { ...L, W: all.subarray(L.w, L.w + L.out * L.in), B: all.subarray(L.b, L.b + L.out) };
        return L;
      });
    }

    static async load(url) {
      const meta = await (await fetch(`${url}/model.json`)).json();
      const buf = await (await fetch(`${url}/weights.bin`)).arrayBuffer();
      return new CNN(meta, buf);
    }

    // x: Float32Array of 28*28 values in 0..1 -> raw scores (logits)
    logits(x) {
      let a = x, C = 1, H = 28, W = 28;
      for (const L of this.layers) {
        if (L.type === "conv") { a = conv3x3(a, C, H, W, L.W, L.B, L.out); C = L.out; }
        else if (L.type === "maxpool") { a = maxpool(a, C, H, W); H = Math.floor(H / 2); W = Math.floor(W / 2); }
        else a = dense(a, L.W, L.B, L.in, L.out, L.relu);
      }
      return a;
    }

    // like the server: the drawing and its mirror image, averaged, then softmax
    predict(pixels) {
      const x = new Float32Array(784), xf = new Float32Array(784);
      for (let y = 0; y < 28; y++) for (let i = 0; i < 28; i++) {
        const v = pixels[y * 28 + i] / 255;
        x[y * 28 + i] = v;
        xf[y * 28 + (27 - i)] = v;
      }
      const a = this.logits(x), b = this.logits(xf), z = a.map((v, i) => (v + b[i]) / 2);
      const m = Math.max(...z);
      const e = z.map((v) => Math.exp(v - m));
      const s = e.reduce((p, q) => p + q, 0);
      return e.map((v) => v / s);
    }
  }

  root.DoodleCNN = CNN;
  if (typeof module !== "undefined") module.exports = { CNN, conv3x3, maxpool, dense };
})(typeof window !== "undefined" ? window : globalThis);
