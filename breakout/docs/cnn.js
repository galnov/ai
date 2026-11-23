// =====================================================================================
//  Runs the exported convolutional networks in the browser (no libraries).
//  The layers come from docs/model/model.json; the numbers from docs/model/weights.bin
//  (float16 to halve the download, turned back into float32 here).
//  Also: Grad-CAM - which parts of the screen the network's decision depends on.
// =====================================================================================
(function (root) {
  function f16ToF32(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      out[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
    return out;
  }

  class Net {
    constructor(spec, buf) {
      this.id = spec.id;
      this.label = spec.label;
      this.layers = spec.layers.map((L) => {
        const nw = L.type === "conv" ? L.out * L.in * L.k * L.k : L.out * L.in;
        return { ...L, W: buf.subarray(L.w, L.w + nw), B: buf.subarray(L.b, L.b + L.out) };
      });
    }

    // x: Float32Array (channels * size * size), values 0..255. Returns { logits, maps (the last conv layer's
    // output, for Grad-CAM), hidden }
    forward(x, size = 48) {
      let a = new Float32Array(x.length);
      for (let i = 0; i < x.length; i++) a[i] = x[i] / 255;
      let ch = this.layers[0].in, n = size, maps = null, mapsN = 0, hidden = null;
      for (const L of this.layers) {
        if (L.type === "conv") {
          const k = L.k, s = L.stride, m = Math.floor((n - k) / s) + 1, out = new Float32Array(L.out * m * m);
          for (let o = 0; o < L.out; o++) {
            const bias = L.B[o];
            for (let y = 0; y < m; y++) for (let xx = 0; xx < m; xx++) {
              let sum = bias;
              for (let c = 0; c < L.in; c++) {
                const wBase = ((o * L.in + c) * k) * k, aBase = c * n * n;
                for (let i = 0; i < k; i++) {
                  const row = aBase + (y * s + i) * n + xx * s, wr = wBase + i * k;
                  for (let j = 0; j < k; j++) sum += L.W[wr + j] * a[row + j];
                }
              }
              out[(o * m + y) * m + xx] = sum > 0 ? sum : 0;
            }
          }
          a = out; ch = L.out; n = m; maps = out; mapsN = m;
        } else {
          const out = new Float32Array(L.out);
          for (let o = 0; o < L.out; o++) {
            let sum = L.B[o];
            const base = o * L.in;
            for (let i = 0; i < L.in; i++) sum += L.W[base + i] * a[i];
            out[o] = L.act === "relu" ? (sum > 0 ? sum : 0) : sum;
          }
          if (L.act === "relu") hidden = out;
          a = out;
        }
      }
      return { logits: Array.from(a), maps, mapsCh: ch, mapsN, hidden };
    }

    act(x) {
      const r = this.forward(x);
      let best = 0;
      r.logits.forEach((v, i) => { if (v > r.logits[best]) best = i; });
      const mx = Math.max(...r.logits), e = r.logits.map((v) => Math.exp(v - mx)), z = e.reduce((p, q) => p + q, 0);
      r.action = best;
      r.probs = e.map((v) => v / z);
      return r;
    }

    // Grad-CAM for the chosen action: how much each place on the last conv layer's 7x7 grid pushed the decision.
    // Returns Float32Array(mapsN * mapsN) scaled to 0..1.
    gradcam(r) {
      const L3 = this.layers[this.layers.length - 2], L4 = this.layers[this.layers.length - 1];
      const a = r.action, h = r.hidden, nh = L3.out, nin = L3.in;
      // d logit_a / d hidden = W4[a] where the hidden unit is active
      const gh = new Float32Array(nh);
      for (let j = 0; j < nh; j++) gh[j] = h[j] > 0 ? L4.W[a * L4.in + j] : 0;
      // d logit_a / d maps = W3^T gh
      const g = new Float32Array(nin);
      for (let j = 0; j < nh; j++) {
        if (!gh[j]) continue;
        const base = j * nin, v = gh[j];
        for (let i = 0; i < nin; i++) g[i] += L3.W[base + i] * v;
      }
      const ch = r.mapsCh, m = r.mapsN, mm = m * m, cam = new Float32Array(mm);
      for (let c = 0; c < ch; c++) {
        let alpha = 0;
        for (let i = 0; i < mm; i++) alpha += g[c * mm + i];
        alpha /= mm;
        for (let i = 0; i < mm; i++) cam[i] += alpha * r.maps[c * mm + i];
      }
      let mx = 0;
      for (let i = 0; i < mm; i++) { cam[i] = Math.max(0, cam[i]); mx = Math.max(mx, cam[i]); }
      if (mx > 0) for (let i = 0; i < mm; i++) cam[i] /= mx;
      return cam;
    }
  }

  async function load(dir) {
    const meta = await fetch(`${dir}/model.json`).then((r) => {
      if (!r.ok) throw new Error(`No exported network (${dir}/model.json: ${r.status})`);
      return r.json();
    });
    const raw = await fetch(`${dir}/weights.bin`).then((r) => {
      if (!r.ok) throw new Error(`weights.bin: ${r.status}`);
      return r.arrayBuffer();
    });
    const buf = meta.dtype === "f16" ? f16ToF32(new Uint16Array(raw)) : new Float32Array(raw);
    const nets = {};
    for (const spec of meta.networks) nets[spec.id] = new Net(spec, buf);
    return { meta, nets };
  }

  root.CNN = { load, Net, f16ToF32 };
})(typeof window !== "undefined" ? window : globalThis);
