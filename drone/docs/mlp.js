// =====================================================================================
//  Small fully connected networks in plain JavaScript - the trained networks, run in the browser.
//
//  model/model.json describes the networks; model/weights.bin holds all their numbers (float32).
//  Both are written by  python -m src.export_web  from the trained PyTorch model.
//
//  Every layer computes  y = act(W x + b),  W stored row by row (out x in), like PyTorch's nn.Linear.
//  "norm" (optional): the observation normalization saved with a PPO network:
//      x <- clip((x - mean) / sqrt(var + 1e-8), -10, 10)
// =====================================================================================
(function (root) {
  const ACT = {
    relu: (v) => (v > 0 ? v : 0),
    tanh: Math.tanh,
    none: (v) => v,
  };

  class Net {
    constructor(spec, buf) {
      Object.assign(this, spec);
      const view = (offset, n) => buf.subarray(offset, offset + n);
      this.layers = spec.layers.map((L) => ({ ...L, W: view(L.w, L.in * L.out), B: view(L.b, L.out), f: ACT[L.act || "none"] }));
      if (spec.norm) {
        const n = spec.layers[0].in;
        this.mean = view(spec.norm.mean, n);
        this.var = view(spec.norm.var, n);
      }
    }

    // x: an array of numbers. Returns { out, hidden: [activations of every hidden layer] }
    forward(x) {
      let h = Float32Array.from(x);
      if (this.mean) {
        for (let i = 0; i < h.length; i++) h[i] = Math.max(-10, Math.min(10, (h[i] - this.mean[i]) / Math.sqrt(this.var[i] + 1e-8)));
      }
      const hidden = [];
      this.layers.forEach((L, k) => {
        const y = new Float32Array(L.out);
        // only the inputs that are not zero (after ReLU, and in a 0/1 picture, most of them are)
        const nz = [];
        for (let i = 0; i < L.in; i++) if (h[i] !== 0) nz.push(i);
        for (let o = 0; o < L.out; o++) {
          let s = L.B[o];
          const row = o * L.in;
          for (let j = 0; j < nz.length; j++) s += L.W[row + nz[j]] * h[nz[j]];
          y[o] = L.f(s);
        }
        if (k < this.layers.length - 1) hidden.push(y);
        h = y;
      });
      return { out: Array.from(h), hidden };
    }

    // the weights of layer k as rows (for drawing the connections)
    rows(k) {
      const L = this.layers[k], out = [];
      for (let o = 0; o < L.out; o++) out.push(L.W.subarray(o * L.in, (o + 1) * L.in));
      return out;
    }
  }

  // Loads model/model.json + model/weights.bin. Returns null if the files are not there (no trained network yet).
  async function load(dir = "model") {
    let meta;
    try {
      const r = await fetch(`${dir}/model.json`, { cache: "no-cache" });
      if (!r.ok) return null;
      meta = await r.json();
    } catch (e) {
      return null;
    }
    const r = await fetch(`${dir}/weights.bin`, { cache: "no-cache" });
    if (!r.ok) throw new Error("model/weights.bin is missing - run  python -m src.export_web  again");
    const buf = new Float32Array(await r.arrayBuffer());
    if (meta.n_weights && buf.length !== meta.n_weights) throw new Error("model/weights.bin does not match model.json - export again");
    const nets = {};
    for (const spec of meta.networks) nets[spec.id] = new Net(spec, buf);
    return { meta, nets };
  }

  root.MLP = { load, Net };
})(typeof window !== "undefined" ? window : globalThis);
