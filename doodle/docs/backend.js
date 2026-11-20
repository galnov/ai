// =====================================================================================
//  The "server" of the static version: everything the Flask server did, done in the browser.
//    info     - the categories and the model's accuracy (docs/model/model.json)
//    predict  - drawing -> 28x28 image (render.js) -> the CNN (cnn.js, in worker.js) -> top 5
//    save     - your drawings are kept in this browser; "Download my drawings" saves them to a file,
//               and  python -m src.import_drawings  adds them to the training data
//    samples  - real drawings from the dataset (docs/model/samples.json)
// =====================================================================================
(function (root) {
  const MODEL_URL = "model";
  const STORE = "doodle.drawings";
  const TOP_K = 5;
  let meta = null, samples = null, worker = null, nextId = 1;
  const waiting = new Map();

  function call(msg, transfer) {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      waiting.set(id, { resolve, reject });
      worker.postMessage({ ...msg, id }, transfer || []);
    });
  }

  const b64ToPixels = (s) => Array.from(atob(s), (ch) => ch.charCodeAt(0));
  const pixelsToB64 = (px) => btoa(String.fromCharCode(...px));
  const stored = () => { try { return JSON.parse(localStorage.getItem(STORE) || "[]"); } catch { return []; } };

  root.Backend = {
    async info() {
      try {
        const r = await fetch(`${MODEL_URL}/model.json`);
        if (!r.ok) throw new Error("the model files are missing - run  python -m src.export_web  and publish the docs folder again");
        meta = await r.json();
        worker = new Worker("worker.js");
        worker.onmessage = (e) => {
          const w = waiting.get(e.data.id);
          if (!w) return;
          waiting.delete(e.data.id);
          e.data.error ? w.reject(new Error(e.data.error)) : w.resolve(e.data);
        };
        await call({ type: "load", url: MODEL_URL });
        try { const s = await fetch(`${MODEL_URL}/samples.json`); samples = s.ok ? await s.json() : null; } catch { samples = null; }
        return { ready: true, error: null, val_acc: meta.val_acc, val_top3: meta.val_top3, quick: meta.quick,
                 categories: meta.categories, has_samples: !!samples };
      } catch (e) {
        return { ready: false, error: e.message, categories: meta ? meta.categories : [], has_samples: false };
      }
    },

    async predict(strokes) {
      const img = DoodleRender.strokesToImage(strokes);
      if (!img.some((v) => v > 0)) return { top: [], image: Array.from(img) };
      const { probs } = await call({ type: "predict", pixels: img });
      const order = probs.map((p, i) => i).sort((a, b) => probs[b] - probs[a]).slice(0, TOP_K);
      return { top: order.map((i) => ({ key: meta.classes[i], p: probs[i] })), image: Array.from(img) };
    },

    async save(strokes, label) {
      const img = DoodleRender.strokesToImage(strokes);
      if (!img.some((v) => v > 0)) throw new Error("The drawing is empty");
      const all = stored();
      all.push({ label, pixels: pixelsToB64(img) });
      localStorage.setItem(STORE, JSON.stringify(all));
      return { saved: all.filter((d) => d.label === label).length, total: all.length };
    },

    savedCount: () => stored().length,

    downloadDrawings() {
      const blob = new Blob([JSON.stringify({ format: "doodle-drawings-v1", size: 28, drawings: stored() })], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "my_drawings.json";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    },

    clearDrawings() { localStorage.removeItem(STORE); },

    async samples(i) {
      const key = meta.classes[i];
      return { images: (samples && samples[key] ? samples[key] : []).map(b64ToPixels) };
    },
  };
})(window);
