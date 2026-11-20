// A background thread for the "server" work: finding the faces (faces.js) and running the emotion network
// on each of them (emotion-cnn.js), so the page stays smooth.
importScripts("faces.js", "emotion-cnn.js");
let M = null, buf = null, cascade = null;

// like expand_box in src/vision.py: a slightly bigger box (FER images include a small margin too)
function expandBox([x, y, w, h], W, H, margin = 0.08) {
  const dx = Math.trunc(w * margin), dy = Math.trunc(h * margin);
  const x0 = Math.max(0, x - dx), y0 = Math.max(0, y - dy), x1 = Math.min(W, x + w + dx), y1 = Math.min(H, y + h + dy);
  return [x0, y0, x1 - x0, y1 - y0];
}

function crop(gray, W, [x, y, w, h]) {
  const out = new Uint8Array(w * h);
  for (let r = 0; r < h; r++) out.set(gray.subarray((y + r) * W + x, (y + r) * W + x + w), r * w);
  return out;
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      cascade = new Faces.Cascade(await (await fetch("face_cascade.json")).json());
      if (m.model) {                                       // without a trained network, only the faces are found
        M = await (await fetch(`${m.url}/model.json`, { cache: "no-cache" })).json();
        buf = new Float32Array(await (await fetch(`${m.url}/weights.bin`, { cache: "no-cache" })).arrayBuffer());
      }
      postMessage({ id: m.id, ok: true });
    } else if (m.type === "analyze") {
      const { gray, W, H } = m;
      // find the faces in a smaller copy of the image (much faster; faces are big enough for it)
      const s = Math.max(1, W / m.detectWidth), dw = Math.round(W / s), dh = Math.round(H / s);
      const small = s > 1 ? Faces.resizeArea(gray, W, H, dw, dh) : gray;
      const boxes = Faces.detectFaces(cascade, small, dw, dh, 4).map((r) => r.map((v) => Math.round(v * s)));
      const faces = boxes.map((box) => {
        const cb = expandBox(box, W, H);
        if (!M) return { box, crop_box: cb, probs: null, top: null, confidence: 0, cam: null };
        const face48 = Faces.resizeArea(crop(gray, W, cb), cb[2], cb[3], 48, 48);
        const { probs, cam } = EmotionNet.predictFace(face48, M, buf, m.gradcam, m.tta);
        let top = 0;
        for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
        return { box, crop_box: cb, probs: Object.fromEntries(M.emotions.map((k, i) => [k, Math.round(probs[i] * 1e4) / 1e4])),
                 top: M.emotions[top], confidence: Math.round(probs[top] * 1e4) / 1e4, cam };
      });
      postMessage({ id: m.id, faces });
    }
  } catch (err) {
    postMessage({ id: m.id, error: String((err && err.message) || err) });
  }
};
