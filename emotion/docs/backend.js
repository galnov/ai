// =====================================================================================
//  The "server" of the static version: what app/server.py and src/predictor.py did, done in the browser.
//    info     - the emotions and the model's accuracy (docs/model/model.json)
//    predict  - a frame from the camera (or a photo) -> the faces in it (faces.js) -> each face cropped to
//               48 x 48 -> the CNN (emotion-cnn.js) -> probabilities, and a Grad-CAM heat map if asked.
//               All of this runs in a background thread (emotion-worker.js).
//  Nothing leaves the visitor's computer: the camera images are analyzed right here, in the browser.
// =====================================================================================
(function (root) {
  const MODEL_URL = "model";
  let meta = null, worker = null, nextId = 1;
  const waiting = new Map();

  function call(msg, transfer) {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      waiting.set(id, { resolve, reject });
      worker.postMessage({ ...msg, id }, transfer || []);
    });
  }

  // the Grad-CAM map (48 x 48, values 0..1) -> a transparent colored picture (like heatmap_png_b64 in vision.py)
  function heatmapUrl(cam) {
    const cv = document.createElement("canvas");
    cv.width = cv.height = 48;
    const ctx = cv.getContext("2d"), im = ctx.createImageData(48, 48);
    const jet = (v) => {                                  // the "jet" color map: blue -> cyan -> yellow -> red
      const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3))), g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2)));
      const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1)));
      return [r * 255, g * 255, b * 255];
    };
    for (let i = 0; i < cam.length; i++) {
      const [r, g, b] = jet(cam[i]);
      im.data.set([r, g, b, Math.pow(cam[i], 1.3) * 190], i * 4);
    }
    ctx.putImageData(im, 0, 0);
    return cv.toDataURL("image/png");
  }

  root.Backend = {
    // ready: the network is loaded. detector: faces can be found (also without a trained network)
    async info() {
      let error = null, detector = false;
      try {
        const r = await fetch(`${MODEL_URL}/model.json`, { cache: "no-cache" });
        if (r.ok) meta = await r.json();
        else error = "No trained network has been exported to this page yet, so the faces are found but their emotion is not. Train it (python -m src.train), then run  python -m src.export_web  and publish again.";
      } catch (e) {
        error = e.message;
      }
      try {
        worker = new Worker("emotion-worker.js");
        worker.onmessage = (e) => {
          const w = waiting.get(e.data.id);
          if (!w) return;
          waiting.delete(e.data.id);
          e.data.error ? w.reject(new Error(e.data.error)) : w.resolve(e.data);
        };
        await call({ type: "load", url: MODEL_URL, model: !!meta });
        detector = true;
      } catch (e) {
        error = (error ? error + "\n" : "") + `Could not start the analysis: ${e.message}`;
        meta = null;
      }
      return { ready: !!meta, detector, error, val_acc: meta?.val_acc, train_samples: meta?.train_samples, epoch: meta?.epoch,
               device: "in this browser", emotions: meta ? meta.categories : FALLBACK };
    },

    // canvas: the frame to analyze. live: a camera frame (faster settings) or a photo
    async predict(canvas, { gradcam = false, live = false } = {}) {
      const W = canvas.width, H = canvas.height;
      const px = canvas.getContext("2d", { willReadFrequently: true }).getImageData(0, 0, W, H).data;
      const gray = new Uint8Array(W * H);                    // like cv2.COLOR_BGR2GRAY
      for (let i = 0, j = 0; i < gray.length; i++, j += 4) gray[i] = Math.round(0.299 * px[j] + 0.587 * px[j + 1] + 0.114 * px[j + 2]);
      const t0 = performance.now();
      const r = await call({ type: "analyze", gray, W, H, gradcam, tta: !live, detectWidth: live ? 320 : 640 }, [gray.buffer]);
      const faces = r.faces.map(({ cam, ...f }) => (cam ? { ...f, heatmap: heatmapUrl(cam) } : f));
      return { faces, width: W, height: H, ms: Math.round(performance.now() - t0) };
    },
  };

  // the emotions, for drawing the page when there is no model (the same as src/config.py)
  const FALLBACK = [["angry", "Angry", "😠", "#ff4d4f"], ["disgust", "Disgust", "🤢", "#7cb342"], ["fear", "Fear", "😨", "#9c6ade"],
                    ["happy", "Happy", "😄", "#ffc53d"], ["neutral", "Neutral", "😐", "#8c9bab"], ["sad", "Sad", "😢", "#4096ff"],
                    ["surprise", "Surprise", "😲", "#ff7a45"]].map(([key, name, emoji, color]) => ({ key, name, emoji, color }));
})(typeof window !== "undefined" ? window : globalThis);
