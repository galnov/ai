// Runs the network in a background thread, so drawing never stutters while it thinks.
importScripts("cnn.js");
let net = null;
onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      net = await self.DoodleCNN.load(m.url);
      postMessage({ id: m.id, ok: true });
    } else if (m.type === "predict") {
      postMessage({ id: m.id, probs: Array.from(net.predict(m.pixels)) });
    }
  } catch (err) {
    postMessage({ id: m.id, error: String(err.message || err) });
  }
};
