// A background thread that runs the camera network (counter.js), so the page stays smooth.
importScripts("counter.js");
let layers = null, buf = null;
onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      const meta = await (await fetch(`${m.url}/model.json`, { cache: "no-cache" })).json();
      buf = new Float32Array(await (await fetch(`${m.url}/weights.bin`, { cache: "no-cache" })).arrayBuffer());
      layers = meta.counter.layers;
      postMessage({ id: m.id, ok: true });
    } else if (m.type === "count") {
      const r = Counter.count(m.pixels, layers, buf);
      postMessage({ id: m.id, count: r.count, density: r.density, h: r.h, w: r.w }, [r.density.buffer]);
    }
  } catch (err) {
    postMessage({ id: m.id, error: String(err && err.message || err) });
  }
};
