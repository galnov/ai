// =====================================================================================
//  The "server" of the static version: everything the Flask server did, done in the browser.
//  The same addresses as app/server.py, answered here instead of over the network:
//     GET /api/info     POST /api/config, /api/reset, /api/step
//  The simulation is in sim.js, the cameras in camera.js. The two networks come from docs/model
//  (python -m src.export_web): the traffic-light agent runs in mlp.js, and the camera network runs in
//  counter.js, in four background threads (one per camera) so the page stays smooth.
// =====================================================================================
(function (root) {
  const { C, World, PRESETS } = root.TrafficSim;
  const MODEL = "model";
  let world = null, info = null;

  // ---------------------------------------------------------------- the camera network, in background threads
  function makePool(n) {
    const workers = [], waiting = new Map();
    let nextId = 1;
    for (let i = 0; i < n; i++) {
      const w = new Worker("counter-worker.js");
      w.onmessage = (e) => {
        const p = waiting.get(e.data.id);
        if (!p) return;
        waiting.delete(e.data.id);
        e.data.error ? p.reject(new Error(e.data.error)) : p.resolve(e.data);
      };
      workers.push(w);
    }
    const call = (i, msg, transfer) => new Promise((resolve, reject) => {
      const id = nextId++;
      waiting.set(id, { resolve, reject });
      workers[i % n].postMessage({ ...msg, id }, transfer || []);
    });
    return {
      load: () => Promise.all(workers.map((_, i) => call(i, { type: "load", url: MODEL }))),
      count: (i, pixels) => call(i, { type: "count", pixels }, [pixels.buffer]),
    };
  }

  async function makeCamera() {
    const scene = await root.Cameras.loadScene(".");
    const pool = makePool(Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2)));
    await pool.load();
    return async (vehicles, condition) => {
      const views = await root.Cameras.renderCameras(scene, vehicles, condition);
      const res = await Promise.all(views.map((v, i) => pool.count(i, v.net)));
      let mx = 0;
      for (const r of res) for (const v of r.density) mx = Math.max(mx, v);
      const norm = Math.max(0.05, mx);
      return {
        images: views.map((v) => v.url),
        pred: res.map((r) => Math.round(r.count * 100) / 100),
        density: res.map((r) => Array.from({ length: r.h }, (_, y) =>
          Array.from(r.density.subarray(y * r.w, (y + 1) * r.w), (v) => Math.round((v / norm) * 1000) / 1000))),
      };
    };
  }

  async function init() {
    const errors = [];
    let controller = null, camera = null, meta = null;
    try {
      const model = await root.MLP.load(MODEL);
      if (model) {
        meta = model.meta;
        const net = model.nets.agent;
        if (net) {
          controller = (obs) => {
            const q = net.forward(obs).out;
            return { action: q[1] > q[0] ? 1 : 0, q };
          };
        }
        if (meta.counter) camera = await makeCamera();
      }
    } catch (e) {
      errors.push(`Could not load the networks: ${e.message}`);
    }
    if (!controller) errors.push("No trained agent has been exported to this page yet (python -m src.export_web).");
    world = new World(controller, camera);
    info = {
      errors, has_agent: !!controller, has_counter: !!camera,
      agent: { val_wait: meta?.agent?.val_wait ?? null, baselines: meta?.agent?.baselines ?? null },
      counter: { val_mae: meta?.counter?.val_mae ?? null },
      presets: Object.entries(PRESETS).map(([id, v]) => ({ id, name: v[0] })),
      conditions: C.CONDITIONS.map((c) => ({ id: c, name: C.CONDITION_NAMES[c] })),
      lanes: C.LANES.map((l) => C.LANE_NAMES[l]),
      dt: C.DT, view_range: C.VIEW_RANGE, vehicle_lengths: C.VEHICLES.map((v) => v[1]),
    };
  }

  const ROUTES = {
    "/api/info": () => ({ ...info, settings: { preset: world.preset, custom: world.custom, left_mode: world.leftMode,
                                                sensor: world.sensor, condition: world.condition } }),
    "/api/config": (d) => {
      if (PRESETS[d.preset]) world.preset = d.preset;
      if (Array.isArray(d.custom) && d.custom.length === 4) world.custom = d.custom.map((v) => Math.min(0.4, Math.max(0, +v)));
      if (d.left_mode === "fixed" || d.left_mode === "max_queue") world.leftMode = d.left_mode;
      if (d.sensor === "camera" || d.sensor === "perfect") world.sensor = d.sensor === "perfect" || world.camera ? d.sensor : "perfect";
      if (C.CONDITIONS.includes(d.condition)) world.condition = d.condition;
      return { ok: true, sensor: world.sensor };
    },
    "/api/reset": () => { world.reset(); return world.state(); },
    "/api/step": async (d) => {
      const st = await world.step(Math.max(0, Math.min(d.n | 0, 40)));
      if (world.cam && d.cam_seq !== world.camSeq) st.camera = world.cam;
      return st;
    },
  };

  // requests are answered one at a time, in order (a step may wait for the camera network)
  let queue = null;
  root.Backend = {
    request(url, body) {
      if (!queue) queue = init();
      const route = ROUTES[url];
      if (!route) return Promise.reject(new Error(`unknown address ${url}`));
      const p = queue.then(() => route(body || {}));
      queue = p.catch(() => {});
      return p;
    },
  };
})(typeof window !== "undefined" ? window : globalThis);
