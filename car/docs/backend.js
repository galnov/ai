// =====================================================================================
//  The "server" of the static version: everything the Flask server did, done in the browser.
//  The same addresses as app/server.py, answered here instead of over the network:
//     GET /api/info     POST /api/track, /api/start, /api/step
//  It is a JavaScript copy of src/track.py (building a track from a drawing, random tracks),
//  src/env.py (the sensors, the physics, laps and crashes) and app/sim.py (the cars in the app).
//  The networks come from docs/model (python -m src.export_web) and run in mlp.js.
// =====================================================================================
(function (root) {
  const { C, Track, CarEnv, raycast } = root.CarSim;     // carsim.js: the physics (shared with the leaderboard server)
  const CAR_COLORS = ["#ffd23f", "#3bceac", "#ff6b6b", "#7aa2ff", "#c792ea", "#ff9f43", "#5ee27a", "#f78fb3"];
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const rnd = (lo, hi) => lo + Math.random() * (hi - lo);

  // ---------------------------------------------------------------- geometry (src/track.py)
  function resampleClosed(points, spacing = C.TRACK_SPACING) {
    const pts = [];                                             // a point equal to the one before it is removed
    for (let i = 0; i < points.length; i++) { const prev = i === 0 ? null : points[i - 1]; if (i === 0 || Math.hypot(points[i][0] - prev[0], points[i][1] - prev[1]) > 1e-6) pts.push(points[i]); }
    const closed = [...pts, pts[0]], cum = [0];
    for (let i = 1; i < closed.length; i++) cum.push(cum[i - 1] + Math.hypot(closed[i][0] - closed[i - 1][0], closed[i][1] - closed[i - 1][1]));
    const total = cum[cum.length - 1], n = Math.max(16, Math.round(total / spacing)), out = [];
    let j = 0;
    for (let k = 0; k < n; k++) {
      const s = (total * k) / n;
      while (j < cum.length - 2 && cum[j + 1] < s) j++;
      const seg = cum[j + 1] - cum[j], t = seg > 0 ? (s - cum[j]) / seg : 0;
      out.push([closed[j][0] + (closed[j + 1][0] - closed[j][0]) * t, closed[j][1] + (closed[j + 1][1] - closed[j][1]) * t]);
    }
    return out;
  }

  function smoothClosed(points, window = 5, iters = 1) {
    let pts = points;
    const h = Math.floor(window / 2);
    for (let it = 0; it < iters; it++) {
      const n = pts.length;
      pts = pts.map((_, i) => {
        let x = 0, y = 0;
        for (let k = -h; k <= h; k++) { const p = pts[(i + k + n) % n]; x += p[0]; y += p[1]; }
        return [x / window, y / window];
      });
    }
    return pts;
  }

  function curvatureRadius(points, step = 3) {
    const n = points.length;
    return points.map((b, i) => {
      const a = points[(i - step + n) % n], c = points[(i + step) % n];
      const ab = Math.hypot(b[0] - a[0], b[1] - a[1]), bc = Math.hypot(c[0] - b[0], c[1] - b[1]), ca = Math.hypot(a[0] - c[0], a[1] - c[1]);
      const cross = Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
      return (ab * bc * ca) / (2 * cross + 1e-9);
    });
  }

  function selfIntersects(p) {
    const n = p.length, cr = (u, v) => u[0] * v[1] - u[1] * v[0];
    const d = p.map((a, i) => { const b = p[(i + 1) % n]; return [b[0] - a[0], b[1] - a[1]]; });
    for (let i = 0; i < n; i++) for (let j = i + 2; j < n; j++) {
      if (j - i >= n - 1) continue;                               // the first and last segments are neighbors
      const den = cr(d[i], d[j]);
      if (Math.abs(den) <= 1e-12) continue;
      const w = [p[j][0] - p[i][0], p[j][1] - p[i][1]];
      const t = cr(w, d[j]) / den, u = cr(w, d[i]) / den;
      if (t > 0 && t < 1 && u > 0 && u < 1) return true;
    }
    return false;
  }

  function partsTooClose(p, width) {
    const n = p.length;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const g = Math.min(Math.abs(i - j), n - Math.abs(i - j)) * C.TRACK_SPACING;
      if (g > width * 2.0 && Math.hypot(p[i][0] - p[j][0], p[i][1] - p[j][1]) < width * 1.15) return true;
    }
    return false;
  }

  class TrackError extends Error {}

  function fitToWorld(points, width, fill = false) {
    const margin = width / 2 + 12;
    const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
    const lo = [Math.min(...xs), Math.min(...ys)], hi = [Math.max(...xs), Math.max(...ys)];
    let s = Math.min((C.WORLD_W - 2 * margin) / Math.max(hi[0] - lo[0], 1), (C.WORLD_H - 2 * margin) / Math.max(hi[1] - lo[1], 1));
    if (!fill) s = Math.min(1.0, s);
    const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2;
    return points.map(([x, y]) => [(x - cx) * s + C.WORLD_W / 2, (y - cy) * s + C.WORLD_H / 2]);
  }

  function makeTrack(points, width, name = "track", maxSmooth = 30, fill = false) {
    let pts = points;
    if (pts.length < 8) throw new TrackError("The track is too short - draw a bigger loop");
    if (fill) pts = fitToWorld(pts, width, true);
    pts = smoothClosed(resampleClosed(pts), 5, 2);
    const minR = (width / 2) * 1.25;
    for (let i = 0; i < maxSmooth; i++) {
      if (Math.min(...curvatureRadius(pts)) >= minR) break;
      pts = resampleClosed(smoothClosed(pts, 7, 2));
    }
    pts = fitToWorld(resampleClosed(pts), width, fill);
    if (Math.min(...curvatureRadius(pts)) < (width / 2) * 1.05) throw new TrackError("The track has a turn that is too sharp. Draw wider turns or make the road narrower");
    if (selfIntersects(pts)) throw new TrackError("The track crosses itself. Draw a single loop without crossings");
    if (partsTooClose(pts, width)) throw new TrackError("Parts of the track are too close to each other. Move them apart or make the road narrower");
    const t = new Track(pts, width, name);
    if (t.length < 1200) throw new TrackError("The track is too short - draw a bigger loop");
    return t;
  }

  function randomTrack() {
    for (let attempt = 0; attempt < 100; attempt++) {
      const w = rnd(...C.TRACK_WIDTH_RANGE);
      let pts;
      if (Math.random() < 0.6) {                                   // polygon: sharp turns (the driver must brake)
        const k = 7 + Math.floor(Math.random() * 7);
        const th = Array.from({ length: k }, () => rnd(0, 2 * Math.PI)).sort((a, b) => a - b);
        const rx = rnd(460, 540), ry = rnd(290, 340);
        pts = th.map((a) => { const r = rnd(0.45, 1.0); return [rx * r * Math.cos(a), ry * r * Math.sin(a)]; });
      } else {                                                     // fourier: gentle, flowing turns
        const amps = [], phases = [];
        for (let k = 2; k <= 5; k++) { amps.push(rnd(0, 0.42 / k ** 0.6)); phases.push(rnd(0, 2 * Math.PI)); }
        const rx = rnd(430, 520), ry = rnd(270, 330);
        pts = Array.from({ length: 240 }, (_, i) => {
          const a = (2 * Math.PI * i) / 240;
          let r = 1;
          amps.forEach((am, j) => (r += am * Math.cos((j + 2) * a + phases[j])));
          r = Math.max(r, 0.35);
          return [rx * r * Math.cos(a), ry * r * Math.sin(a)];
        });
      }
      const rot = rnd(-0.25, 0.25), c = Math.cos(rot), s = Math.sin(rot);
      pts = pts.map(([x, y]) => [x * c - y * s, x * s + y * c]);
      try {
        const t = makeTrack(pts, w, "Random", 8, true);
        return Math.random() < 0.5 ? t.reversed() : t;
      } catch (e) {
        if (!(e instanceof TrackError)) throw e;
      }
    }
    throw new Error("Could not create a random track");
  }

  // ---------------------------------------------------------------- the cars (src/env.py)
  // ---------------------------------------------------------------- the app's simulation (app/sim.py)
  class Sim {
    constructor(drivers) { this.drivers = drivers; this.track = null; this.env = null; this.cars = []; }

    setTrack(track) { this.track = track; this.env = null; this.cars = []; }

    start(cars, spread) {
      this.cars = cars;
      const n = cars.length;
      if (!n) { this.env = null; return; }
      const t = this.track, e = (this.env = new CarEnv(t, n));
      if (spread) {                                               // watch mode: spread out along the track
        e.cars.forEach((c, i) => {
          const k = Math.floor((i * t.n) / n) % t.n;
          c.idx = k; [c.x, c.y] = t.center[k]; c.heading = Math.atan2(t.tangent[k][1], t.tangent[k][0]);
        });
      } else {                                                    // race: all on the start line, side by side
        e.cars.forEach((c, i) => {
          const [ox, oy] = root.CarSim.startOffset(t, n, i);
          c.x += ox; c.y += oy;
        });
      }
      // the human's keys, one digit per step (for the leaderboard: the server replays the race from them)
      this.human = { index: cars.findIndex((c) => c.kind === "human"), n, keys: [] };
      e.observe();
    }

    focus() {
      const ai = this.cars.map((c, i) => (c.kind === "ai" ? i : -1)).filter((i) => i >= 0);
      const best = ai.filter((i) => this.cars[i].driver === "best");
      return (best.length ? best : ai.length ? ai : [-1])[0];
    }

    step(nSteps, keys) {
      const env = this.env;
      if (!env) return { cars: [], t: 0, crashes: [], brain: null };
      const crashes = [];
      let brain = null;
      for (let s = 0; s < nSteps; s++) {
        const obs = env.observe(), f = this.focus();
        const controls = this.cars.map((car, i) => {
          if (car.kind === "human") {
            const steer = (keys.right ? 1 : 0) - (keys.left ? 1 : 0), throttle = (keys.up ? 1 : 0) - (keys.down ? 1 : 0);
            if (this.human.keys.length < 20000) this.human.keys.push((steer + 1) * 3 + (throttle + 1));
            return [steer, throttle];
          }
          const d = this.drivers[car.driver], { out, hidden } = d.net.forward(obs[i]);
          let a = 0;
          for (let k = 1; k < out.length; k++) if (out[k] > out[a]) a = k;
          if (i === f) {
            const r3 = (v) => Math.round(v * 1000) / 1000;
            brain = { driver: car.driver, obs: obs[i].map(r3), q: out.map(r3), action: a,
                      h1: d.brain.h1_idx.map((k) => r3(hidden[0][k])), h2: d.brain.h2_idx.map((k) => r3(hidden[1][k])) };
          }
          return C.ACTIONS[a];
        });
        const { crashed, stuck } = env.step(controls);
        this.cars.forEach((car, i) => {
          const bad = crashed[i] || (stuck[i] && car.kind === "ai");       // a human driver may stand still
          if (!bad) return;
          const c = env.cars[i];
          crashes.push({ car: i, x: c.x, y: c.y, stuck: stuck[i] && !crashed[i] });
          env.respawn(i);
        });
      }
      env.observe();
      const r = (v, k) => Math.round(v * 10 ** k) / 10 ** k;
      const tMax = Math.max(...env.cars.map((c) => c.steps));
      return {
        t: tMax * C.DT,
        cars: env.cars.map((c) => ({
          x: r(c.x, 1), y: r(c.y, 1), h: r(c.heading, 4), v: r(c.speed / C.MAX_SPEED, 3),
          rays: c.rays.map((x) => r(x / C.RAY_LENGTH, 3)), laps: c.laps, progress: r(c.progress / this.track.length, 4),
          lap_time: r((c.steps - c.lapStart) * C.DT, 2), last_lap: c.lastLap === null ? null : r(c.lastLap, 2),
          best_lap: c.bestLap === null ? null : r(c.bestLap, 2),
        })),
        crashes, brain,
      };
    }
  }

  // ---------------------------------------------------------------- the "server"
  let sim = null, info = null, presets = {}, drivers = {}, meta = [];

  async function init() {
    let error = null;
    try {
      const r = await fetch("tracks.json", { cache: "no-cache" });
      for (const t of await r.json()) { presets[t.id] = new Track(t.center, t.width, t.name); presets[t.id].id = t.id; }
    } catch (e) { console.warn("No preset tracks", e); }
    try {
      const model = await root.MLP.load("model");
      if (model) {
        for (const spec of model.meta.networks) {
          drivers[spec.id] = { net: model.nets[spec.id], brain: spec.brain };
          meta.push({ name: spec.id, label: spec.label, best: spec.id === "best" });
        }
      }
      if (!drivers.best) error = "No trained driver has been exported to this page yet. To add it: train (python -m src.train), then run  python -m src.export_web  and publish again.";
    } catch (e) {
      error = `Could not load the drivers: ${e.message}`;
    }
    sim = new Sim(drivers);
    sim.setTrack(Object.values(presets)[0] || randomTrack());
    const brains = Object.fromEntries(Object.entries(drivers).map(([k, d]) => [k, { w1: d.brain.w1, w2: d.brain.w2, w3: d.brain.w3 }]));
    info = {
      error, drivers: meta, brains, presets: Object.entries(presets).map(([id, t]) => ({ id, name: t.name })),
      world: [C.WORLD_W, C.WORLD_H], car: [C.CAR_LENGTH, C.CAR_WIDTH], ray_angles: C.RAY_ANGLES.map((a) => Math.round(a * 1e4) / 1e4),
      ray_length: C.RAY_LENGTH, max_speed: C.MAX_SPEED, dt: C.DT, actions: C.ACTION_NAMES, colors: CAR_COLORS,
    };
  }

  const ROUTES = {
    "/api/info": () => ({ ...info, track: sim.track.toJson() }),
    // the human's race so far, for the leaderboard: the track, the number of cars, the human's place on
    // the start line and the keys of every step
    "/api/humanlog": () => ({ track: sim.track.id || null, n: sim.human?.n || 0, index: sim.human?.index ?? -1,
                              keys: (sim.human?.keys || []).join("") }),
    "/api/track": (d) => {
      let t;
      if (presets[d.preset]) t = presets[d.preset];
      else if (d.points && d.points.length) t = makeTrack(d.points.map((p) => [+p[0], +p[1]]), +(d.width || 95), "Drawing");
      else t = randomTrack();
      sim.setTrack(t);
      return { track: t.toJson() };
    },
    "/api/start": (d) => {
      const mode = d.mode || "watch", cars = [];
      if (mode === "stages") {
        meta.forEach((m, i) => cars.push({ kind: "ai", driver: m.name, label: m.label,
                                           color: m.best ? CAR_COLORS[0] : CAR_COLORS[(i + 2) % CAR_COLORS.length] }));
      } else {
        const nAi = drivers.best ? +(d.n_ai ?? 1) : 0;
        for (let i = 0; i < clamp(nAi, 0, 8); i++) cars.push({ kind: "ai", driver: "best", label: nAi > 1 ? `AI ${i + 1}` : "AI", color: CAR_COLORS[i % CAR_COLORS.length] });
        if (mode === "race" || mode === "solo") cars.push({ kind: "human", driver: null, label: "You", color: "#ffffff" });
      }
      sim.start(cars, mode === "watch");
      return { cars: cars.map(({ kind, label, color, driver }) => ({ kind, label, color, driver })) };
    },
    "/api/step": (d) => sim.step(clamp(d.n | 0, 0, 12), d.keys || {}),
  };

  root.Backend = {
    async request(url, body) {
      if (!sim) await init();
      const route = ROUTES[url];
      if (!route) throw new Error(`unknown address ${url}`);
      try {
        return route(body || {});
      } catch (e) {
        if (e instanceof TrackError) throw new Error(e.message);     // shown to the user, like the server's 400 error
        throw e;
      }
    },
  };
  root.CarSim = { ...root.CarSim, TrackError, makeTrack, randomTrack, resampleClosed, smoothClosed, curvatureRadius, selfIntersects, partsTooClose, fitToWorld, Sim };
})(typeof window !== "undefined" ? window : globalThis);
