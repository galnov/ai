// =====================================================================================
//  The "server" of the static version: everything the Flask server did, done in the browser.
//  The same URLs as app/server.py, answered here instead of over the network:
//     GET  /api/info    POST /api/mode, /api/config, /api/click, /api/shape, /api/clear_route, /api/step
//     GET  /api/value   (the Critic's value map)
//  It is a JavaScript copy of src/drone.py (the physics), src/pid.py (the PID controller) and
//  app/world.py (the four modes). The networks come from docs/model (python -m src.export_web)
//  and run in mlp.js.
// =====================================================================================
(function (root) {
  // the same constants as src/config.py (keep them in sync)
  const C = {
    WORLD_W: 20.0, WORLD_H: 12.0, G: 9.81,
    MASS: 1.0, ARM: 0.25, INERTIA: 0.02, DRAG: 0.35, ANG_DRAG: 0.02, MOTOR_TAU: 0.05,
    PHYS_DT: 0.01, CONTROL_DT: 0.04, GUST_TAU: 1.5, GUST_STD: 1.5, MAX_TILT: 1.4,
  };
  C.T_HOVER = (C.MASS * C.G) / 2;
  C.T_MAX = 2.0 * C.T_HOVER;
  C.SUBSTEPS = Math.round(C.CONTROL_DT / C.PHYS_DT);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  // normal random numbers (Box-Muller)
  function gauss() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }

  // ---------------------------------------------------------------- the drones (src/drone.py), app mode only
  class DroneEnv {
    constructor(n) {
      this.n = n;
      const z = () => new Float64Array(n);
      Object.assign(this, { x: z(), y: z(), vx: z(), vy: z(), th: z(), om: z(), wind: z(), gust: z(),
                            tx: z(), ty: z(), ix: z(), iy: z(), a1: z(), a2: z() });
      this.t1 = new Float64Array(n).fill(C.T_HOVER);
      this.t2 = new Float64Array(n).fill(C.T_HOVER);
      this.mass = new Float64Array(n).fill(C.MASS);
      this.gustStd = new Float64Array(n).fill(C.GUST_STD);
    }

    newTarget(i, tx, ty) { this.tx[i] = tx; this.ty[i] = ty; this.ix[i] = 0; this.iy[i] = 0; }

    // what the drone knows (11 numbers): see DroneEnv.observe in src/drone.py
    observe(i) {
      return [clamp(this.tx[i] - this.x[i], -8, 8) / 4, clamp(this.ty[i] - this.y[i], -8, 8) / 4,
              this.vx[i] / 4, this.vy[i] / 4, Math.sin(this.th[i]), Math.cos(this.th[i]), this.om[i] / 6,
              this.ix[i] / 4, this.iy[i] / 4, this.a1[i], this.a2[i]];
    }

    // one decision (0.04 s = 4 physics steps); act[i] = [left, right] in [-1, 1]. Returns which drones crashed.
    step(act) {
      const crashed = [];
      const dt = C.PHYS_DT;
      for (let i = 0; i < this.n; i++) {
        const a1 = clamp(act[i][0], -1, 1), a2 = clamp(act[i][1], -1, 1);
        const cmd1 = C.T_HOVER * (1 + a1), cmd2 = C.T_HOVER * (1 + a2);
        for (let s = 0; s < C.SUBSTEPS; s++) {
          const k = dt / C.MOTOR_TAU;                                    // the motors respond with a small delay
          this.t1[i] += (cmd1 - this.t1[i]) * k;
          this.t2[i] += (cmd2 - this.t2[i]) * k;
          this.gust[i] += (-this.gust[i] / C.GUST_TAU) * dt + this.gustStd[i] * Math.sqrt((2 * dt) / C.GUST_TAU) * gauss();
          const air = this.wind[i] + this.gust[i];
          const F = this.t1[i] + this.t2[i];
          const ax = (-F * Math.sin(this.th[i]) - C.DRAG * (this.vx[i] - air)) / this.mass[i];
          const ay = (F * Math.cos(this.th[i]) - C.DRAG * this.vy[i]) / this.mass[i] - C.G;
          const alpha = ((this.t2[i] - this.t1[i]) * C.ARM - C.ANG_DRAG * this.om[i]) / C.INERTIA;
          this.vx[i] += ax * dt; this.vy[i] += ay * dt; this.om[i] += alpha * dt;
          this.x[i] += this.vx[i] * dt; this.y[i] += this.vy[i] * dt; this.th[i] += this.om[i] * dt;
        }
        this.th[i] = ((((this.th[i] + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
        const ex = this.tx[i] - this.x[i], ey = this.ty[i] - this.y[i], leak = 1 - C.CONTROL_DT / 6.0;
        this.ix[i] = clamp(this.ix[i] * leak + ex * C.CONTROL_DT, -4, 4);
        this.iy[i] = clamp(this.iy[i] * leak + ey * C.CONTROL_DT, -4, 4);
        this.a1[i] = a1; this.a2[i] = a2;
        if (this.y[i] < 0.15 || Math.abs(this.th[i]) > C.MAX_TILT || this.x[i] < -1 || this.x[i] > C.WORLD_W + 1 ||
            this.y[i] > C.WORLD_H + 2) crashed.push(i);
      }
      return crashed;
    }
  }

  // ---------------------------------------------------------------- the PID controller (src/pid.py)
  const KP = 2.2, KI = 1.0, KD = 2.6, KP_ANG = 160.0, KD_ANG = 22.0;
  const MAX_ACC_X = 6.0, MAX_ACC_Y = 6.0, MAX_ACC_DOWN = 4.0, PID_TILT = 0.6, I_LIMIT = 5.0;
  class PID {
    constructor(n) { this.ix = new Float64Array(n); this.iy = new Float64Array(n); this.last = Array.from({ length: n }, () => [NaN, NaN]); }
    act(env, i) {
      if (env.tx[i] !== this.last[i][0] || env.ty[i] !== this.last[i][1]) { this.ix[i] = 0; this.iy[i] = 0; }   // new target
      this.last[i] = [env.tx[i], env.ty[i]];
      const ex = env.tx[i] - env.x[i], ey = env.ty[i] - env.y[i], dt = C.CONTROL_DT;
      this.ix[i] = clamp(this.ix[i] + ex * dt, -I_LIMIT / KI, I_LIMIT / KI);
      this.iy[i] = clamp(this.iy[i] + ey * dt, -I_LIMIT / KI, I_LIMIT / KI);
      const ax = clamp(KP * ex + KI * this.ix[i] - KD * env.vx[i], -MAX_ACC_X, MAX_ACC_X);
      const ay = clamp(KP * ey + KI * this.iy[i] - KD * env.vy[i], -MAX_ACC_DOWN, MAX_ACC_Y);
      const fx = C.MASS * ax, fy = C.MASS * (ay + C.G);
      const thDes = clamp(Math.atan2(-fx, Math.max(fy, 0.5)), -PID_TILT, PID_TILT);
      const F = Math.max(0, -fx * Math.sin(env.th[i]) + fy * Math.cos(env.th[i]));
      const tau = C.INERTIA * (KP_ANG * (thDes - env.th[i]) - KD_ANG * env.om[i]);
      const t1 = F / 2 - tau / (2 * C.ARM), t2 = F / 2 + tau / (2 * C.ARM);
      return [clamp(t1 / C.T_HOVER - 1, -1, 1), clamp(t2 / C.T_HOVER - 1, -1, 1)];
    }
  }

  // ---------------------------------------------------------------- the world (app/world.py)
  const BUILDINGS = [[1.2, 3.2, 3.0], [4.6, 1.8, 5.2], [8.4, 2.2, 2.4], [11.6, 1.6, 6.0], [14.2, 2.4, 3.6], [17.6, 1.4, 4.4]];
  const SWARM_N = 40;

  function shapePoints(name, n = SWARM_N) {
    const t = Array.from({ length: n }, (_, i) => (2 * Math.PI * i) / n);
    const cx = C.WORLD_W / 2, cy = C.WORLD_H / 2 + 0.5;
    if (name === "circle") return t.map((a) => [cx + 4.2 * Math.cos(a), cy + 3.6 * Math.sin(a)]);
    if (name === "heart") return t.map((a) => {
      const x = 16 * Math.sin(a) ** 3, y = 13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a);
      return [cx + x * 0.28, cy + y * 0.26 + 0.6];
    });
    if (name === "star") {
      const corners = Array.from({ length: 10 }, (_, i) => {
        const a = (2 * Math.PI * i) / 10, r = i % 2 === 0 ? 4.4 : 1.9;
        return [cx + r * Math.sin(a), cy + r * Math.cos(a)];
      });
      const per = Math.floor(n / 10), pts = [];
      for (let i = 0; i < 10; i++) {
        const a = corners[i], b = corners[(i + 1) % 10];
        for (let k = 0; k < per; k++) { const f = k / per; pts.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f]); }
      }
      return pts.slice(0, n);
    }
    if (name === "wave") return t.map((_, i) => { const x = 2.5 + ((C.WORLD_W - 5) * i) / (n - 1); return [x, cy + 2.6 * Math.sin(x / 2.0)]; });
    if (name === "grid") return t.map((_, i) => [cx + ((i % 10) - 4.5) * 1.3, cy + 2.4 - Math.floor(i / 10) * 1.6]);
    throw new Error(name);
  }

  // greedy assignment: each point to the drone closest to it (to avoid crossings)
  function assign(drones, points) {
    const d = drones.map((p) => points.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])));
    const out = new Array(points.length);
    for (let k = 0; k < points.length; k++) {
      let bi = 0, bj = 0, best = Infinity;
      for (let i = 0; i < d.length; i++) for (let j = 0; j < points.length; j++) if (d[i][j] < best) { best = d[i][j]; bi = i; bj = j; }
      out[bj] = bi;
      for (let j = 0; j < points.length; j++) d[bi][j] = Infinity;
      for (let i = 0; i < d.length; i++) d[i][bj] = Infinity;
    }
    const res = new Array(points.length);
    out.forEach((drone, j) => (res[drone] = points[j]));
    return res;
  }

  class World {
    constructor(policies, meta) {
      this.policies = policies; this.meta = meta;
      this.wind = 0; this.gust = 0; this.payload = 0; this.comparePid = true;
      this.setMode("free");
    }

    setMode(mode) {
      this.mode = mode; this.events = []; this.deliveries = 0;
      this.route = mode !== "delivery" ? [] : [[3.0, 8.0], [9.6, 4.6], [16.0, 9.0], [19.0, 7.8]];
      this.routeI = null;
      const best = this.policies.best ? "best" : null;
      let starts, target = null;
      if (mode === "free") {
        this.drones = [{ ctrl: best || "pid", label: "PPO network", color: "#34d1bf" }];
        if (this.comparePid) this.drones.push({ ctrl: "pid", label: "PID", color: "#ff8a5b" });
        starts = [[4.0, 3.0], [4.0, 4.4]].slice(0, this.drones.length);
        target = [14.0, 7.0];
      } else if (mode === "delivery") {
        this.drones = [{ ctrl: best || "pid", label: "Delivery", color: "#34d1bf" }];
        starts = [[0.6, 7.0]]; target = this.route[0]; this.routeI = 0;
      } else if (mode === "stages") {
        const pal = ["#8d99ae", "#c792ea", "#ffb020", "#ff6b6b", "#34d1bf"];
        this.drones = this.meta.map((m, i) => ({ ctrl: m.name, label: m.label, color: pal[i % pal.length] }));
        starts = this.drones.map((_, i) => [3.0, 3.0 + 1.3 * i]);
        target = [15.0, 7.0];
      } else {
        this.drones = Array.from({ length: SWARM_N }, () => ({ ctrl: best || "pid", label: "", color: "#34d1bf" }));
        starts = Array.from({ length: SWARM_N }, (_, i) => [1.0 + ((C.WORLD_W - 2.0) * i) / (SWARM_N - 1), 1.2]);
      }
      const n = this.drones.length;
      this.env = new DroneEnv(n);
      this.starts = starts;
      starts.forEach(([x, y], i) => { this.env.x[i] = x; this.env.y[i] = y; });
      this.pid = new PID(n);
      this.applyConditions();
      if (mode === "swarm") {
        this.shape = "heart";
        assign(starts, shapePoints(this.shape)).forEach(([x, y], i) => this.env.newTarget(i, x, y));
      } else {
        for (let i = 0; i < n; i++) this.env.newTarget(i, target[0], target[1]);
      }
      this.trails = Array.from({ length: n }, () => []);
    }

    applyConditions() {
      const e = this.env;
      for (let i = 0; i < e.n; i++) { e.wind[i] = this.wind; e.gustStd[i] = this.gust; e.mass[i] = C.MASS + this.payload; }
    }

    configure({ wind, gust, payload, compare_pid }) {
      if (wind != null) this.wind = clamp(+wind, -7, 7);
      if (gust != null) this.gust = clamp(+gust, 0, 3);
      if (payload != null) this.payload = clamp(+payload, 0, 0.6);
      this.applyConditions();
      if (compare_pid != null && !!compare_pid !== this.comparePid) {
        this.comparePid = !!compare_pid;
        if (this.mode === "free") this.setMode("free");
      }
    }

    click(x, y) {
      x = clamp(x, 0.8, C.WORLD_W - 0.8); y = clamp(y, 1.5, C.WORLD_H - 0.8);
      const n = this.env.n;
      if (this.mode === "free" || this.mode === "stages") for (let i = 0; i < n; i++) this.env.newTarget(i, x, y);
      else if (this.mode === "delivery") {
        if (this.route.length >= 8) this.route = [];
        this.route.push([x, y]);
        if (this.route.length === 1) { this.routeI = 0; for (let i = 0; i < n; i++) this.env.newTarget(i, x, y); }
      }
    }

    setShape(name) {
      if (this.mode !== "swarm") return;
      this.shape = name;
      const pos = Array.from({ length: this.env.n }, (_, i) => [this.env.x[i], this.env.y[i]]);
      assign(pos, shapePoints(name)).forEach(([x, y], i) => this.env.newTarget(i, x, y));
    }

    actions() {
      const e = this.env;
      return this.drones.map((d, i) => (d.ctrl === "pid" ? this.pid.act(e, i) : this.policies[d.ctrl](e.observe(i))));
    }

    crashedIntoBuilding(i) {
      if (this.mode !== "delivery") return false;
      const e = this.env;
      return BUILDINGS.some(([bx, bw, bh]) => e.x[i] > bx - 0.25 && e.x[i] < bx + bw + 0.25 && e.y[i] < bh + 0.1);
    }

    step(nSteps) {
      const e = this.env;
      for (let s = 0; s < nSteps; s++) {
        const crashed = new Set(e.step(this.actions()));
        for (let i = 0; i < e.n; i++) if (this.crashedIntoBuilding(i)) crashed.add(i);
        for (const i of crashed) { this.events.push({ type: "crash", x: e.x[i], y: e.y[i] }); this.respawn(i); }
        if (this.mode === "delivery" && this.route.length) {
          if (Math.hypot(e.tx[0] - e.x[0], e.ty[0] - e.y[0]) < 0.6) {
            this.routeI = (this.routeI + 1) % this.route.length;
            if (this.routeI === 0) { this.deliveries += 1; this.events.push({ type: "delivered", x: e.x[0], y: e.y[0] }); }
            e.newTarget(0, ...this.route[this.routeI]);
          }
        }
        for (let i = 0; i < e.n; i++) {
          const tr = this.trails[i];
          tr.push([Math.round(e.x[i] * 100) / 100, Math.round(e.y[i] * 100) / 100]);
          if (tr.length > 90) tr.shift();
        }
      }
      return this.state();
    }

    respawn(i) {
      const e = this.env;
      [e.x[i], e.y[i]] = this.starts[i];
      e.vx[i] = e.vy[i] = e.th[i] = e.om[i] = 0;
      e.t1[i] = e.t2[i] = C.T_HOVER;
      e.ix[i] = e.iy[i] = 0;
      this.pid.ix[i] = this.pid.iy[i] = 0;
      this.trails[i] = [];
      if (this.mode === "delivery" && this.route.length) { this.routeI = 0; e.newTarget(i, ...this.route[0]); }
    }

    state() {
      const e = this.env, r = (a) => Array.from(a, (v) => Math.round(v * 1000) / 1000);
      const events = this.events; this.events = [];
      let gust = 0;
      for (let i = 0; i < e.n; i++) gust += e.gust[i] / e.n;
      return {
        mode: this.mode,
        drones: { x: r(e.x), y: r(e.y), th: r(e.th), t1: r(e.t1.map((v) => v / C.T_MAX)), t2: r(e.t2.map((v) => v / C.T_MAX)),
                  tx: r(e.tx), ty: r(e.ty) },
        meta: this.drones.map((d) => ({ label: d.label, color: d.color, pid: d.ctrl === "pid" })),
        trails: this.mode !== "swarm" ? this.trails.map((t) => t.slice()) : [],
        air: e.n ? this.wind + gust : this.wind,
        events, route: this.route, route_i: this.routeI, deliveries: this.deliveries,
        buildings: this.mode === "delivery" ? BUILDINGS : [],
      };
    }

    // the Critic's value map V(s): how "good" it is to be at each point (standing still and level)
    valueMap(valueFn, nx = 40, ny = 24) {
      const tx = this.env.tx[0], ty = this.env.ty[0], v = [];
      for (let r = 0; r < ny; r++) {
        const row = [];
        for (let c = 0; c < nx; c++) {
          const x = 0.25 + ((C.WORLD_W - 0.5) * c) / (nx - 1), y = 0.25 + ((C.WORLD_H - 0.5) * r) / (ny - 1);
          row.push(valueFn([clamp(tx - x, -8, 8) / 4, clamp(ty - y, -8, 8) / 4, 0, 0, 0, 1, 0, 0, 0, 0, 0]));
        }
        v.push(row);
      }
      const flat = v.flat(), lo = Math.min(...flat), hi = Math.max(...flat);
      return { nx, ny, v: v.map((row) => row.map((x) => Math.round(((x - lo) / (hi - lo + 1e-9)) * 1000) / 1000)) };
    }
  }

  // ---------------------------------------------------------------- the "server"
  let world = null, info = null, valueFn = null;

  async function init() {
    const policies = {}, meta = [];
    let error = null;
    try {
      const model = await root.MLP.load("model");
      if (model) {
        for (const spec of model.meta.networks) {
          if (spec.role !== "actor") continue;
          const net = model.nets[spec.id];
          policies[spec.id] = (obs) => net.forward(obs).out;
          meta.push({ name: spec.id, label: spec.label });
        }
        const critic = model.nets["best:critic"];
        if (critic) valueFn = (obs) => critic.forward(obs).out[0];
      }
      if (!policies.best) error = "No trained network has been exported to this page yet, so the drone is flown by a PID controller. To add it: train, then run  python -m src.export_web  and publish again.";
    } catch (e) {
      error = `Could not load the network: ${e.message}`;
    }
    meta.sort((a, b) => (a.name === "best") - (b.name === "best"));       // the trained network last, like the server
    world = new World(policies, meta);
    info = { error, has_model: !!policies.best, stages: meta.length > 1, has_value: !!valueFn,
             world: [C.WORLD_W, C.WORLD_H], dt: C.CONTROL_DT, arm: C.ARM };
  }

  const ROUTES = {
    "/api/info": () => info,
    "/api/mode": (d) => { world.setMode(d.mode || "free"); return world.state(); },
    "/api/config": (d) => { world.configure(d); return { ok: true }; },
    "/api/click": (d) => { world.click(+(d.x ?? 10), +(d.y ?? 6)); return { ok: true }; },
    "/api/shape": (d) => { world.setShape(d.name || "circle"); return { ok: true }; },
    "/api/clear_route": () => { world.route = []; return { ok: true }; },
    "/api/step": (d) => world.step(clamp(d.n | 0, 0, 25)),
    "/api/value": () => (valueFn ? world.valueMap(valueFn) : { error: "No trained network" }),
  };

  root.Backend = {
    async request(url, body) {
      if (!world) await init();
      const route = ROUTES[url];
      if (!route) throw new Error(`unknown address ${url}`);
      return route(body || {});
    },
  };
  root.DroneSim = { C, DroneEnv, PID, World, shapePoints, assign };
})(typeof window !== "undefined" ? window : globalThis);
