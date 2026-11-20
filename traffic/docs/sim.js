// =====================================================================================
//  The intersection simulation in JavaScript - a copy of src/traffic.py (cars with the IDM model,
//  the traffic light, the queues) and of app/world.py (two identical intersections with exactly the
//  same traffic: a regular light on the left, the AI on the right). Used by backend.js.
// =====================================================================================
(function (root) {
  // the same constants as src/config.py (keep them in sync)
  const C = {
    N_LANES: 4, PHASE_LANES: [[0, 1], [2, 3]], ROAD_LENGTH: 250.0, EXIT_AT: -40.0, VIEW_RANGE: 120.0,
    DT: 0.5, MAX_CARS: 64, V0: 14.0, T_HEADWAY: 1.3, A_MAX: 1.6, B_COMF: 2.5, S0: 2.0,
    VEHICLES: [["car", 4.5, 0.85, 1.0], ["truck", 9.0, 0.10, 0.6], ["bus", 12.0, 0.05, 0.6]],
    WAIT_SPEED: 1.0, DECISION_EVERY: 5.0, YELLOW: 3.0, ALL_RED: 1.0, FIXED_GREEN: 30.0, COUNT_SCALE: 20.0,
    LANES: ["N", "S", "E", "W"], LANE_NAMES: { N: "North", S: "South", E: "East", W: "West" },
    CONDITIONS: ["day", "dusk", "night", "rain"], CONDITION_NAMES: { day: "Day", dusk: "Dusk", night: "Night", rain: "Rain" },
  };
  const GREEN = 0, YELLOW = 1, ALL_RED = 2;
  const LEN = C.VEHICLES.map((v) => v[1]), PROB = C.VEHICLES.map((v) => v[2]), ACC_F = C.VEHICLES.map((v) => v[3]);
  const SPD = Math.round(C.DECISION_EVERY / C.DT);           // simulation steps per decision

  // a small random generator with a seed (mulberry32)
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const choice = (r, p) => { let u = r(), k = 0; while (k < p.length - 1 && u >= p[k]) { u -= p[k]; k++; } return k; };

  // ---------------------------------------------------------------- one intersection (src/traffic.py, one environment)
  class Intersection {
    constructor(phase) {
      this.lanes = [[], [], [], []];          // cars sorted by s (distance to the stop line): [0] is the leader
      this.t = 0; this.phase = phase; this.mode = GREEN; this.modeTimer = 0; this.greenTime = 0;
      this.backlog = [0, 0, 0, 0]; this.totalWait = 0; this.arrived = 0; this.served = 0; this.switches = 0;
      this.nextId = 0;
    }

    clone() {
      const o = Object.assign(Object.create(Intersection.prototype), this);
      o.lanes = this.lanes.map((l) => l.map((c) => ({ ...c })));
      o.backlog = this.backlog.slice();
      return o;
    }

    isGreen(l) { return this.mode === GREEN && C.PHASE_LANES[this.phase].includes(l); }

    laneLight() {                               // 0 green, 1 yellow, 2 red - for drawing
      return [0, 1, 2, 3].map((l) => (!C.PHASE_LANES[this.phase].includes(l) ? 2 : this.mode === GREEN ? 0 : this.mode === YELLOW ? 1 : 2));
    }

    switchLight() {
      if (this.mode !== GREEN) return;
      this.mode = YELLOW; this.modeTimer = C.YELLOW; this.switches += 1;
      // cars too close to the line continue through the yellow (they can't stop comfortably)
      for (const l of C.PHASE_LANES[this.phase]) {
        for (const c of this.lanes[l]) if (c.s > 0 && c.s < (c.v * c.v) / (2 * C.B_COMF * 1.4) + 1.0) c.committed = true;
      }
    }

    updateLights() {
      const dt = C.DT;
      if (this.mode === GREEN) { this.greenTime += dt; return; }
      this.modeTimer -= dt;
      if (this.mode === YELLOW) {
        if (this.modeTimer <= 0) { this.mode = ALL_RED; this.modeTimer = C.ALL_RED; }
      } else if (this.modeTimer <= 0) {
        this.mode = GREEN; this.phase = 1 - this.phase; this.greenTime = 0;
        for (const lane of this.lanes) for (const c of lane) c.committed = false;
      }
    }

    // one step of 0.5 s. spawn = [arrive (4 booleans), kinds (4 vehicle types)]
    simStep(spawn) {
      const dt = C.DT;
      this.updateLights();
      let nWait = 0;
      for (let l = 0; l < C.N_LANES; l++) {
        const cars = this.lanes[l], go = this.isGreen(l);
        const old = cars.map((c) => ({ s: c.s, v: c.v, len: LEN[c.kind] }));
        cars.forEach((c, i) => {
          const lead = i > 0 ? old[i - 1] : null;
          let gap = lead ? c.s - lead.s - lead.len : Infinity;
          let dv = c.v - (lead ? lead.v : 0);
          const mustStop = !go && c.s > 0 && !c.committed;           // the light as a standing obstacle at the stop line
          if (mustStop && c.s - 0.5 < gap) { gap = c.s - 0.5; dv = c.v; }
          const a = C.A_MAX * ACC_F[c.kind];                         // the IDM model
          const sStar = C.S0 + Math.max(0, c.v * C.T_HEADWAY + (c.v * dv) / (2 * Math.sqrt(a * C.B_COMF)));
          const inter = Number.isFinite(gap) ? (sStar / Math.max(gap, 0.1)) ** 2 : 0;
          const acc = Math.max(-9.0, Math.min(a, a * (1 - (c.v / C.V0) ** 4 - inter)));
          let vNew = Math.max(0, c.v + acc * dt);
          let sNew = c.s - ((c.v + vNew) / 2) * dt;
          if (lead && sNew < lead.s + lead.len + 0.5) { sNew = lead.s + lead.len + 0.5; vNew = Math.min(vNew, lead.v); }   // don't pass the car in front
          if (mustStop && sNew < 0.3) { sNew = 0.3; vNew = 0; }                                                           // don't cross a red light
          c.s = sNew; c.v = vNew;
          if (c.v < C.WAIT_SPEED && c.s > 0) { c.wait += dt; nWait += 1; }
        });
        const before = cars.length;                                   // leaving the intersection
        this.lanes[l] = cars.filter((c) => c.s >= C.EXIT_AT);
        this.served += before - this.lanes[l].length;
        this.lanes[l].sort((p, q) => p.s - q.s);
      }
      nWait += this.backlog.reduce((x, y) => x + y, 0);
      this.totalWait += nWait * dt;

      const [arrive, kinds] = spawn;                                  // new cars entering
      for (let l = 0; l < C.N_LANES; l++) if (arrive[l]) { this.arrived += 1; this.backlog[l] += 1; }
      for (let l = 0; l < C.N_LANES; l++) {
        const cars = this.lanes[l], last = cars[cars.length - 1];
        const room = !last || C.ROAD_LENGTH - last.s > LEN[kinds[l]] + C.S0 + 6;
        if (this.backlog[l] > 0 && room && cars.length < C.MAX_CARS) {
          cars.push({ s: C.ROAD_LENGTH, v: last ? Math.min(C.V0, last.v + 2) : C.V0, kind: kinds[l], wait: 0, committed: false, cid: this.nextId++ });
          this.backlog[l] -= 1;
        }
      }
      this.t += dt;
      return nWait;
    }

    // the "true" count: vehicles within the camera range in each lane
    counts() { return this.lanes.map((cars) => cars.filter((c) => c.s >= 0 && c.s <= C.VIEW_RANGE).length); }

    // the agent's observation (6 numbers), ordered relative to the current phase
    observe(c) {
      const g = this.phase === 0 ? [c[0], c[1]] : [c[2], c[3]], r = this.phase === 0 ? [c[2], c[3]] : [c[0], c[1]];
      return [g[0] / C.COUNT_SCALE, g[1] / C.COUNT_SCALE, r[0] / C.COUNT_SCALE, r[1] / C.COUNT_SCALE,
              this.mode !== GREEN ? 1 : 0, Math.min(this.greenTime / 60.0, 1.0)];
    }

    waiting() {
      let n = this.backlog.reduce((x, y) => x + y, 0);
      for (const cars of this.lanes) for (const c of cars) if (c.v < C.WAIT_SPEED && c.s > 0) n++;
      return n;
    }
  }

  // the baseline lights (src/traffic.py)
  const fixedTime = (env) => (env.greenTime >= C.FIXED_GREEN - 1e-6 ? 1 : 0);
  function maxQueue(env, c, minGreen = 10.0, margin = 2) {
    const g = env.phase === 0 ? c[0] + c[1] : c[2] + c[3], r = env.phase === 0 ? c[2] + c[3] : c[0] + c[1];
    return env.greenTime >= minGreen && r > g + margin ? 1 : 0;
  }

  // ---------------------------------------------------------------- the two intersections (app/world.py)
  const PRESETS = {
    normal: ["Normal", [0.15, 0.14, 0.12, 0.11]],
    north: ["Heavy from the north", [0.30, 0.10, 0.08, 0.07]],
    rush: ["Rush hour", [0.22, 0.20, 0.18, 0.16]],
    quiet: ["Quiet night", [0.05, 0.03, 0.04, 0.03]],
    waves: ["Alternating waves", null],
    custom: ["Manual", null],
  };

  class World {
    // controller: obs -> {action, q} or null.  camera: async (vehicles, condition) -> {images, pred, density} or null
    constructor(controller, camera) {
      this.controller = controller; this.camera = camera;
      this.preset = "normal"; this.custom = [0.15, 0.15, 0.12, 0.12]; this.leftMode = "fixed";
      this.sensor = camera ? "camera" : "perfect"; this.condition = "day";
      this.reset();
    }

    rates() {
      const t = this.envs ? this.envs[0].t : 0;
      if (this.preset === "waves") {                         // every 4 minutes the busy direction changes
        const f = 0.5 + 0.5 * Math.sin((2 * Math.PI * t) / 480);
        return [0.06 + 0.24 * f, 0.05 + 0.2 * f, 0.06 + 0.24 * (1 - f), 0.05 + 0.2 * (1 - f)];
      }
      if (this.preset === "custom") return this.custom.slice();
      return PRESETS[this.preset][1].slice();
    }

    reset() {
      this.rand = rng(Math.floor(Math.random() * 2 ** 31));
      this.envs = null;
      const first = new Intersection(this.rand() < 0.5 ? 0 : 1);
      this.envs = [first];
      for (let i = 0; i < 40; i++) first.simStep(this.arrivals());   // start with some traffic
      Object.assign(first, { t: 0, totalWait: 0, arrived: 0, served: 0 });
      this.envs = [first, first.clone()];                    // the same starting state in both intersections
      this.steps = 0; this.history = []; this.cam = null; this.camSeq = 0; this.brain = null;
    }

    arrivals() {
      const rate = this.rates();
      const arrive = rate.map((r) => this.rand() < 1 - Math.exp(-r * C.DT));
      const kinds = rate.map(() => choice(this.rand, PROB));
      return [arrive, kinds];
    }

    vehicles(env) {                                          // for the cameras: [lane, s, kind, id]
      const out = [];
      env.lanes.forEach((cars, l) => cars.forEach((c) => out.push([l, c.s, c.kind, c.cid])));
      return out;
    }

    async decide() {
      const [left, right] = this.envs;
      if ((this.leftMode === "fixed" ? fixedTime(left) : maxQueue(left, left.counts())) === 1) left.switchLight();
      const trueCounts = right.counts();
      let cam = null;
      if (this.camera) {                                      // the camera wall works with either sensor, like the server
        cam = await this.camera(this.vehicles(right), this.condition);
        cam.true = trueCounts;
        cam.seq = ++this.camSeq;
        this.cam = cam;
      }
      const counts = this.sensor === "camera" && cam && cam.pred ? cam.pred : trueCounts;
      const obs = right.observe(counts);
      let a;
      if (this.controller) {
        const { action, q } = this.controller(obs);
        a = action;
        const r3 = (v) => Math.round(v * 1000) / 1000;
        this.brain = { obs: obs.map(r3), q: q.map(r3), action, counts: counts.map((v) => Math.round(v * 10) / 10) };
      } else {
        a = maxQueue(right, counts);
      }
      if (a === 1) right.switchLight();
    }

    async step(n) {
      for (let k = 0; k < n; k++) {
        if (this.steps % SPD === 0) await this.decide();
        const spawn = this.arrivals();
        for (const env of this.envs) env.simStep(spawn);        // exactly the same cars enter both intersections
        this.steps += 1;
        if (this.steps % 4 === 0) {
          this.history.push([Math.round(this.envs[0].t * 10) / 10, this.envs[0].waiting(), this.envs[1].waiting()]);
          if (this.history.length > 400) this.history.shift();
        }
      }
      return this.state();
    }

    side(env) {
      const cars = [];
      env.lanes.forEach((lane, l) => lane.forEach((c) => {
        if (c.s < 75) cars.push([l, Math.round(c.s * 100) / 100, c.kind, c.cid, c.v < C.WAIT_SPEED ? 1 : 0]);
      }));
      const queue = env.lanes.map((lane, l) => lane.filter((c) => c.v < C.WAIT_SPEED && c.s > 0).length + env.backlog[l]);
      return { cars, lights: env.laneLight(), waiting: env.waiting(),
               avg_wait: Math.round((env.totalWait / Math.max(env.arrived, 1)) * 100) / 100,
               served: env.served, queue, switches: env.switches };
    }

    state() {
      return { t: Math.round(this.envs[0].t * 10) / 10, rates: this.rates().map((r) => Math.round(r * 1000) / 1000),
               left: this.side(this.envs[0]), right: this.side(this.envs[1]),
               history: this.history.length ? this.history.slice(-1) : [], brain: this.brain, cam_seq: this.camSeq };
    }
  }

  root.TrafficSim = { C, LEN, PROB, SPD, Intersection, World, PRESETS, fixedTime, maxQueue, rng };
})(typeof window !== "undefined" ? window : globalThis);
