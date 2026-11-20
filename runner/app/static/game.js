// =====================================================================================
//  The game rules - the JavaScript twin of src/game.py.
//  Whole numbers only, and the same random generator, so the browser and Python produce
//  exactly the same run from the same seed and the same actions. Keep the two files in sync.
// =====================================================================================
(function (root) {
  const C = {
    V0: 50, VMAX: 110, ACCEL_EVERY: 40,
    N_LANES: 3, LANE_TICKS: 4, JUMP_TICKS: 13, JUMP_HEIGHT: 120, SLIDE_TICKS: 12, HALF_LEN: 35,
    LOW: 1, HIGH: 2, TRAIN: 3, HURDLE_CLEAR: 50, BAR_LEN: 50,
    FIRST_ROW: 3000, GEN_AHEAD: 7000, MAX_OBSTACLES: 32,
    NOOP: 0, LEFT: 1, RIGHT: 2, JUMP: 3, SLIDE: 4,
    N_BINS: 32, BIN_CM: 150, TICK_MS: 50,
  };

  function xorshift32(x) {
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    return x;
  }

  function seedState(seed) {
    let x = (Math.imul(seed, 2654435761) + 12345) >>> 0;
    if (x === 0) x = 1;
    for (let i = 0; i < 4; i++) x = xorshift32(x);
    return x;
  }

  function jumpHeight(jt) {
    return Math.floor((4 * C.JUMP_HEIGHT * jt * (C.JUMP_TICKS - jt)) / (C.JUMP_TICKS * C.JUMP_TICKS));
  }

  class Game {
    constructor(seed, tOff = 0) { this.reset(seed, tOff); }

    reset(seed, tOff = 0) {
      this.seed = seed; this.t = 0; this.tOff = tOff; this.pz = 0;
      this.lane = 1; this.lcFrom = 1; this.lcTo = 1; this.lcT = 0; this.pending = 0;
      this.jt = 0; this.st = 0;
      this.alive = true; this.crashType = 0;
      this.obs = [];
      for (let k = 0; k < C.MAX_OBSTACLES; k++) this.obs.push({ lane: 0, z0: 0, z1: 0, typ: 0 });
      this.rng = seedState(seed);
      this.nextZ = C.FIRST_ROW;
      this.generate();
    }

    speed() { return Math.min(C.VMAX, C.V0 + Math.floor((this.t + this.tOff) / C.ACCEL_EVERY)); }
    rand(n) { this.rng = xorshift32(this.rng); return this.rng % n; }

    place(lane, z0, z1, typ) {
      for (const o of this.obs) {
        if (o.typ === 0) { o.lane = lane; o.z0 = z0; o.z1 = z1; o.typ = typ; return; }
      }
    }

    genRow() {
      const z = this.nextZ, v = this.speed();
      let r = this.rand(100);
      const n = r < 35 ? 1 : r < 80 ? 2 : 3;
      let lanes;
      if (n === 1) lanes = [this.rand(3)];
      else if (n === 2) { const free = this.rand(3); lanes = [0, 1, 2].filter((l) => l !== free); }
      else lanes = [0, 1, 2];
      const types = lanes.map(() => { const q = this.rand(100); return q < 35 ? C.LOW : q < 65 ? C.HIGH : C.TRAIN; });
      if (n === 3 && types.every((t) => t === C.TRAIN)) {
        const k = this.rand(3);
        types[k] = this.rand(2) === 0 ? C.LOW : C.HIGH;
      }
      let end = z;
      lanes.forEach((lane, i) => {
        const typ = types[i];
        const len = typ !== C.TRAIN ? C.BAR_LEN : 800 + this.rand(17) * 100;
        this.place(lane, z, z + len, typ);
        end = Math.max(end, z + len);
      });
      const gap = 18 + this.rand(16);
      this.nextZ = end + v * gap;
    }

    generate() {
      for (const o of this.obs) if (o.typ > 0 && o.z1 < this.pz - 1000) o.typ = 0;
      while (this.nextZ < this.pz + C.GEN_AHEAD) this.genRow();
    }

    effectiveLane() {
      if (this.lcT > C.LANE_TICKS / 2) return this.lcFrom;
      if (this.lcT > 0) return this.lcTo;
      return this.lane;
    }

    startChange(target) { this.lcFrom = this.lane; this.lcTo = target; this.lcT = C.LANE_TICKS; }

    // one tick; returns true if the runner crashed now
    step(a) {
      if (!this.alive) return false;
      // 1. the action
      const d = a === C.LEFT ? -1 : a === C.RIGHT ? 1 : 0;
      if (d !== 0) {
        if (this.lcT > 0) this.pending = d;
        else if (this.lane + d >= 0 && this.lane + d < C.N_LANES) this.startChange(this.lane + d);
      }
      if (a === C.JUMP && this.jt === 0) { this.jt = 1; this.st = 0; }
      if (a === C.SLIDE) { this.jt = 0; this.st = C.SLIDE_TICKS; }

      // 2. move forward
      const v = this.speed(), pzPrev = this.pz;
      this.pz += v; this.t += 1;

      // 3. collisions
      const lane = this.effectiveLane(), h = jumpHeight(this.jt);
      for (const o of this.obs) {
        if (o.typ === 0 || o.lane !== lane) continue;
        if (this.pz + C.HALF_LEN < o.z0 || pzPrev - C.HALF_LEN > o.z1) continue;
        const deadly = o.typ === C.TRAIN || (o.typ === C.LOW && h <= C.HURDLE_CLEAR) || (o.typ === C.HIGH && this.st === 0);
        if (deadly) { this.alive = false; this.crashType = o.typ; break; }
      }

      // 4. timers
      if (this.lcT > 0) {
        this.lcT -= 1;
        if (this.lcT === 0) {
          this.lane = this.lcTo;
          const p = this.pending; this.pending = 0;
          if (p !== 0 && this.lane + p >= 0 && this.lane + p < C.N_LANES) this.startChange(this.lane + p);
        }
      }
      if (this.jt > 0) { this.jt += 1; if (this.jt >= C.JUMP_TICKS) this.jt = 0; }
      this.st = Math.max(this.st - 1, 0);

      // 5. new obstacles (only while alive, like Python)
      if (this.alive && this.nextZ < this.pz + C.GEN_AHEAD) this.generate();
      return !this.alive;
    }

    // for drawing: the lane position as a smooth number (e.g. 1.5 = halfway between lanes 1 and 2)
    laneVisual(frac = 0) {
      if (this.lcT === 0) return this.lane;
      const done = (C.LANE_TICKS - this.lcT + frac) / C.LANE_TICKS;
      return this.lcFrom + (this.lcTo - this.lcFrom) * Math.min(1, done);
    }

    // what the network sees: grid[channel][bin][lane] (channels: hurdles, bars, trains)
    observeGrid() {
      const g = [0, 1, 2].map(() => Array.from({ length: C.N_BINS }, () => [0, 0, 0]));
      for (const o of this.obs) {
        if (o.typ === 0) continue;
        const r0 = o.z0 - this.pz, r1 = o.z1 - this.pz;
        for (let b = 0; b < C.N_BINS; b++) {
          const lo = b * C.BIN_CM;
          if (r0 < lo + C.BIN_CM && r1 >= lo) g[o.typ - 1][b][o.lane] = 1;
        }
      }
      return g;
    }

    distance() { return this.pz / 100; }
  }

  const api = { C, Game, xorshift32, seedState, jumpHeight };
  root.Runner = api;
  if (typeof module !== "undefined") module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
