// =====================================================================================
//  The car's physics: the track (src/track.py) and the cars (src/env.py) - the sensors, the driving,
//  laps and crashes. Used by the page (backend.js) and by the leaderboard server, which replays a race
//  from the recorded keys to check its time.
// =====================================================================================
(function (root) {
  // the same constants as src/config.py (keep them in sync)
  const C = {
    WORLD_W: 1200, WORLD_H: 800, TRACK_SPACING: 8.0, TRACK_WIDTH_RANGE: [80.0, 115.0],
    DT: 0.05, MAX_SPEED: 260.0, ACCEL: 220.0, BRAKE: 480.0, COAST_DRAG: 40.0, FRICTION: 0.15,
    TURN_RATE: 3.2, TURN_SPEED_FULL: 60.0, HIGH_SPEED_TURN_LOSS: 0.3, CAR_RADIUS: 7.0, CAR_LENGTH: 22.0, CAR_WIDTH: 11.0,
    ACTIONS: [[0, 1], [-1, 1], [1, 1], [-1, 0], [1, 0], [0, -1]],
    ACTION_NAMES: ["Gas", "Gas + left", "Gas + right", "Left", "Right", "Brake"],
    RAY_ANGLES: [-90, -60, -35, -15, 0, 15, 35, 60, 90].map((d) => (d * Math.PI) / 180),
    RAY_LENGTH: 260.0, STUCK_STEPS: 60,
  };
  const WINDOW = 14;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  // ---------------------------------------------------------------- deterministic math
  // Math.sin, Math.cos, Math.exp, Math.atan2 and Math.hypot may give a slightly different last digit in
  // different browsers (Chrome, Safari, Firefox). For a leaderboard the server replays every game and must get
  // exactly the same result, so the physics uses these versions instead: only + - * / and Math.sqrt, which
  // give the same answer everywhere. (They are as accurate as the built-in ones, to about 1e-16.)
  const DMath = (() => {
    const PIO2_HI = 1.57079632673412561417e+00, PIO2_LO = 6.07710050650619224932e-11, INV_PIO2 = 6.36619772367581382433e-01;
    const S1 = -1.66666666666666324348e-01, S2 = 8.33333333332248946124e-03, S3 = -1.98412698298579493134e-04,
      S4 = 2.75573137070700676789e-06, S5 = -2.50507602534068634195e-08, S6 = 1.58969099521155010221e-10;
    const C1 = 4.16666666666666019037e-02, C2 = -1.38888888888741095749e-03, C3 = 2.48015872894767294178e-05,
      C4 = -2.75573143513906633035e-07, C5 = 2.08757232129817482790e-09, C6 = -1.13596475577881948265e-11;
    const ksin = (r) => { const z = r * r; return r + r * z * (S1 + z * (S2 + z * (S3 + z * (S4 + z * (S5 + z * S6))))); };
    const kcos = (r) => { const z = r * r; return 1 - 0.5 * z + z * z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6))))); };
    function reduce(x) {
      const n = Math.round(x * INV_PIO2);
      return [(x - n * PIO2_HI) - n * PIO2_LO, ((n % 4) + 4) % 4];
    }
    function sin(x) {
      const [r, q] = reduce(x);
      return q === 0 ? ksin(r) : q === 1 ? kcos(r) : q === 2 ? -ksin(r) : -kcos(r);
    }
    function cos(x) {
      const [r, q] = reduce(x);
      return q === 0 ? kcos(r) : q === 1 ? -ksin(r) : q === 2 ? -kcos(r) : ksin(r);
    }
    const LN2_HI = 6.93147180369123816490e-01, LN2_LO = 1.90821492927058770002e-10, INV_LN2 = 1.44269504088896338700e+00;
    const P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03, P3 = 6.61375632143793436117e-05,
      P4 = -1.65339022054652515390e-06, P5 = 4.13813679705723846039e-08;
    function exp(x) {
      const k = Math.round(x * INV_LN2), r = (x - k * LN2_HI) - k * LN2_LO, z = r * r;
      const c = r - z * (P1 + z * (P2 + z * (P3 + z * (P4 + z * P5))));
      let y = 1 - ((r * c) / (c - 2) - r);
      for (let i = 0; i < k; i++) y *= 2;               // multiplying by 2 is exact
      for (let i = 0; i > k; i--) y /= 2;
      return y;
    }
    function atan(t) {                                   // t >= 0
      let inv = false;
      if (t > 1) { t = 1 / t; inv = true; }
      t = t / (1 + Math.sqrt(1 + t * t));                // atan(t) = 2 atan(t / (1 + sqrt(1 + t^2)))
      t = t / (1 + Math.sqrt(1 + t * t));                // twice: now t <= 0.199
      const z = t * t;
      let s = 0;
      for (let k = 23; k >= 0; k--) s = 1 / (2 * k + 1) - z * s;
      const a = 4 * t * s;
      return inv ? PIO2_HI + PIO2_LO - a : a;
    }
    function atan2(y, x) {
      if (x === 0 && y === 0) return 0;
      const PI = 2 * PIO2_HI + 2 * PIO2_LO;
      if (x === 0) return y > 0 ? PI / 2 : -PI / 2;
      const a = atan(Math.abs(y / x));
      if (x > 0) return y >= 0 ? a : -a;
      return y >= 0 ? PI - a : a - PI;
    }
    const hypot = (a, b, c = 0) => Math.sqrt(a * a + b * b + c * c);
    return { sin, cos, exp, atan2, hypot };
  })();

  class Track {
    constructor(center, width, name = "track") {
      this.center = center; this.width = width; this.name = name;
      const n = (this.n = center.length);
      this.segLen = center.map((c, i) => { const q = center[(i + 1) % n]; return DMath.hypot(q[0] - c[0], q[1] - c[1]); });
      this.cum = [0];
      for (const s of this.segLen) this.cum.push(this.cum[this.cum.length - 1] + s);
      this.length = this.cum[n];
      this.tangent = center.map((_, i) => {
        const a = center[(i - 1 + n) % n], b = center[(i + 1) % n], l = DMath.hypot(b[0] - a[0], b[1] - a[1]);
        return [(b[0] - a[0]) / l, (b[1] - a[1]) / l];
      });
      const half = width / 2;
      this.left = center.map((c, i) => [c[0] - this.tangent[i][1] * half, c[1] + this.tangent[i][0] * half]);
      this.right = center.map((c, i) => [c[0] + this.tangent[i][1] * half, c[1] - this.tangent[i][0] * half]);
      this.walls = [];                                             // all wall segments - for the sensors
      for (const side of [this.left, this.right]) side.forEach((a, i) => this.walls.push([a, side[(i + 1) % n]]));
    }
    reversed() { return new Track([...this.center].reverse(), this.width, this.name); }
    toJson() {
      const r = (pts) => pts.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
      return { id: this.id ?? null, name: this.name, width: this.width, length: Math.round(this.length), center: r(this.center), left: r(this.left), right: r(this.right) };
    }
  }

  function raycast(x, y, angle, walls) {
    const dx = DMath.cos(angle), dy = DMath.sin(angle);
    let best = C.RAY_LENGTH;
    for (const [a, b] of walls) {
      const ex = b[0] - a[0], ey = b[1] - a[1];
      const den = dx * ey - dy * ex;
      if (Math.abs(den) < 1e-9) continue;
      const px = a[0] - x, py = a[1] - y;
      const t = (px * ey - py * ex) / den, u = (px * dy - py * dx) / den;
      if (t > 0 && u >= 0 && u <= 1 && t < best) best = t;
    }
    return best;
  }

  class CarEnv {
    constructor(track, n) {
      this.track = track; this.n = n;
      const t = track, h0 = DMath.atan2(t.tangent[0][1], t.tangent[0][0]);
      this.cars = Array.from({ length: n }, () => ({
        x: t.center[0][0], y: t.center[0][1], heading: h0, speed: 0, idx: 0, progress: 0, best: 0,
        steps: 0, sinceBest: 0, laps: 0, lapStart: 0, lastLap: null, bestLap: null, rays: [],
      }));
    }

    observe() {
      return this.cars.map((c) => {
        c.rays = C.RAY_ANGLES.map((a) => raycast(c.x, c.y, c.heading + a, this.track.walls));
        return [...c.rays.map((r) => r / C.RAY_LENGTH), c.speed / C.MAX_SPEED];
      });
    }

    // the nearest point on the center line, searching only around the previous position
    locate(c) {
      const t = this.track;
      let best = Infinity, bi = c.idx, bu = 0;
      const n = t.n, x = c.x, y = c.y;
      let k = (((c.idx - WINDOW) % n) + n) % n;
      for (let o = -WINDOW; o <= WINDOW; o++, k = k + 1 === n ? 0 : k + 1) {
        const a = t.center[k], b = t.center[k + 1 === n ? 0 : k + 1];
        const abx = b[0] - a[0], aby = b[1] - a[1];
        let u = ((x - a[0]) * abx + (y - a[1]) * aby) / (abx * abx + aby * aby);
        u = u < 0 ? 0 : u > 1 ? 1 : u;
        const dx = x - (a[0] + abx * u), dy = y - (a[1] + aby * u);
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < best) { best = d; bi = k; bu = u; }
      }
      return [bi, t.cum[bi] + bu * t.segLen[bi], best];
    }

    // controls: [steer, throttle] for each car. Returns {crashed, stuck} arrays.
    step(controls) {
      const t = this.track, dt = C.DT, crashed = [], stuck = [];
      this.cars.forEach((c, i) => {
        const [steer, throttle] = controls[i];
        const sOld = c.s ?? this.locate(c)[1];          // where the last step left it (or find it)
        let v = c.speed;
        if (throttle > 0) v += C.ACCEL * dt;
        if (throttle < 0) v -= C.BRAKE * dt;
        if (throttle === 0) v -= C.COAST_DRAG * dt;
        v -= v * C.FRICTION * dt;
        v = clamp(v, 0, C.MAX_SPEED);
        const grip = Math.min(1.0, v / C.TURN_SPEED_FULL) * (1 - (C.HIGH_SPEED_TURN_LOSS * v) / C.MAX_SPEED);
        c.heading += steer * C.TURN_RATE * grip * dt;
        c.x += DMath.cos(c.heading) * v * dt;
        c.y += DMath.sin(c.heading) * v * dt;
        c.speed = v;
        const [newIdx, sNew, dist] = this.locate(c);
        let ds = sNew - sOld;
        if (ds > t.length / 2) ds -= t.length;                      // crossing the start line
        if (ds < -t.length / 2) ds += t.length;
        c.idx = newIdx; c.s = sNew; c.progress += ds; c.steps += 1;
        if (c.progress > c.best + 1e-6) { c.best = c.progress; c.sinceBest = 0; } else c.sinceBest += 1;
        const lapNow = Math.floor(c.progress / t.length);
        if (lapNow > c.laps) {
          const lapTime = (c.steps - c.lapStart) * dt;
          c.lastLap = lapTime;
          if (c.bestLap === null || lapTime < c.bestLap) c.bestLap = lapTime;
          c.lapStart = c.steps;
          c.laps = lapNow;
        }
        crashed.push(dist > t.width / 2 - C.CAR_RADIUS);
        stuck.push(c.sinceBest >= C.STUCK_STEPS);
      });
      return { crashed, stuck };
    }

    // a crashed car goes back to the middle of the track, where it was, at zero speed (for the app)
    respawn(i) {
      const c = this.cars[i], t = this.track;
      [c.x, c.y] = t.center[c.idx];
      c.heading = DMath.atan2(t.tangent[c.idx][1], t.tangent[c.idx][0]);
      c.speed = 0; c.best = c.progress; c.sinceBest = 0; c.s = undefined;
    }
  }


  // where car i of n starts in a race: all on the start line, side by side (as in app/sim.py)
  function startOffset(track, n, i) {
    const t = track, normal = [-t.tangent[0][1], t.tangent[0][0]], span = t.width / 2 - C.CAR_RADIUS - 6;
    const off = n > 1 ? -span + (2 * span * i) / (n - 1) : 0;
    return [off * normal[0], off * normal[1]];
  }

  root.CarSim = { C, Track, CarEnv, raycast, startOffset, DMath };
})(typeof window !== "undefined" ? window : globalThis);
