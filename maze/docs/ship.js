// =====================================================================================
//  The maze and the ship's physics in JavaScript - a copy of src/maze.py + src/ship.py,
//  used when you fly yourself, and (in this web version) for the AI's flights too (backend.js).
//  Spaceship controls: the ship turns around its own axes ("right" = the right of the screen, always),
//  and nothing turns unless you press a key. Humans can also roll (Q / E).
// =====================================================================================
(function (root) {
  const DIRS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
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
  const V = {
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
    dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    len: (a) => DMath.hypot(a[0], a[1], a[2]),
    norm: (a) => { const l = DMath.hypot(a[0], a[1], a[2]) || 1e-9; return [a[0] / l, a[1] / l, a[2] / l]; },
    lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
  };

  // a small random generator with a seed (mulberry32), so the same seed gives the same maze
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

  // ---------------------------------------------------------------- the maze (src/maze.py)
  function makeMaze(seed, rooms, cell = 4.0) {
    const G = 2 * rooms + 1, n = G * G * G, rand = rng(seed);
    const idx = (x, y, z) => (x * G + y) * G + z;
    const solid = new Uint8Array(n).fill(1);
    const visited = new Uint8Array(rooms * rooms * rooms);
    const ridx = (r) => (r[0] * rooms + r[1]) * rooms + r[2];
    const stack = [[0, 0, 0]];
    visited[0] = 1;
    solid[idx(1, 1, 1)] = 0;
    while (stack.length) {                                   // depth-first search ("recursive backtracker")
      const room = stack[stack.length - 1];
      const order = [0, 1, 2, 3, 4, 5];
      for (let i = 5; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      let pick = null;
      for (const k of order) {
        const d = DIRS[k], nb = [room[0] + d[0], room[1] + d[1], room[2] + d[2]];
        if (nb.every((c) => c >= 0 && c < rooms) && !visited[ridx(nb)]) { pick = [nb, d]; break; }
      }
      if (!pick) { stack.pop(); continue; }
      const [nb, d] = pick;
      visited[ridx(nb)] = 1;
      const a = room.map((c) => 2 * c + 1);
      solid[idx(a[0] + d[0], a[1] + d[1], a[2] + d[2])] = 0;                 // the passage
      solid[idx(a[0] + 2 * d[0], a[1] + 2 * d[1], a[2] + 2 * d[2])] = 0;     // the next room
      stack.push(nb);
    }
    const bfs = (src) => {
      const dist = new Int32Array(n).fill(-1);
      dist[idx(...src)] = 0;
      const q = [src];
      for (let h = 0; h < q.length; h++) {
        const c = q[h];
        for (const d of DIRS) {
          const nb = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
          if (nb.some((v) => v < 0 || v >= G)) continue;
          const i = idx(...nb);
          if (!solid[i] && dist[i] < 0) { dist[i] = dist[idx(...c)] + 1; q.push(nb); }
        }
      }
      return dist;
    };
    const start = [1, 1, 1];
    const dStart = bfs(start);
    let exit = start, best = -1;                              // the exit: the room farthest from the start
    for (let x = 1; x < G; x += 2) for (let y = 1; y < G; y += 2) for (let z = 1; z < G; z += 2) {
      const v = dStart[idx(x, y, z)];
      if (v > best) { best = v; exit = [x, y, z]; }
    }
    const dist = bfs(exit);
    const next = [];                                          // the direction to the exit, for every empty cell
    for (let x = 0; x < G; x++) for (let y = 0; y < G; y++) for (let z = 0; z < G; z++) {
      const i = idx(x, y, z);
      if (solid[i]) continue;
      let dir = [0, 0, 0];
      if (dist[i] > 0) {
        for (const d of DIRS) {
          const nb = [x + d[0], y + d[1], z + d[2]];
          if (nb.some((v) => v < 0 || v >= G)) continue;
          if (dist[idx(...nb)] === dist[i] - 1) { dir = d.slice(); break; }
        }
      }
      next.push(dir);
    }
    const nextOf = (c) => { let k = 0; for (let i = 0; i < idx(...c); i++) if (!solid[i]) k++; return next[k]; };
    const path = [start];
    while (path[path.length - 1].join() !== exit.join()) {
      const c = path[path.length - 1], d = nextOf(c);
      path.push([c[0] + d[0], c[1] + d[1], c[2] + d[2]]);
    }
    return { seed, rooms, G, cell, solid: Array.from(solid).join(""), start, exit, path, next };
  }

  class MazeData {
    constructor(j) {
      this.seed = j.seed; this.rooms = j.rooms; this.G = j.G; this.cell = j.cell;
      this.start = j.start; this.exit = j.exit; this.path = j.path;
      const G = this.G, n = G * G * G;
      this.solid = new Uint8Array(n);
      for (let i = 0; i < n; i++) this.solid[i] = j.solid.charCodeAt(i) === 49;   // "1"
      this.next = new Int8Array(n * 3);
      let k = 0;
      for (let i = 0; i < n; i++) if (!this.solid[i]) { this.next.set(j.next[k], i * 3); k++; }
      // distance to the exit (BFS), for the progress display
      this.dist = new Int32Array(n).fill(-1);
      const q = [this.exit];
      this.dist[this.idx(...this.exit)] = 0;
      while (q.length) {
        const c = q.shift(), dc = this.dist[this.idx(...c)];
        for (const d of DIRS) {
          const nb = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
          if (this.isSolidCell(nb)) continue;
          const ni = this.idx(...nb);
          if (this.dist[ni] < 0) { this.dist[ni] = dc + 1; q.push(nb); }
        }
      }
    }
    idx(x, y, z) { return (x * this.G + y) * this.G + z; }
    isSolidCell(c) {
      const G = this.G;
      if (c[0] < 0 || c[1] < 0 || c[2] < 0 || c[0] >= G || c[1] >= G || c[2] >= G) return true;
      return this.solid[this.idx(c[0], c[1], c[2])] === 1;
    }
    cellOf(p) { return [Math.floor(p[0] / this.cell), Math.floor(p[1] / this.cell), Math.floor(p[2] / this.cell)]; }
    isSolidPoint(p) { return this.isSolidCell(this.cellOf(p)); }
    center(c) { return [(c[0] + 0.5) * this.cell, (c[1] + 0.5) * this.cell, (c[2] + 0.5) * this.cell]; }
    nextDir(c) { if (this.isSolidCell(c)) return [0, 0, 0]; const i = this.idx(...c) * 3; return [this.next[i], this.next[i + 1], this.next[i + 2]]; }
    pathLength() { return (this.path.length - 1) * this.cell; }
  }

  // the 26 points checked against the walls, and the 81 camera rays (same as ship.py)
  function offsets(radius) {
    const out = [];
    for (const i of [-1, 0, 1]) for (const j of [-1, 0, 1]) for (const k of [-1, 0, 1]) {
      if (!i && !j && !k) continue;
      out.push(V.mul(V.norm([i, j, k]), radius));
    }
    return out;
  }
  function rayDirs(P) {
    const t = Math.tan((P.fov / 2) * Math.PI / 180), out = [];
    for (let yi = 0; yi < P.rays_y; yi++) for (let xi = 0; xi < P.rays_x; xi++) {
      const x = -t + (2 * t * xi) / (P.rays_x - 1), y = t - (2 * t * yi) / (P.rays_y - 1);
      out.push(V.norm([x, y, 1]));
    }
    return out;
  }

  class Ship {
    constructor(maze, P) {
      this.m = maze; this.P = P;
      this.OFF = offsets(P.radius); this.RAYS = rayDirs(P);
      const d = maze.nextDir(maze.start).map(Number);
      this.f = V.norm(d);
      this.u = Math.abs(this.f[1]) < 0.5 ? [0, 1, 0] : [1, 0, 0];
      this.pos = maze.center(maze.start);
      this.vel = V.mul(d.map(Number), P.v_min + 1);
      this.t = 0; this.crashed = false; this.reached = false; this.dist = 0;
      this.L0 = Math.max(this.remaining(), 1);
    }
    right() { return V.cross(this.f, this.u); }

    // turn around the ship's own axes (same as ship.py), plus roll for human players
    rotate(pitch, yaw, roll) {
      let f = this.f, u = this.u, r = V.cross(f, u);
      f = V.add(V.mul(f, DMath.cos(yaw)), V.mul(r, DMath.sin(yaw)));                    // yaw: nose to the right
      const cp = DMath.cos(pitch), sp = DMath.sin(pitch);
      [f, u] = [V.add(V.mul(f, cp), V.mul(u, sp)), V.sub(V.mul(u, cp), V.mul(f, sp))];   // pitch: nose up
      if (roll) { r = V.cross(f, u); u = V.add(V.mul(u, DMath.cos(roll)), V.mul(r, DMath.sin(roll))); }
      f = V.norm(f);
      u = V.norm(V.sub(u, V.mul(f, V.dot(u, f))));
      this.f = f; this.u = u;
    }

    // one decision step (0.1 s); a = [pitch, yaw, throttle] (+ roll for humans), each in [-1, 1]
    step(a) {
      if (this.crashed || this.reached) return;
      const P = this.P, dt = P.dt / P.substeps;
      const target = P.v_min + ((a[2] + 1) / 2) * (P.v_max - P.v_min);
      for (let s = 0; s < P.substeps; s++) {
        this.rotate(a[0] * P.pitch_rate * dt, a[1] * P.yaw_rate * dt, (a[3] || 0) * P.yaw_rate * dt);
        const k = 1 - DMath.exp(-P.accel_k * dt);
        this.vel = V.add(this.vel, V.mul(V.sub(V.mul(this.f, target), this.vel), k));
        const step = V.mul(this.vel, dt);
        this.pos = V.add(this.pos, step);
        this.dist += V.len(step);
        if (this.touchesWall()) { this.crashed = true; break; }
      }
      this.t += P.dt;
      const c = this.m.cellOf(this.pos);
      if (!this.crashed && c[0] === this.m.exit[0] && c[1] === this.m.exit[1] && c[2] === this.m.exit[2]) this.reached = true;
    }

    // is any of the 26 points around the ship inside a wall? (the same as OFF.some(isSolidPoint), without
    // making new arrays - the leaderboard server replays whole flights, so this has to be fast)
    touchesWall() {
      const m = this.m, cell = m.cell, G = m.G, p = this.pos, OFF = this.OFF;
      for (let i = 0; i < OFF.length; i++) {
        const o = OFF[i];
        const cx = Math.floor((p[0] + o[0]) / cell), cy = Math.floor((p[1] + o[1]) / cell), cz = Math.floor((p[2] + o[2]) / cell);
        if (cx < 0 || cy < 0 || cz < 0 || cx >= G || cy >= G || cz >= G) return true;
        if (m.solid[(cx * G + cy) * G + cz] === 1) return true;
      }
      return false;
    }

    remaining() {
      const c = this.m.cellOf(this.pos);
      if (this.m.isSolidCell(c)) return this.lastL ?? 0;
      const d = this.m.dist[this.m.idx(...c)], nd = this.m.nextDir(c);
      this.lastL = d * this.m.cell - V.dot(V.sub(this.pos, this.m.center(c)), nd);
      return this.lastL;
    }
    progress() { return Math.max(0, Math.min(1, 1 - this.remaining() / this.L0)); }

    // the depth camera: distances along the 81 rays (DDA through the grid), divided by the range
    castRays() {
      const r = this.right(), out = [], cell = this.m.cell, maxT = this.P.ray_range / cell;
      for (const d0 of this.RAYS) {
        const D = V.add(V.add(V.mul(r, d0[0]), V.mul(this.u, d0[1])), V.mul(this.f, d0[2]));
        const Pp = V.mul(this.pos, 1 / cell);
        const c = Pp.map(Math.floor), step = D.map((x) => (x > 0 ? 1 : -1));
        const tMax = [0, 1, 2].map((i) => (Math.abs(D[i]) > 1e-12 ? ((c[i] + (step[i] > 0 ? 1 : 0)) - Pp[i]) / D[i] : Infinity));
        const tDelta = D.map((x) => (Math.abs(x) > 1e-12 ? Math.abs(1 / x) : Infinity));
        let hit = maxT;
        for (let it = 0; it < 30; it++) {
          const ax = tMax[0] <= tMax[1] ? (tMax[0] <= tMax[2] ? 0 : 2) : (tMax[1] <= tMax[2] ? 1 : 2);
          const t = tMax[ax];
          if (t >= maxT) break;
          c[ax] += step[ax]; tMax[ax] += tDelta[ax];
          if (this.m.isSolidCell(c)) { hit = t; break; }
        }
        out.push(Math.min(hit * cell, this.P.ray_range) / this.P.ray_range);
      }
      return out;
    }
  }

  root.MazeSim = { MazeData, Ship, V, makeMaze, rng };
})(typeof window !== "undefined" ? window : globalThis);
