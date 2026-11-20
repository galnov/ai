// =====================================================================================
//  The maze and the ship's physics in JavaScript - a copy of src/maze.py + src/ship.py,
//  used when you fly yourself (the AI's flights come from the Python simulation).
//  Spaceship controls: the ship turns around its own axes ("right" = the right of the screen, always),
//  and nothing turns unless you press a key. Humans can also roll (Q / E).
// =====================================================================================
(function (root) {
  const DIRS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const V = {
    add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
    dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    len: (a) => Math.hypot(a[0], a[1], a[2]),
    norm: (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1e-9; return [a[0] / l, a[1] / l, a[2] / l]; },
    lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
  };

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
      f = V.add(V.mul(f, Math.cos(yaw)), V.mul(r, Math.sin(yaw)));                    // yaw: nose to the right
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      [f, u] = [V.add(V.mul(f, cp), V.mul(u, sp)), V.sub(V.mul(u, cp), V.mul(f, sp))];   // pitch: nose up
      if (roll) { r = V.cross(f, u); u = V.add(V.mul(u, Math.cos(roll)), V.mul(r, Math.sin(roll))); }
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
        const k = 1 - Math.exp(-P.accel_k * dt);
        this.vel = V.add(this.vel, V.mul(V.sub(V.mul(this.f, target), this.vel), k));
        const step = V.mul(this.vel, dt);
        this.pos = V.add(this.pos, step);
        this.dist += V.len(step);
        if (this.OFF.some((o) => this.m.isSolidPoint(V.add(this.pos, o)))) { this.crashed = true; break; }
      }
      this.t += P.dt;
      const c = this.m.cellOf(this.pos);
      if (!this.crashed && c[0] === this.m.exit[0] && c[1] === this.m.exit[1] && c[2] === this.m.exit[2]) this.reached = true;
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

  root.MazeSim = { MazeData, Ship, V };
})(window);
