// =====================================================================================
//  The air hockey table - a JavaScript copy of src/physics.py and src/bot.py (one table instead of many).
//  Keep the numbers in sync with src/config.py.
//
//  Units: meters, seconds. Seen from above: x from -0.5 to 0.5, y from -1 (player 0's goal) to +1 (player 1's).
//  Player 0 is at the bottom, player 1 at the top. Each player sees the table from its own side
//  (player 1's view is turned 180 degrees), so one network can play either side.
// =====================================================================================
(function (root) {
  const C = {
    HALF_W: 0.5, HALF_L: 1.0, GOAL_HALF: 0.15, PUCK_R: 0.035, PADDLE_R: 0.05,
    DT: 1 / 30, SUBSTEPS: 8, PADDLE_VMAX: 2.6, PADDLE_ACCEL: 30.0, PUCK_VMAX: 6.0, PUCK_FRICTION: 0.25,
    WALL_E: 0.88, PADDLE_E: 0.85, SERVE_SPEED: 0.25, POINT_TIME: 12.0, MATCH_GOALS: 7,
    OBS_SIZE: 12, REACTION_STEPS: 2,
    GOAL_REWARD: 10.0, PUSH_REWARD: 0.6, TOUCH_REWARD: 0.05, TIMEOUT_REWARD: -3.0,
  };
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

  class Table {
    constructor(rand = Math.random) {
      this.rand = rand;
      this.pp = [[0, -0.8], [0, 0.8]];  // paddle positions [player][x, y]
      this.pv = [[0, 0], [0, 0]];       // paddle velocities
      this.kp = [0, 0];                 // puck position
      this.kv = [0, 0];                 // puck velocity
      this.t = 0;                       // time in the current point
      this.hist = [];                   // recent observations (the AI's reaction time)
      this.reset(this.rand() < 0.5 ? 0 : 1);
    }

    // a new point. server: the player who gets the puck (the one who conceded). puckX: optional start x
    reset(server, puckX) {
      const side = server === 0 ? -1 : 1;
      this.pp = [[0, -0.8], [0, 0.8]];
      this.pv = [[0, 0], [0, 0]];
      this.kp = [puckX ?? (this.rand() * 0.3 - 0.15), side * 0.3];
      this.kv = [0, side * C.SERVE_SPEED];
      this.t = 0;
      const o = this.observeNow();
      this.hist = [];
      for (let i = 0; i <= C.REACTION_STEPS; i++) this.hist.push(o);
    }

    // [player] -> 12 numbers, from the player's own side
    observeNow() {
      const out = [];
      for (let p = 0; p < 2; p++) {
        const s = p === 0 ? 1 : -1, sx = s / C.HALF_W, sy = s / C.HALF_L, q = 1 - p;
        const vp = s / C.PADDLE_VMAX, vk = s / C.PUCK_VMAX;
        out.push([
          this.pp[p][0] * sx, this.pp[p][1] * sy, this.pv[p][0] * vp, this.pv[p][1] * vp,
          this.pp[q][0] * sx, this.pp[q][1] * sy, this.pv[q][0] * vp, this.pv[q][1] * vp,
          this.kp[0] * sx, this.kp[1] * sy, this.kv[0] * vk, this.kv[1] * vk,
        ]);
      }
      return out;
    }

    // delayed (what the AI sees): the puck and the opponent as they were REACTION_STEPS decisions ago,
    // its own paddle as it is now (a player feels where its own hand is)
    observe(delayed = true) {
      if (!delayed) return this.observeNow();
      const old = this.hist[0], now = this.hist[this.hist.length - 1];
      return [0, 1].map((p) => [...now[p].slice(0, 4), ...old[p].slice(4)]);
    }

    // actions: [[ax, ay], [ax, ay]] in [-1, 1], each in its player's own view.
    // Returns { goal: +1 player 0 scored / -1 player 1 scored / 0, timeout, touched: [b, b], reward: [r0, r1],
    //           hitSpeed: the relative speed of the strongest paddle hit (for sounds) }.
    // Unlike the Python version, it doesn't start the next point by itself - the page calls reset().
    step(actions) {
      const want = actions.map((a, p) => {
        const s = p === 0 ? 1 : -1;
        return [s * clamp(a[0], -1, 1) * C.PADDLE_VMAX, s * clamp(a[1], -1, 1) * C.PADDLE_VMAX];
      });
      const h = C.DT / C.SUBSTEPS, R = C.PUCK_R + C.PADDLE_R;
      let live = true, goal = 0, hitSpeed = 0, wallHit = 0;
      const touched = [false, false];
      const y0 = this.kp[1];
      for (let sub = 0; sub < C.SUBSTEPS && live; sub++) {
        // 1. paddles: steer toward the wanted velocity, with limited acceleration and top speed
        for (let p = 0; p < 2; p++) {
          let dx = want[p][0] - this.pv[p][0], dy = want[p][1] - this.pv[p][1];
          const dn = Math.hypot(dx, dy), lim = C.PADDLE_ACCEL * h;
          if (dn > lim) { dx *= lim / Math.max(dn, 1e-12); dy *= lim / Math.max(dn, 1e-12); }
          let vx = this.pv[p][0] + dx, vy = this.pv[p][1] + dy;
          const vn = Math.hypot(vx, vy);
          if (vn > C.PADDLE_VMAX) { vx *= C.PADDLE_VMAX / Math.max(vn, 1e-12); vy *= C.PADDLE_VMAX / Math.max(vn, 1e-12); }
          const px = this.pp[p][0] + vx * h, py = this.pp[p][1] + vy * h;
          const loY = p === 0 ? -C.HALF_L + C.PADDLE_R : C.PADDLE_R;
          const hiY = p === 0 ? -C.PADDLE_R : C.HALF_L - C.PADDLE_R;
          const cx = clamp(px, -C.HALF_W + C.PADDLE_R, C.HALF_W - C.PADDLE_R), cy = clamp(py, loY, hiY);
          if (cx !== px) vx = 0;              // stopped by the wall or the center line
          if (cy !== py) vy = 0;
          this.pp[p] = [cx, cy];
          this.pv[p] = [vx, vy];
        }
        // 2. the puck glides
        const f = Math.exp(-C.PUCK_FRICTION * h);
        let kvx = this.kv[0] * f, kvy = this.kv[1] * f;
        let kpx = this.kp[0] + kvx * h, kpy = this.kp[1] + kvy * h;
        [kpx, kpy, kvx, kvy, wallHit] = walls(kpx, kpy, kvx, kvy, wallHit);
        // 3. paddle hits
        for (let q = 0; q < 2; q++) {
          const dx = kpx - this.pp[q][0], dy = kpy - this.pp[q][1];
          const dist = Math.hypot(dx, dy);
          if (dist < R && dist > 1e-9) {
            const nx = dx / dist, ny = dy / dist;
            kpx = this.pp[q][0] + nx * R;
            kpy = this.pp[q][1] + ny * R;
            const vn = (kvx - this.pv[q][0]) * nx + (kvy - this.pv[q][1]) * ny;
            if (vn < 0) {
              kvx -= (1 + C.PADDLE_E) * vn * nx;
              kvy -= (1 + C.PADDLE_E) * vn * ny;
              touched[q] = true;
              hitSpeed = Math.max(hitSpeed, -vn);
            }
          }
        }
        const sp = Math.hypot(kvx, kvy);
        if (sp > C.PUCK_VMAX) { kvx *= C.PUCK_VMAX / Math.max(sp, 1e-12); kvy *= C.PUCK_VMAX / Math.max(sp, 1e-12); }
        [kpx, kpy, kvx, kvy, wallHit] = walls(kpx, kpy, kvx, kvy, wallHit);
        this.kp = [kpx, kpy];
        this.kv = [kvx, kvy];
        // 4. goals
        if (Math.abs(kpx) < C.GOAL_HALF && Math.abs(kpy) > C.HALF_L) {
          goal = Math.sign(kpy);
          live = false;
        }
      }
      this.t += C.DT;
      const timeout = goal === 0 && this.t >= C.POINT_TIME;
      const dy = goal === 0 ? this.kp[1] - y0 : 0;
      const tr = timeout ? C.TIMEOUT_REWARD : 0;
      const reward = [
        C.GOAL_REWARD * goal + C.PUSH_REWARD * dy + C.TOUCH_REWARD * touched[0] + tr,
        -C.GOAL_REWARD * goal - C.PUSH_REWARD * dy + C.TOUCH_REWARD * touched[1] + tr,
      ];
      this.hist.shift();
      this.hist.push(this.observeNow());
      return { goal, timeout, touched, reward, hitSpeed, wallHit };
    }
  }

  function walls(kpx, kpy, kvx, kvy, wallHit) {
    const lo = -C.HALF_W + C.PUCK_R, hi = C.HALF_W - C.PUCK_R;
    if (kpx < lo) { kpx = 2 * lo - kpx; wallHit = Math.max(wallHit, Math.abs(kvx)); kvx = -C.WALL_E * kvx; }
    else if (kpx > hi) { kpx = 2 * hi - kpx; wallHit = Math.max(wallHit, Math.abs(kvx)); kvx = -C.WALL_E * kvx; }
    // the end walls, except the goal mouth
    const end = C.HALF_L - C.PUCK_R;
    if (Math.abs(kpx) >= C.GOAL_HALF) {
      if (kpy < -end) { kpy = -2 * end - kpy; wallHit = Math.max(wallHit, Math.abs(kvy)); kvy = -C.WALL_E * kvy; }
      else if (kpy > end) { kpy = 2 * end - kpy; wallHit = Math.max(wallHit, Math.abs(kvy)); kvy = -C.WALL_E * kvy; }
    }
    return [kpx, kpy, kvx, kvy, wallHit];
  }

  // ---------------------------------------------------------------- the hand-written bot (src/bot.py)
  const BOT = { GAIN: 6.0, SPEED: 0.7, DRIVE: 0.7 };
  const sgn = (v) => (v > 0 ? 1 : v < 0 ? -1 : 0);

  // o: 12 numbers in the bot's own view. Returns [ax, ay] in [-1, 1].
  function botAction(o, skill = 1.0) {
    const me = [o[0] * C.HALF_W, o[1] * C.HALF_L];
    const puck = [o[8] * C.HALF_W, o[9] * C.HALF_L];
    const pvy = o[11] * C.PUCK_VMAX;
    // defend: on the line from our goal to the puck, 0.18 m in front of the goal line
    const tx = puck[0], ty = puck[1] + C.HALF_L;
    const t = clamp(0.18 / Math.max(ty, 0.05), 0, 1);
    const guard = [tx * t, Math.max(-C.HALF_L + ty * t, -C.HALF_L + C.PADDLE_R + 0.02)];
    // attack: get behind the puck (seen from the opponent's goal), then push through it
    let ax = 0 - puck[0], ay = C.HALF_L - puck[1];
    let an = Math.hypot(ax, ay) + 1e-9;
    ax /= an; ay /= an;
    const reach = C.PUCK_R + C.PADDLE_R + 0.03;
    if (puck[1] - ay * reach < -C.HALF_L + C.PADDLE_R + 0.01) {     // no room behind it: knock it out sideways
      ax = -sgn(puck[0] + 1e-9); ay = 0.35;
      an = Math.hypot(ax, ay);
      ax /= an; ay /= an;
    }
    const behind = [puck[0] - ax * reach, puck[1] - ay * reach];
    const linedUp = (me[0] - behind[0]) ** 2 + (me[1] - behind[1]) ** 2 < 0.04 ** 2;
    let attackPt = linedUp ? [puck[0] + ax * 0.25, puck[1] + ay * 0.25] : behind;
    const inFront = me[1] > puck[1] - 0.02;
    if (inFront) {                                                   // never pass the puck on the way back
      const beside = Math.abs(me[0] - puck[0]) > C.PUCK_R + C.PADDLE_R + 0.03;
      const away = sgn(me[0] - puck[0] + 1e-9);
      attackPt = beside ? [me[0], puck[1] - 0.12]
        : [clamp(puck[0] + away * 0.16, -C.HALF_W + C.PADDLE_R, C.HALF_W - C.PADDLE_R), me[1]];
    }
    const attack = puck[1] < 0 && !(pvy < -1.0);
    const point = attack ? attackPt : guard;
    let act = [clamp((point[0] - me[0]) * BOT.GAIN, -1, 1), clamp((point[1] - me[1]) * BOT.GAIN, -1, 1)];
    const k = (attack && linedUp && !inFront ? BOT.DRIVE : 1) * BOT.SPEED * Math.min(skill, 1);
    return [act[0] * k, act[1] * k];
  }

  root.Hockey = { C, Table, botAction };
})(typeof window !== "undefined" ? window : globalThis);
