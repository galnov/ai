// =====================================================================================
//  Breakout - a JavaScript copy of src/game.py for one board, and the 48 x 48 screen the network sees.
//  Keep the numbers in sync with src/config.py. Units: pixels of the network's screen, seconds; y goes down.
// =====================================================================================
(function (root) {
  const C = {
    SCREEN: 48, BRICK_ROWS: 5, BRICK_COLS: 8, BRICK_W: 6, BRICK_H: 3, BRICK_TOP: 6,
    PADDLE_W: 8, PADDLE_Y: 44,
    DT: 1 / 15, SUBSTEPS: 4, PADDLE_SPEED: 66.0, BALL_SPEED: 34.0, BALL_SPEED_MAX: 50.0, SPEEDUP: 1.02,
    MAX_ANGLE: 60, MIN_VY: 0.3, LIVES: 3, SERVE_ANGLE: [15, 45], MAX_STEPS: 3000, FRAMES: 2,
  };
  const S = C.SCREEN;
  const ROW_GRAY = [230, 200, 170, 140, 110];
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
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const rad = (d) => d * (Math.PI / 180);       // the same rounding as numpy.radians

  class Board {
    // rand: a function returning [0, 1) - two boards with the same seed get the same serves.
    // headless: no screens (the leaderboard server replays games this way - much faster)
    constructor(rand = Math.random, headless = false) {
      this.rand = rand;
      this.headless = headless;
      this.reset();
    }

    reset() {
      this.px = S / 2;
      this.bricks = [];
      for (let r = 0; r < C.BRICK_ROWS; r++) this.bricks.push(new Array(C.BRICK_COLS).fill(true));
      this.lives = C.LIVES;
      this.score = 0;
      this.level = 1;
      this.steps = 0;
      this.over = false;
      this.serve();
      this.frames = [];
      if (this.headless) return;
      const scr = this.render();
      for (let i = 0; i < C.FRAMES; i++) this.frames.push(scr);
    }

    serve() {
      const a = rad(C.SERVE_ANGLE[0] + this.rand() * (C.SERVE_ANGLE[1] - C.SERVE_ANGLE[0])) * (this.rand() < 0.5 ? -1 : 1);
      this.speed = C.BALL_SPEED;
      this.bx = this.px;
      this.by = C.PADDLE_Y - 1.0;
      this.vx = C.BALL_SPEED * DMath.sin(a);
      this.vy = -C.BALL_SPEED * DMath.cos(a);
    }

    // move: -1 left, 0 stay, +1 right; or a number in between (your paddle follows the pointer smoothly).
    // Returns { reward, bricks: [[row, col], ...] broken, lost, paddle (the ball hit the paddle), wall, cleared }
    step(move) {
      const ev = { reward: 0, bricks: [], lost: false, paddle: false, wall: false, cleared: false };
      if (this.over) return ev;
      const m = clamp(move, -1, 1) * C.PADDLE_SPEED, h = C.DT / C.SUBSTEPS, half = C.PADDLE_W / 2;
      for (let sub = 0; sub < C.SUBSTEPS; sub++) {
        this.px = clamp(this.px + m * h, half, S - half);
        const ox = this.bx, oy = this.by;
        this.bx += this.vx * h;
        this.by += this.vy * h;
        if (this.bx < 1) { this.bx = 2 - this.bx; this.vx = Math.abs(this.vx); ev.wall = true; }
        else if (this.bx > S - 1) { this.bx = 2 * (S - 1) - this.bx; this.vx = -Math.abs(this.vx); ev.wall = true; }
        if (this.by < 1) { this.by = 2 - this.by; this.vy = Math.abs(this.vy); ev.wall = true; }
        // bricks
        const row = Math.floor((this.by - C.BRICK_TOP) / C.BRICK_H), col = Math.floor(this.bx / C.BRICK_W);
        if (row >= 0 && row < C.BRICK_ROWS && col >= 0 && col < C.BRICK_COLS && this.bricks[row][col]) {
          this.bricks[row][col] = false;
          ev.reward += 1;
          ev.bricks.push([row, col]);
          this.score += 1;
          const orow = Math.floor((oy - C.BRICK_TOP) / C.BRICK_H);
          if (orow !== row) { this.vy = -this.vy; this.by = oy; }
          else { this.vx = -this.vx; this.bx = ox; }
          this.speed = Math.min(this.speed * C.SPEEDUP, C.BALL_SPEED_MAX);
          this.fixSpeed();
        }
        // the paddle
        const land = C.PADDLE_Y - 1;
        if (this.vy > 0 && oy <= land && this.by > land && Math.abs(this.bx - this.px) <= half + 1) {
          const off = clamp((this.bx - this.px) / (half + 1), -1, 1), a = rad(off * C.MAX_ANGLE);
          this.vx = this.speed * DMath.sin(a);
          this.vy = -this.speed * DMath.cos(a);
          this.by = 2 * land - this.by;
          ev.paddle = true;
        }
        // a lost ball
        if (this.by > S + 1) {
          ev.lost = true;
          ev.reward -= 1;
          this.lives -= 1;
          this.serve();
        }
        // a cleared wall
        if (!this.bricks.some((r) => r.some(Boolean))) {
          for (const r of this.bricks) r.fill(true);
          this.level += 1;
          ev.cleared = true;
        }
      }
      this.steps += 1;
      if (this.lives <= 0 || this.steps >= C.MAX_STEPS) this.over = true;
      if (!this.headless) {
        this.frames.shift();
        this.frames.push(this.render());
      }
      return ev;
    }

    fixSpeed() {
      const sp = this.speed, n = DMath.hypot(this.vx, this.vy) + 1e-9;
      let vx = (this.vx / n) * sp, vy = (this.vy / n) * sp;
      if (Math.abs(vy) < C.MIN_VY * sp) {
        vy = Math.sign(vy + 1e-9) * C.MIN_VY * sp;
        vx = Math.sign(vx + 1e-9) * Math.sqrt(Math.max(sp * sp - vy * vy, 0));
      }
      this.vx = vx;
      this.vy = vy;
    }

    // the 48 x 48 gray screen (Uint8Array, row by row), exactly as src/game.py draws it
    render() {
      const img = new Uint8Array(S * S);
      for (let r = 0; r < C.BRICK_ROWS; r++) for (let c = 0; c < C.BRICK_COLS; c++) {
        if (!this.bricks[r][c]) continue;
        for (let dy = 0; dy < C.BRICK_H - 1; dy++) {
          const y = C.BRICK_TOP + r * C.BRICK_H + dy;
          for (let dx = 0; dx < C.BRICK_W - 1; dx++) img[y * S + c * C.BRICK_W + dx] = ROW_GRAY[r];
        }
      }
      const x0 = Math.floor(this.px - C.PADDLE_W / 2);
      for (let i = 0; i < C.PADDLE_W; i++) {
        const x = clamp(x0 + i, 0, S - 1);
        img[C.PADDLE_Y * S + x] = 255;
        img[(C.PADDLE_Y + 1) * S + x] = 255;
      }
      if (!this.over) {
        const bx0 = Math.floor(this.bx - 1), by0 = Math.floor(this.by - 1);
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
          const x = bx0 + dx, y = by0 + dy;
          if (x >= 0 && x < S && y >= 0 && y < S) img[y * S + x] = 255;
        }
      }
      return img;
    }

    // what the network gets: FRAMES screens, the oldest first (FRAMES * 48 * 48 numbers, 0..255)
    observe() {
      const out = new Float32Array(C.FRAMES * S * S);
      this.frames.forEach((f, i) => out.set(f, i * S * S));
      return out;
    }
  }

  // a small seeded random generator, so the two boards in a race get the same serves
  function seeded(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  root.Breakout = { C, Board, seeded, ROW_GRAY };
})(typeof window !== "undefined" ? window : globalThis);
