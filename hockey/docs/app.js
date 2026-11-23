// =====================================================================================
//  Air Hockey AI - the web page.
//  physics.js: the table (a copy of src/physics.py) and the bot.  mlp.js: the trained networks.
//  You are player 0 (the near, red side). The AI is player 1 (the far, blue side).
// =====================================================================================
(() => {
  const { C, Table, botAction } = window.Hockey;
  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

  const S = {
    mode: "play",            // "play" (you vs AI) | "watch" (AI vs AI)
    phase: "menu",           // "menu" | "live" | "break" (after a goal) | "over"
    paused: false,
    table: new Table(),
    prev: null,              // the state before the last physics step (to draw smoothly between steps)
    acc: 0,
    breakLeft: 0,
    nextServer: 0,
    score: [0, 0],
    nets: null,
    order: [],               // the opponent choices: [{ id, label }]
    target: null,            // where the human wants the paddle (table coordinates)
    trail: [],
    sound: true,
  };

  // ---------------------------------------------------------------- the networks
  function brain(id) {
    if (id === "bot") return (o) => botAction(o);
    const net = S.nets[id];
    return (o) => net.forward(o).out.map((v) => clamp(v, -1, 1));
  }
  let topBrain = brain("bot"), bottomBrain = brain("bot");

  async function loadModels() {
    let m = null;
    try { m = await window.MLP.load("model"); } catch (e) { showBanner(e.message); }
    const opts = [];
    if (m) {
      S.nets = m.nets;
      const list = m.meta.networks;
      const fin = list.find((n) => n.id === "final");
      if (fin) opts.push({ id: fin.id, label: fin.label });
      list.filter((n) => n.id !== "final").reverse().forEach((n) => opts.push({ id: n.id, label: n.label }));
    } else {
      showBanner("No trained network has been exported yet, so the opponent is the rule-based bot. " +
        "Train in Colab, run  python -m src.export_web  and add docs/model/.");
    }
    opts.push({ id: "bot", label: "Rule-based bot (no learning)" });
    S.order = opts;
    for (const sel of [$("opponent"), $("bottom")]) {
      sel.innerHTML = opts.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
    }
    $("opponent").value = opts[0].id;
    $("bottom").value = opts[0].id;
    pickBrains();
  }

  function pickBrains() {
    topBrain = brain($("opponent").value);
    bottomBrain = brain($("bottom").value);
    const short = (sel) => sel.options[sel.selectedIndex].text.split(" (")[0].split(":")[0];
    $("nameTop").textContent = short($("opponent"));
    $("nameBottom").textContent = S.mode === "play" ? "You" : short($("bottom"));
  }

  function showBanner(text) {
    const b = $("banner");
    b.textContent = text;
    b.hidden = false;
  }

  // ---------------------------------------------------------------- the camera
  // World: x to the right, y away from you (along the table), z up. The camera stands behind your goal.
  const RAIL_W = 0.06, RAIL_H = 0.04, BODY = 0.09;
  const cam = { pos: [0, -1.95, 1.2], f: [0, 0, 0], r: [1, 0, 0], u: [0, 0, 1], s: 1, ox: 0, oy: 0 };
  const canvas = $("view"), ctx = canvas.getContext("2d");
  const bg = document.createElement("canvas"), bgx = bg.getContext("2d");
  const fg = document.createElement("canvas"), fgx = fg.getContext("2d");
  let W = 0, H = 0, DPR = 1;

  // aspect = height / width of the canvas. A tall (phone) screen gets a camera that looks more from above,
  // so the long table fills the height.
  function setCamera(aspect) {
    const k = clamp((aspect - 0.8) / 1.2, 0, 1);       // 0: wide screen, 1: tall phone screen
    cam.pos = [0, -1.95 + 0.75 * k, 1.2 + 1.7 * k];
    const target = [0, 0.12 - 0.08 * k, 0];
    const f = target.map((v, i) => v - cam.pos[i]);
    const fl = Math.hypot(...f);
    cam.f = f.map((v) => v / fl);
    cam.r = [1, 0, 0];
    // up = r x f
    cam.u = [cam.r[1] * cam.f[2] - cam.r[2] * cam.f[1], cam.r[2] * cam.f[0] - cam.r[0] * cam.f[2], cam.r[0] * cam.f[1] - cam.r[1] * cam.f[0]];
  }

  function rawProj(x, y, z) {
    const dx = x - cam.pos[0], dy = y - cam.pos[1], dz = z - cam.pos[2];
    const d = dx * cam.f[0] + dy * cam.f[1] + dz * cam.f[2];
    return [(dx * cam.r[0] + dy * cam.r[1] + dz * cam.r[2]) / d, -(dx * cam.u[0] + dy * cam.u[1] + dz * cam.u[2]) / d, d];
  }
  const P = (x, y, z = 0) => {
    const [u, v, d] = rawProj(x, y, z);
    return [cam.ox + u * cam.s, cam.oy + v * cam.s, d];
  };

  // screen point -> the point on the table under it
  function unproject(sx, sy, z = 0) {
    const u = (sx - cam.ox) / cam.s, v = (sy - cam.oy) / cam.s;
    const dir = [0, 1, 2].map((i) => cam.f[i] + u * cam.r[i] - v * cam.u[i]);
    if (dir[2] >= -1e-6) return null;
    const t = (z - cam.pos[2]) / dir[2];
    return [cam.pos[0] + t * dir[0], cam.pos[1] + t * dir[1]];
  }

  function resize() {
    const r = canvas.getBoundingClientRect();
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    W = Math.max(1, Math.round(r.width * DPR));
    H = Math.max(1, Math.round(r.height * DPR));
    for (const c of [canvas, bg, fg]) { c.width = W; c.height = H; }
    setCamera(r.height / Math.max(1, r.width));
    // fit the whole table (with its rails) into the canvas, below the scoreboard
    const X = C.HALF_W + RAIL_W, Y = C.HALF_L + RAIL_W;
    const pts = [[-X, -Y, -BODY], [X, -Y, -BODY], [-X, Y, RAIL_H], [X, Y, RAIL_H], [-X, -Y, RAIL_H], [X, -Y, RAIL_H]].map((p) => rawProj(...p));
    const minU = Math.min(...pts.map((p) => p[0])), maxU = Math.max(...pts.map((p) => p[0]));
    const minV = Math.min(...pts.map((p) => p[1])), maxV = Math.max(...pts.map((p) => p[1]));
    const top = 58 * DPR, bottom = 10 * DPR, side = 10 * DPR;
    cam.s = Math.min((W - 2 * side) / (maxU - minU), (H - top - bottom) / (maxV - minV));
    cam.ox = W / 2 - ((minU + maxU) / 2) * cam.s;
    cam.oy = top + (H - top - bottom) / 2 - ((minV + maxV) / 2) * cam.s;
    drawStatic();
  }

  // ---------------------------------------------------------------- drawing helpers
  function poly(g, pts, fill, stroke, lw = 1) {
    g.beginPath();
    pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
    g.closePath();
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw; g.stroke(); }
  }
  function ring(cx, cy, z, r, n = 40, a0 = 0, a1 = Math.PI * 2) {
    const out = [];
    for (let i = 0; i <= n; i++) {
      const a = a0 + ((a1 - a0) * i) / n;
      out.push(P(cx + r * Math.cos(a), cy + r * Math.sin(a), z));
    }
    return out;
  }
  function line(g, pts, color, lw) {
    g.beginPath();
    pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
    g.strokeStyle = color; g.lineWidth = lw; g.stroke();
  }
  const quad = (a, b, c, d) => [P(...a), P(...b), P(...c), P(...d)];

  // the table, markings, side and far rails (drawn once per resize)
  function drawStatic() {
    const g = bgx, k = DPR;
    g.clearRect(0, 0, W, H);
    const X = C.HALF_W, Y = C.HALF_L, XR = X + RAIL_W, YR = Y + RAIL_W;
    // the table's body (the near front, below the surface)
    poly(g, quad([-XR, -YR, RAIL_H], [XR, -YR, RAIL_H], [XR, -YR, -BODY], [-XR, -YR, -BODY]), "#0c1322");
    // the surface
    const s = P(0, -Y), e = P(0, Y);
    const grad = g.createLinearGradient(s[0], s[1], e[0], e[1]);
    grad.addColorStop(0, "#f4f8fa"); grad.addColorStop(1, "#d9e6ee");
    poly(g, quad([-X, -Y, 0], [X, -Y, 0], [X, Y, 0], [-X, Y, 0]), grad);
    // the air holes
    g.fillStyle = "rgba(60, 90, 120, 0.22)";
    for (let iy = -19; iy <= 19; iy++) for (let ix = -9; ix <= 9; ix++) {
      const p = P(ix * 0.05, iy * 0.05);
      const rr = Math.max(0.6, 0.0045 * cam.s / p[2]);
      g.beginPath(); g.arc(p[0], p[1], rr, 0, Math.PI * 2); g.fill();
    }
    // markings
    line(g, [P(-X, 0), P(X, 0)], "rgba(230, 57, 70, 0.75)", 3 * k);
    line(g, ring(0, 0, 0, 0.14), "rgba(230, 57, 70, 0.55)", 2 * k);
    line(g, ring(0, -Y, 0, 0.26, 30, 0, Math.PI), "rgba(230, 57, 70, 0.5)", 2 * k);
    line(g, ring(0, Y, 0, 0.26, 30, Math.PI, Math.PI * 2), "rgba(43, 142, 240, 0.55)", 2 * k);
    line(g, [P(-X, -Y / 2), P(X, -Y / 2)], "rgba(230, 57, 70, 0.25)", 1.5 * k);
    line(g, [P(-X, Y / 2), P(X, Y / 2)], "rgba(43, 142, 240, 0.3)", 1.5 * k);
    // the far goal: a dark slot
    poly(g, quad([-C.GOAL_HALF, Y, 0], [C.GOAL_HALF, Y, 0], [C.GOAL_HALF, YR, 0], [-C.GOAL_HALF, YR, 0]), "#0a0f1a");
    // far rail: inner face + top, on both sides of the goal
    const railFace = "#3a4d70", railTop = "#55698f";
    for (const [a, b] of [[-XR, -C.GOAL_HALF], [C.GOAL_HALF, XR]]) {
      poly(g, quad([a, Y, 0], [b, Y, 0], [b, Y, RAIL_H], [a, Y, RAIL_H]), railFace);
      poly(g, quad([a, Y, RAIL_H], [b, Y, RAIL_H], [b, YR, RAIL_H], [a, YR, RAIL_H]), railTop);
    }
    poly(g, quad([-C.GOAL_HALF, Y, RAIL_H * 0.4], [C.GOAL_HALF, Y, RAIL_H * 0.4], [C.GOAL_HALF, Y, RAIL_H], [-C.GOAL_HALF, Y, RAIL_H]), "rgba(43,142,240,0.9)");
    // side rails
    for (const sgn of [-1, 1]) {
      poly(g, quad([sgn * X, -Y, 0], [sgn * X, Y, 0], [sgn * X, Y, RAIL_H], [sgn * X, -Y, RAIL_H]), railFace);
      poly(g, quad([sgn * X, -YR, RAIL_H], [sgn * X, YR, RAIL_H], [sgn * XR, YR, RAIL_H], [sgn * XR, -YR, RAIL_H]), railTop);
      poly(g, quad([sgn * XR, -YR, RAIL_H], [sgn * XR, YR, RAIL_H], [sgn * XR, YR, -BODY], [sgn * XR, -YR, -BODY]), "#101a2c");
    }
    // the near rail goes in front of everything (a separate layer)
    const h = fgx;
    h.clearRect(0, 0, W, H);
    for (const [a, b] of [[-XR, -C.GOAL_HALF], [C.GOAL_HALF, XR]]) {
      poly(h, quad([a, -YR, RAIL_H], [b, -YR, RAIL_H], [b, -Y, RAIL_H], [a, -Y, RAIL_H]), railTop);
      poly(h, quad([a, -YR, RAIL_H], [b, -YR, RAIL_H], [b, -YR, -BODY], [a, -YR, -BODY]), "#24334f");
    }
    poly(h, quad([-C.GOAL_HALF, -YR, 0], [C.GOAL_HALF, -YR, 0], [C.GOAL_HALF, -YR, -BODY], [-C.GOAL_HALF, -YR, -BODY]), "#0a0f1a");
    poly(h, quad([-C.GOAL_HALF, -YR, -0.005], [C.GOAL_HALF, -YR, -0.005], [C.GOAL_HALF, -YR, -0.02], [-C.GOAL_HALF, -YR, -0.02]), "rgba(230,57,70,0.9)");
  }

  // a cylinder standing on the table: (x, y), radius, from z0 to z1
  function cylinder(x, y, r, z0, z1, side, top, rim) {
    const b = ring(x, y, z0, r, 36), t = ring(x, y, z1, r, 36);
    let li = 0, ri = 0;
    b.forEach((p, i) => { if (p[0] < b[li][0]) li = i; if (p[0] > b[ri][0]) ri = i; });
    poly(ctx, b, side);
    poly(ctx, [b[li], t[li], t[ri], b[ri]], side);
    poly(ctx, t, top, rim, 1.5 * DPR);
  }

  function shadow(x, y, r) {
    poly(ctx, ring(x + 0.012, y + 0.018, 0, r * 1.05, 24), "rgba(20, 35, 60, 0.18)");
  }

  function paddle(x, y, color, dark) {
    shadow(x, y, C.PADDLE_R);
    cylinder(x, y, C.PADDLE_R, 0, 0.022, dark, color, "rgba(255,255,255,0.35)");
    line(ctx, ring(x, y, 0.022, C.PADDLE_R * 0.72, 30), "rgba(0,0,0,0.18)", 2 * DPR);
    cylinder(x, y, C.PADDLE_R * 0.42, 0.022, 0.06, dark, color, "rgba(255,255,255,0.5)");
    // a highlight on the knob
    const p = P(x - 0.008, y + 0.006, 0.06);
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(1, 0.006 * cam.s / p[2]), 0, Math.PI * 2); ctx.fill();
  }

  function puck(x, y) {
    shadow(x, y, C.PUCK_R);
    cylinder(x, y, C.PUCK_R, 0, 0.012, "#b07a00", "#ffb703", "rgba(255,255,255,0.6)");
    line(ctx, ring(x, y, 0.012, C.PUCK_R * 0.55, 24), "rgba(120, 70, 0, 0.45)", 1.5 * DPR);
  }

  // ---------------------------------------------------------------- one frame
  function render(alpha) {
    const T = S.table, Q = S.prev || T;
    const lerp = (a, b) => a + (b - a) * alpha;
    const pp = [0, 1].map((p) => [lerp(Q.pp[p][0], T.pp[p][0]), lerp(Q.pp[p][1], T.pp[p][1])]);
    const kp = [lerp(Q.kp[0], T.kp[0]), lerp(Q.kp[1], T.kp[1])];
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(bg, 0, 0);
    // the puck's trail
    S.trail.push(kp);
    if (S.trail.length > 9) S.trail.shift();
    S.trail.forEach((q, i) => {
      if (i === S.trail.length - 1) return;
      const a = (i + 1) / S.trail.length;
      poly(ctx, ring(q[0], q[1], 0.001, C.PUCK_R * (0.5 + 0.5 * a), 16), `rgba(255, 183, 3, ${0.18 * a})`);
    });
    // far to near
    const items = [
      { y: pp[1][1], draw: () => paddle(pp[1][0], pp[1][1], "#2b8ef0", "#185ea3") },
      { y: pp[0][1], draw: () => paddle(pp[0][0], pp[0][1], "#e63946", "#9c1f2a") },
      { y: kp[1], draw: () => (Math.abs(kp[1]) > C.HALF_L ? null : puck(kp[0], kp[1])) },     // in the goal: gone
    ].sort((a, b) => b.y - a.y);
    items.forEach((it) => it.draw());
    // where you are aiming (a faint ring), while playing
    if (S.mode === "play" && S.target && S.phase === "live") {
      line(ctx, ring(S.target[0], S.target[1], 0.001, 0.02, 20), "rgba(230,57,70,0.35)", 1.5 * DPR);
    }
    ctx.drawImage(fg, 0, 0);
  }

  // ---------------------------------------------------------------- the game
  function copyState(T) {
    return { pp: T.pp.map((p) => p.slice()), kp: T.kp.slice() };
  }

  function humanAction() {
    const me = S.table.pp[0];
    if (!S.target) return [0, 0];
    const tx = clamp(S.target[0], -C.HALF_W + C.PADDLE_R, C.HALF_W - C.PADDLE_R);
    const ty = clamp(S.target[1], -C.HALF_L + C.PADDLE_R, -C.PADDLE_R);
    return [clamp((tx - me[0]) * 12, -1, 1), clamp((ty - me[1]) * 12, -1, 1)];
  }

  function physicsStep() {
    const T = S.table;
    S.prev = copyState(T);
    const obs = T.observe(true);
    const a0 = S.mode === "play" ? humanAction() : bottomBrain(obs[0]);
    const a1 = topBrain(obs[1]);
    const r = T.step([a0, a1]);
    if (r.hitSpeed > 0.15) sfx("hit", r.hitSpeed);
    else if (r.wallHit > 0.4) sfx("wall", r.wallHit);
    if (r.goal !== 0) {
      const who = r.goal > 0 ? 0 : 1;
      S.score[who] += 1;
      updateScore();
      sfx("goal", who);
      const youScored = S.mode === "play" && who === 0;
      const name = who === 0 ? $("nameBottom").textContent : $("nameTop").textContent;
      flash(S.mode === "play" ? (youScored ? "Goal!" : "AI scores") : `${name} scores`, who === 0 ? "var(--red)" : "var(--blue)");
      S.nextServer = 1 - who;
      if (S.score[who] >= C.MATCH_GOALS) return endMatch(who);
      S.phase = "break";
      S.breakLeft = 1.1;
    } else if (r.timeout) {
      flash("Replay", "var(--ink)");
      S.nextServer = Math.random() < 0.5 ? 0 : 1;
      S.phase = "break";
      S.breakLeft = 0.7;
    }
  }

  function newPoint() {
    S.table.reset(S.nextServer);
    S.prev = null;
    S.trail = [];
    S.phase = "live";
    // your paddle starts where it is reset to: aim there until you move
    if (S.mode === "play" && !S.target) S.target = S.table.pp[0].slice();
  }

  function startMatch() {
    pickBrains();
    S.score = [0, 0];
    updateScore();
    S.nextServer = 0;
    hideOverlay();
    S.paused = false;
    setPauseButton();
    newPoint();
    if (S.mode === "play") window.Plays?.played("hockey");
  }

  function endMatch(winner) {
    S.phase = "over";
    let title, text;
    if (S.mode === "play") {
      title = winner === 0 ? "You win!" : "The AI wins";
      text = winner === 0 ? "You beat " + $("nameTop").textContent + ". Try a stronger opponent?" : "It learned this by playing against itself. Another round?";
    } else {
      title = (winner === 0 ? $("nameBottom").textContent : $("nameTop").textContent) + " wins";
      text = "Pick two other players to compare the learning stages.";
    }
    setTimeout(() => showOverlay(`
      <div class="card">
        <h2>${title}</h2>
        <div class="final"><span class="r">${S.score[0]}</span> : <span class="b">${S.score[1]}</span></div>
        <p>${text}</p>
        <button class="btn primary" id="again">${S.mode === "play" ? "Play again" : "Watch again"}</button>
      </div>`), 900);
  }

  function updateScore() {
    $("scoreBottom").textContent = S.score[0];
    $("scoreTop").textContent = S.score[1];
  }

  let flashTimer = 0;
  function flash(text, color) {
    const f = $("flash");
    f.textContent = text;
    f.style.color = color;
    f.classList.add("on");
    clearTimeout(flashTimer);
    flashTimer = setTimeout(() => f.classList.remove("on"), 900);
  }

  function showOverlay(html) {
    const o = $("overlay");
    o.innerHTML = html;
    o.classList.add("on");
    const b = o.querySelector("#again, #start");
    if (b) { b.addEventListener("click", startMatch); b.focus({ preventScroll: true }); }
  }
  function hideOverlay() { $("overlay").classList.remove("on"); }

  function menu() {
    S.phase = "menu";
    showOverlay(`
      <div class="card">
        <h2>Air hockey vs a neural network</h2>
        <p>First to 7 goals. Move your red paddle with the mouse, or drag with your finger. Defend the near goal and shoot at the far one.</p>
        <button class="btn primary" id="start">Start match</button>
      </div>`);
  }

  // ---------------------------------------------------------------- the main loop
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (!S.paused && S.phase !== "menu") {
      if (S.phase === "live") {
        S.acc += dt;
        while (S.acc >= C.DT && S.phase === "live") { S.acc -= C.DT; physicsStep(); }
      } else if (S.phase === "break") {
        S.breakLeft -= dt;
        S.acc = 0;
        if (S.breakLeft <= 0) newPoint();
      }
    }
    render(S.phase === "live" ? clamp(S.acc / C.DT, 0, 1) : 1);
    requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------- input
  const stage = $("stage");
  function aim(e) {
    if (S.mode !== "play") return;
    const r = canvas.getBoundingClientRect();
    const p = unproject((e.clientX - r.left) * DPR, (e.clientY - r.top) * DPR, 0.03);
    if (!p) return;
    // with a finger, the paddle sits a little above the touch point, so the finger doesn't hide it
    S.target = [p[0], p[1] + (e.pointerType === "touch" ? 0.07 : 0)];
  }
  stage.addEventListener("pointermove", aim);
  stage.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".overlay.on")) return;
    aim(e);
    try { stage.setPointerCapture(e.pointerId); } catch (_) {}
    unlockAudio();
  });

  // ---------------------------------------------------------------- sound (made on the fly, no files)
  let audio = null;
  function unlockAudio() {
    if (!S.sound) return;
    try {
      if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
      if (audio.state === "suspended") audio.resume();
    } catch (_) { audio = null; }
  }
  function sfx(kind, v) {
    if (!S.sound || !audio || audio.state !== "running") return;
    const t = audio.currentTime;
    const o = audio.createOscillator(), g = audio.createGain();
    o.connect(g); g.connect(audio.destination);
    if (kind === "hit") {
      o.type = "triangle"; o.frequency.setValueAtTime(420 + v * 60, t);
      o.frequency.exponentialRampToValueAtTime(140, t + 0.07);
      g.gain.setValueAtTime(Math.min(0.35, 0.06 + v * 0.06), t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.08);
      o.start(t); o.stop(t + 0.09);
    } else if (kind === "wall") {
      o.type = "square"; o.frequency.setValueAtTime(900, t);
      g.gain.setValueAtTime(Math.min(0.08, 0.02 + v * 0.015), t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.03);
      o.start(t); o.stop(t + 0.04);
    } else {
      o.type = "sine";
      const up = S.mode !== "play" || v === 0;
      o.frequency.setValueAtTime(up ? 523 : 392, t);
      o.frequency.setValueAtTime(up ? 784 : 262, t + 0.12);
      g.gain.setValueAtTime(0.2, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.35);
      o.start(t); o.stop(t + 0.36);
    }
  }

  // ---------------------------------------------------------------- controls
  function setMode(mode) {
    S.mode = mode;
    document.querySelectorAll("#modes button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    $("bottomField").hidden = mode !== "watch";
    stage.classList.toggle("watching", mode === "watch");
    $("hint").textContent = mode === "play"
      ? "Move your red paddle with the mouse or your finger. You defend the near goal."
      : "Two AI players play each other. Pick the bottom player and the opponent to compare the learning stages.";
    pickBrains();
    if (mode === "watch") startMatch(); else menu();
  }
  document.querySelectorAll("#modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("newMatch").addEventListener("click", () => { unlockAudio(); startMatch(); });
  $("opponent").addEventListener("change", () => { pickBrains(); if (S.phase !== "menu") startMatch(); });
  $("bottom").addEventListener("change", () => { pickBrains(); if (S.phase !== "menu") startMatch(); });
  function setPauseButton() {
    $("pause").textContent = S.paused ? "Resume" : "Pause";
    $("pause").setAttribute("aria-pressed", String(S.paused));
  }
  $("pause").addEventListener("click", () => { S.paused = !S.paused; setPauseButton(); });
  $("sound").addEventListener("click", () => {
    S.sound = !S.sound;
    $("sound").setAttribute("aria-pressed", String(S.sound));
    $("sound").textContent = S.sound ? "🔊" : "🔈";
    if (S.sound) unlockAudio();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "p" || e.key === "P" || (e.key === " " && e.target === document.body)) {
      e.preventDefault();
      if (S.phase !== "menu" && S.phase !== "over") { S.paused = !S.paused; setPauseButton(); }
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && S.phase === "live" && S.mode === "play") { S.paused = true; setPauseButton(); }
  });
  $("overlay").addEventListener("pointerdown", unlockAudio);

  new ResizeObserver(resize).observe(canvas);
  window.addEventListener("resize", resize);

  // ---------------------------------------------------------------- start
  resize();
  loadModels().then(() => {
    menu();
    requestAnimationFrame(frame);
  });
  window.HockeyApp = S;            // for testing
  window.Plays?.badge("hockey", document.querySelector("header h1"));
})();
