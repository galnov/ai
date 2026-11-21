// =====================================================================================
//  Runner AI - the client side.
//  The game runs here (game.js) at 20 ticks per second; the picture is drawn at the screen's
//  frame rate, smoothly in between ticks. The AI ghost plays the same track: its key presses
//  come from backend.js (in this page) in chunks of 10 seconds.
// =====================================================================================
const { C, Game, jumpHeight } = window.Runner;
const { Engine } = window.Engine3D;
const $ = (id) => document.getElementById(id);
const TOUCH = matchMedia("(pointer: coarse)").matches || "ontouchstart" in window;   // a phone or a tablet

const LANE_W = 2.5;                                 // meters between lane centers
const laneX = (l) => (l - 1) * LANE_W;
const CRASH_TEXT = { hurdle: "a hurdle", "overhead bar": "an overhead bar", train: "a train" };
const KEYS = {
  ArrowLeft: C.LEFT, a: C.LEFT, A: C.LEFT, ArrowRight: C.RIGHT, d: C.RIGHT, D: C.RIGHT,
  ArrowUp: C.JUMP, w: C.JUMP, W: C.JUMP, " ": C.JUMP, ArrowDown: C.SLIDE, s: C.SLIDE, S: C.SLIDE,
};
const ACTION_NAMES = ["Run", "Left", "Right", "Jump", "Slide"];

const S = {
  info: null, mode: "race", model: null, speedMul: 1,
  phase: "menu",              // menu | countdown | running | ending | over
  me: null, ghost: null, seed: 0,
  myActions: [], input: [],
  sid: null, gActions: [], gScores: [], gOver: false, fetching: false, gFinal: null,
  snap: { me: null, ghost: null }, prev: { me: null, ghost: null },
  acc: 0, last: 0, phaseT: 0, focus: "me", toast: null, scores: null, shake: 0,
};

let engine;
try {
  engine = new Engine($("view"));
} catch (e) {
  $("overlay").innerHTML = `<div class="card-o"><h2>3D graphics are not available</h2><p>${e.message}</p></div>`;
  throw e;
}

// ---------------------------------------------------------------- the "server" (backend.js, in this page)

async function loadInfo() {
  S.info = await Backend.info();
  const sel = $("opponent");
  sel.innerHTML = S.info.players.map((p) => `<option value="${p.id}">${p.label}</option>`).join("");
  const saved = localStorage.getItem("runner.model");
  S.model = S.info.players.some((p) => p.id === saved) ? saved : S.info.players[0].id;
  sel.value = S.model;
  $("banner").hidden = !S.info.message;
  $("banner").textContent = S.info.message || "";
  renderBoard(S.info.leaderboard);
  showMenu();
}

// ---------------------------------------------------------------- starting and ending runs
async function startRun() {
  if (S.mode === "race") window.Plays?.played("runner");       // one more race, for the counter everyone sees
  if (TOUCH) document.querySelector(".stage").scrollIntoView({ behavior: "smooth", block: "start" });   // the picture and the buttons on the screen
  S.seed = Math.floor(Math.random() * 2 ** 31);
  S.me = S.mode === "race" ? new Game(S.seed) : null;
  S.ghost = new Game(S.seed);
  S.myActions = []; S.input = []; S.gActions = []; S.gScores = []; S.gOver = false; S.gFinal = null;
  S.toast = null; S.scores = null; S.focus = S.mode === "race" ? "me" : "ghost";
  for (const k of ["me", "ghost"]) { S.snap[k] = snapshot(S[k]); S.prev[k] = S.snap[k]; }
  S.phase = "countdown"; S.phaseT = 0; S.acc = 0;
  $("overlay").innerHTML = "";
  const r = await Backend.ghostStart({ model: S.model, seed: S.seed });
  S.sid = r.sid; S.model = r.model;
  addGhostChunk(r);
}

function addGhostChunk(r) {
  S.gActions.push(...r.actions);
  S.gScores.push(...r.scores);
  S.gOver = r.over;
}

async function fetchMore() {
  if (S.fetching || S.gOver || !S.sid) return;
  S.fetching = true;
  const sid = S.sid;
  const r = await Backend.ghostMore({ sid });
  if (sid === S.sid && !r.error) addGhostChunk(r);
  S.fetching = false;
}

async function myCrash() {
  S.phase = "ending"; S.phaseT = 0; S.shake = 1;
  const me = S.me;
  let ai;
  if (!S.ghost.alive) ai = { distance: S.ghost.distance(), crash: S.ghost.crashType ? crashName(S.ghost.crashType) : null };
  else {
    S.toast = { text: "You crashed! The AI is still running…", t: 0 };
    S.gOver = true;                                                 // no more chunks: the backend now plays to the end
    const r = await Backend.ghostFinish({ sid: S.sid });
    ai = { distance: r.distance, crash: r.crash, timeout: !r.crash };
  }
  S.gFinal = ai;
  let saved = null;
  if ($("record").checked) {
    const presses = [];
    S.myActions.forEach((a, t) => { if (a) presses.push([t, a]); });
    saved = await Backend.recording({
      seed: S.seed, ticks: S.myActions.length, presses, distance: me.distance(),
      player: $("name").value.trim(), ai: { model: S.model, distance: ai.distance },
    });
    if (saved.leaderboard) renderBoard(saved.leaderboard);
  }
  const wait = Math.max(0, 1600 - S.phaseT);
  setTimeout(() => showResult(me, ai, saved), wait);
}

function crashName(t) { return { 1: "hurdle", 2: "overhead bar", 3: "train" }[t]; }

function showResult(me, ai, saved) {
  S.phase = "over";
  const mine = me.distance(), win = mine > ai.distance;
  const aiLabel = S.info.players.find((p) => p.id === S.model)?.label || "AI";
  const savedText = !$("record").checked ? "Not recorded (recording is off)."
    : saved?.saved ? "Saved in this browser. Download your runs (under the leaderboard) to use them as training data."
    : `Not saved: ${saved?.error || "unknown error"}`;
  $("overlay").innerHTML = `
    <div class="card-o result ${win ? "win" : "lose"}">
      <h2>${win ? "You beat the AI!" : "The AI wins this one"}</h2>
      <div class="vs">
        <div class="you"><small>You</small><b>${fmt(mine)}</b><span>hit ${CRASH_TEXT[crashName(me.crashType)] || "something"}</span></div>
        <div class="ai"><small>${aiLabel}</small><b>${fmt(ai.distance)}</b><span>${ai.crash ? "hit " + CRASH_TEXT[ai.crash] : "still running after 10 minutes"}</span></div>
      </div>
      <p class="note">${savedText}</p>
      <button class="btn primary" id="again">Race again <kbd>Enter</kbd></button>
    </div>`;
  $("again").onclick = startRun;
}

function showMenu() {
  S.phase = "menu";
  const watch = S.mode === "watch";
  $("overlay").innerHTML = `
    <div class="card-o">
      <h2>${watch ? "Watch the AI run" : "Race the AI"}</h2>
      <p>${watch ? "The AI plays a new track every time. Change the opponent on the right to compare the three networks."
        : "You and the AI ghost run the same track. Whoever gets farther wins."}</p>
      ${watch ? "" : TOUCH ? `<ul class="keys">
        <li>Swipe <b>left / right</b> to change lanes (trains)</li>
        <li>Swipe <b>up</b> to jump (hurdles)</li>
        <li>Swipe <b>down</b> to slide (overhead bars)</li>
        <li>or use the buttons under the picture</li></ul>` : `<ul class="keys">
        <li><kbd>←</kbd><kbd>→</kbd> or <kbd>A</kbd><kbd>D</kbd> change lanes (trains)</li>
        <li><kbd>↑</kbd> <kbd>W</kbd> or <kbd>Space</kbd> jump (hurdles)</li>
        <li><kbd>↓</kbd> or <kbd>S</kbd> slide (overhead bars)</li></ul>`}
      <button class="btn primary" id="go">${watch ? "Start" : "Start running"} <kbd>Enter</kbd></button>
    </div>`;
  $("go").onclick = startRun;
}

// ---------------------------------------------------------------- one game tick
function snapshot(g) {
  if (!g) return null;
  return { pz: g.pz, x: laneX(g.laneVisual()), h: jumpHeight(g.jt) / 100, slide: g.st > 0, alive: g.alive, jt: g.jt };
}

function tick() {
  const g = S.ghost;
  const outOfMoves = g.alive && g.t >= S.gActions.length;
  if (outOfMoves && !S.gOver) { fetchMore(); return false; }        // wait for the AI's next moves
  S.prev.me = S.snap.me; S.prev.ghost = S.snap.ghost;

  if (S.me && S.me.alive && S.phase === "running") {
    const a = S.input.shift() ?? C.NOOP;
    S.myActions.push(a);
    if (S.me.step(a)) myCrash();
  }
  if (g.alive && !outOfMoves) {                                     // (out of moves + over = the AI's 10 minutes are up)
    const a = S.gActions[g.t];
    S.scores = S.gScores[g.t] ?? null;
    S.lastAction = a;
    if (g.step(a)) {
      const where = fmt(g.distance());
      if (S.mode === "race" && S.me?.alive) S.toast = { text: `The AI hit ${CRASH_TEXT[crashName(g.crashType)]} at ${where}. Keep going!`, t: 0 };
      if (S.mode === "watch") { S.toast = { text: `Crashed into ${CRASH_TEXT[crashName(g.crashType)]} at ${where}`, t: 0 }; S.phase = "ending"; S.phaseT = 0; }
    }
  }
  if (S.gActions.length - g.t < 120) fetchMore();
  S.snap.me = snapshot(S.me); S.snap.ghost = snapshot(g);
  return true;
}

// ---------------------------------------------------------------- the main loop
function frame(now) {
  const dt = Math.min(100, now - (S.last || now));
  S.last = now;
  S.phaseT += dt;
  if (S.toast) S.toast.t += dt;
  S.shake = Math.max(0, S.shake - dt / 400);

  const countdown = S.mode === "race" ? 2400 : 300;
  if (S.phase === "countdown" && S.phaseT > countdown && S.sid) { S.phase = "running"; S.acc = 0; }
  const ticking = S.phase === "running" || S.phase === "ending";
  if (ticking) {
    S.acc += dt * (S.mode === "watch" ? S.speedMul : 1);
    while (S.acc >= C.TICK_MS) {
      if (!tick()) { S.acc = 0; break; }
      S.acc -= C.TICK_MS;
    }
    if (S.phase === "ending" && S.mode === "race" && S.phaseT > 1100 && S.ghost.alive) S.focus = "ghost";
    if (S.phase === "ending" && S.mode === "watch" && S.phaseT > 2200) startRun();
  }
  draw(ticking ? Math.min(1, S.acc / C.TICK_MS) : 1);
  hud();
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- drawing the world
const lerp = (a, b, t) => a + (b - a) * t;
function interp(k, t) {
  const a = S.prev[k], b = S.snap[k];
  if (!a || !b) return b;
  return { pz: lerp(a.pz, b.pz, t), x: lerp(a.x, b.x, t), h: lerp(a.h, b.h, t), slide: b.slide, alive: b.alive, jt: b.jt };
}

function hash(i, k = 0) {
  let x = (Math.imul(i + 7919 * k, 2654435761) ^ (S.seed + 1)) >>> 0;
  x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
  return x / 4294967296;
}

const BUILDING = ["#d9a679", "#c97b63", "#8fb3c9", "#b8c4a0", "#e3d3b0", "#9a8fb5", "#c9b8a6"];
const TRAINS = [["#2f6db5", "#f2c14e"], ["#c8423c", "#f2f2f2"], ["#3e9b6e", "#f2f2f2"], ["#6c5ba7", "#ffd166"]];

function draw(t) {
  const me = S.me ? interp("me", t) : null, gh = interp("ghost", t);
  const focus = S.focus === "me" && me ? me : gh;
  const world = S.focus === "me" && S.me ? S.me : S.ghost;      // whose obstacles to draw (they are the same track)
  const D = focus.pz;                                             // the camera follows this distance (cm)
  const zr = (z) => -(z - D) / 100;                               // world distance -> scene z (meters, ahead = negative)
  engine.scroll = D / 100;

  // camera: behind and above the runner, following its lane
  S.camX = lerp(S.camX ?? 0, focus.x, 0.12);
  const shake = S.shake * 0.25 * Math.sin(performance.now() / 25);
  const speed = (S.ghost.speed() - C.V0) / (C.VMAX - C.V0);
  // a tall screen (a phone held upright) gets a wider view, so all three lanes still fit
  const aspect = $("view").clientWidth / Math.max(1, $("view").clientHeight);
  let fov = 1.0 + speed * 0.15;
  if (aspect < 1.6) fov = 2 * Math.atan(Math.tan(fov / 2) * Math.min(1.6, 1.6 / aspect));
  engine.camera([S.camX + shake, 3.3, 6.2], [S.camX * 0.55, 0.9, -12], fov);

  // ground: three railway lanes, gravel between, sidewalks
  for (let l = 0; l < 3; l++) engine.box(laneX(l), -0.1, -50, 2.3, 0.2, 130, "#8a8178", { pattern: 3, color2: "#5b4a3c" });
  for (const x of [-1.25, 1.25]) engine.box(x, -0.12, -50, 0.2, 0.2, 130, "#a59b8f");
  for (const s of [-1, 1]) {
    engine.box(s * 4.85, 0.05, -50, 2.2, 0.3, 130, "#c9c4ba", { pattern: 4 });
    engine.box(s * 29, -0.2, -50, 46.5, 0.2, 130, "#9fb98c");
  }

  // buildings (one block every 14 m on each side)
  const first = Math.floor(D / 1400) - 1;
  for (let i = first; i < first + 10; i++) {
    for (const s of [-1, 1]) {
      const r = hash(i, s), w = 6 + hash(i, s + 5) * 4, h = 5 + r * 18;
      const z = zr(i * 1400) - 6;
      engine.box(s * (6.6 + w / 2), h / 2, z, w, h, 12, BUILDING[Math.floor(hash(i, s + 9) * BUILDING.length)],
        { pattern: 1, color2: "#5f7d95" });
    }
  }

  // obstacles
  for (const o of world.obs) {
    if (o.typ === 0) continue;
    const z0 = zr(o.z0), len = (o.z1 - o.z0) / 100, x = laneX(o.lane);
    if (z0 < -120 || z0 - len > 15) continue;
    if (o.typ === C.LOW) {
      for (const s of [-1, 1]) engine.box(x + s * 1.0, 0.38, z0 - 0.25, 0.1, 0.76, 0.1, "#eeeeee");
      engine.box(x, 0.62, z0 - 0.25, 2.1, 0.3, 0.12, "#d62828", { pattern: 2, color2: "#ffffff" });
    } else if (o.typ === C.HIGH) {
      for (const s of [-1, 1]) engine.box(x + s * 1.12, 1.15, z0 - 0.25, 0.14, 2.3, 0.14, "#3a3a3a");
      engine.box(x, 1.62, z0 - 0.25, 2.36, 0.46, 0.22, "#ffbe0b", { pattern: 2, color2: "#1d1d1d" });
    } else {
      const [body, stripe] = TRAINS[(o.z0 / 100 + o.lane) % TRAINS.length | 0];
      engine.box(x, 1.65, z0 - len / 2, 2.3, 2.9, len - 0.2, body, { pattern: 1, color2: "#cfe8f7" });
      engine.box(x, 0.85, z0 - 0.06, 2.32, 0.35, 0.06, stripe);
      engine.box(x, 2.2, z0 - 0.06, 1.7, 0.75, 0.05, "#1c2a38");
      engine.box(x, 0.12, z0 - len / 2, 1.9, 0.24, len - 0.6, "#2b2b2b");
    }
  }

  // runners: the ghost (transparent) and you
  if (gh && (gh.alive || S.mode === "watch" || S.focus === "ghost"))
    drawRunner(gh, zr(gh.pz), S.ghost.alive ? "#9b7bff" : "#6c6c80", S.mode === "watch" ? 1 : 0.45);
  if (me) drawRunner(me, zr(me.pz), me.alive ? "#ff7a1a" : "#a05a2c", 1);
  engine.render();
}

function drawRunner(r, z, color, alpha) {
  const x = r.x, y = r.h;
  const dist = r.pz / 100, ph = (dist / 1.5) * Math.PI * 2;
  const running = r.alive && r.h < 0.05 && !r.slide;
  let hip = 0.88, pitch = 0, l1 = 0, l2 = 0, a1 = 0, a2 = 0;
  if (!r.alive) { pitch = -0.9; hip = 0.5; l1 = 0.6; l2 = -0.3; a1 = -1.6; a2 = -1.4; }
  else if (r.slide) { hip = 0.32; pitch = 1.15; l1 = 1.45; l2 = 1.3; a1 = 0.6; a2 = 0.9; }      // feet first
  else if (r.h > 0.05) { l1 = -1.1; l2 = 0.4; a1 = -2.4; a2 = -2.2; }
  else if (running) { l1 = Math.sin(ph) * 0.75; l2 = -l1; a1 = -l1 * 0.9; a2 = l1 * 0.9; }
  const o = { alpha };
  const part = (dx, py, pz, sx, sy, sz, c, rx) => {
    // rotate (py, pz) around the hip by the body pitch (leaning back is positive)
    const cy = Math.cos(pitch), sn = Math.sin(pitch);
    const yy = py * cy - pz * sn, zz = py * sn + pz * cy;
    engine.box(x + dx, y + hip + yy, z + zz, sx, sy, sz, c, { ...o, rx: rx + pitch });
  };
  const limb = (dx, py, len, ang, w, c) => {                      // a limb hanging from a joint at height py
    const cy = py - Math.cos(ang) * len / 2, cz = -Math.sin(ang) * len / 2;
    part(dx, cy, cz, w, len, w, c, ang);
  };
  // legs (hang from the hip, not tilted with the body)
  for (const [dx, a] of [[-0.13, l1], [0.13, l2]]) {
    const cy = -Math.cos(a) * 0.42, cz = -Math.sin(a) * 0.42;
    engine.box(x + dx, y + hip + cy, z + cz, 0.18, 0.84, 0.2, "#24324a", { ...o, rx: a });
  }
  part(0, 0.36, 0, 0.5, 0.66, 0.3, color, 0);                       // body
  part(0, 0.88, 0, 0.34, 0.34, 0.34, "#f1c27d", 0);                  // head
  part(0, 1.02, 0.02, 0.38, 0.1, 0.38, color, 0);                    // cap
  limb(-0.33, 0.64, 0.62, a1, 0.13, color);                          // arms
  limb(0.33, 0.64, 0.62, a2, 0.13, color);
  if (r.alive) engine.box(x, 0.02, z, 0.9, 0.02, 0.7, "#000000", { alpha: 0.25 * alpha * Math.max(0.2, 1 - y / 1.4) });
}

// ---------------------------------------------------------------- the panels
const fmt = (m) => `${Math.floor(m).toLocaleString("en-US")} m`;

function hud() {
  const g = S.ghost;
  if (!g) return;
  const me = S.me;
  const main = S.mode === "race" && me ? me : g;
  $("dist").textContent = fmt(main.distance());
  $("spd").textContent = `${Math.round(main.speed() * 20 * 3.6 / 100)} km/h`;
  const aiLabel = S.info?.players.find((p) => p.id === S.model)?.label || "AI";
  $("aiName").textContent = aiLabel;
  $("aiDist").textContent = g.alive ? fmt(g.distance()) : `${fmt(g.distance())} - crashed`;
  $("aiChip").classList.toggle("down", !g.alive);
  $("hudYou").hidden = S.mode !== "race";

  if (S.phase === "countdown" && S.mode === "race") {
    const n = 3 - Math.floor(S.phaseT / 800);
    $("count").textContent = n > 0 ? n : "Go!";
    $("count").hidden = false;
  } else $("count").hidden = true;

  const toast = $("toast");
  if (S.toast && S.toast.t < 3500) { toast.textContent = S.toast.text; toast.hidden = false; }
  else toast.hidden = true;

  drawSees();
  drawThinking();
}

function drawSees() {
  const cv = $("sees"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, g = S.ghost;
  ctx.fillStyle = "#0e1a2b"; ctx.fillRect(0, 0, W, H);
  if (!g) return;
  const grid = g.observeGrid(), cw = W / 3, ch = (H - 24) / C.N_BINS;
  ctx.strokeStyle = "#22344f"; ctx.lineWidth = 1;
  for (let l = 1; l < 3; l++) { ctx.beginPath(); ctx.moveTo(l * cw, 0); ctx.lineTo(l * cw, H); ctx.stroke(); }
  const colors = ["#ff9f1c", "#ffd23f", "#3a86ff"];
  for (let c = 0; c < 3; c++) for (let b = 0; b < C.N_BINS; b++) for (let l = 0; l < 3; l++) {
    if (!grid[c][b][l]) continue;
    ctx.fillStyle = colors[c];
    ctx.fillRect(l * cw + 4, H - 24 - (b + 1) * ch + 1, cw - 8, ch - 1);
  }
  // the runner
  const x = (g.laneVisual() + 0.5) * cw;
  ctx.fillStyle = g.alive ? "#9b7bff" : "#666";
  ctx.beginPath(); ctx.arc(x, H - 12, 8, 0, Math.PI * 2); ctx.fill();
  if (g.jt > 0) { ctx.fillStyle = "#fff"; ctx.font = "10px system-ui"; ctx.textAlign = "center"; ctx.fillText("↑", x, H - 8); }
  if (g.st > 0) { ctx.fillStyle = "#fff"; ctx.font = "10px system-ui"; ctx.textAlign = "center"; ctx.fillText("↓", x, H - 8); }
}

function drawThinking() {
  const kind = S.info?.players.find((p) => p.id === S.model)?.kind;
  $("thinkKind").textContent = kind === "imitation" ? "probability a human would press it"
    : kind === "q" ? "Q value: expected future reward" : "rule-based bot: no network";
  const sc = S.scores;
  let html = "";
  for (let i = 0; i < 5; i++) {
    let w = 0, label = "";
    if (sc) {
      if (kind === "imitation") { w = sc[i]; label = `${Math.round(sc[i] * 100)}%`; }
      else {
        const lo = Math.min(...sc), hi = Math.max(...sc);
        w = hi - lo < 1e-6 ? 1 : 0.08 + 0.92 * (sc[i] - lo) / (hi - lo);
        label = sc[i].toFixed(2);
      }
    }
    const on = S.lastAction === i && S.ghost?.alive;
    html += `<div class="arow ${on ? "on" : ""}"><span>${ACTION_NAMES[i]}</span><div class="track"><i style="width:${(w * 100).toFixed(1)}%"></i></div><b>${label}</b></div>`;
  }
  $("think").innerHTML = html;
}

function renderBoard(b) {
  const s = b.summary;
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  $("recStats").textContent = s.runs
    ? `${plural(s.runs, "human run")} recorded (${s.minutes.toFixed(1)} minutes of play)`
    : "No human runs recorded yet";
  const h = b.head_to_head;
  $("h2h").textContent = h.races ? `Humans won ${h.human_wins} of ${plural(h.races, "race")} against the AI` : "";
  $("board").innerHTML = b.top.length
    ? b.top.map((r, i) => `<li><span>${i + 1}. ${escapeHtml(r.player)}</span><b>${fmt(r.distance)}</b></li>`).join("")
    : `<li class="empty">Play a run to get on the board</li>`;
}

$("downloadRuns").onclick = () => Backend.downloadRecordings();

const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------------------------------------------------------- input
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") {
    if (e.key === "Enter") e.target.blur();                         // Enter leaves the name field
    return;
  }
  if ((S.phase === "menu" || S.phase === "over") && (e.key === "Enter" || (e.key === " " && S.phase === "menu"))) {
    e.preventDefault(); startRun(); return;
  }
  const a = KEYS[e.key];
  if (a === undefined) return;
  e.preventDefault();
  if (S.phase === "running" && S.mode === "race" && !e.repeat && S.input.length < 3) S.input.push(a);
});

// ---------------------------------------------------------------- touch screens
// a swipe on the picture: the move counts as soon as the finger has gone 24 px (no need to lift it)
const press = (a) => { if (S.phase === "running" && S.mode === "race" && S.input.length < 3) S.input.push(a); };
let touch = null;
const stage = document.querySelector(".stage");
stage.addEventListener("touchstart", (e) => {
  if (e.target.closest("button, select, input")) return;
  touch = { x: e.touches[0].clientX, y: e.touches[0].clientY, used: false };
}, { passive: true });
stage.addEventListener("touchmove", (e) => {
  if (!touch) return;
  if (S.phase === "running") e.preventDefault();
  if (touch.used) return;
  const dx = e.touches[0].clientX - touch.x, dy = e.touches[0].clientY - touch.y;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
  touch.used = true;
  press(Math.abs(dx) > Math.abs(dy) ? (dx < 0 ? C.LEFT : C.RIGHT) : (dy < 0 ? C.JUMP : C.SLIDE));
}, { passive: false });
stage.addEventListener("touchend", () => { touch = null; });
stage.addEventListener("touchcancel", () => { touch = null; });
if (TOUCH) $("pad").hidden = false;
document.querySelectorAll("#pad button").forEach((b) => {
  b.addEventListener("pointerdown", (e) => { e.preventDefault(); press(+b.dataset.a); });
});

$("opponent").onchange = (e) => {
  S.model = e.target.value; localStorage.setItem("runner.model", S.model);
  if (S.mode === "watch" && S.phase !== "menu") startRun();
};
for (const b of document.querySelectorAll("#modes button")) {
  b.onclick = () => {
    S.mode = b.dataset.mode;
    document.querySelectorAll("#modes button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    $("speedBox").hidden = S.mode !== "watch";
    S.sid = null; showMenu();
  };
}
for (const b of document.querySelectorAll("#speed button")) {
  b.onclick = () => {
    S.speedMul = +b.dataset.v;
    document.querySelectorAll("#speed button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  };
}
$("name").value = localStorage.getItem("runner.name") || "";
$("name").oninput = () => localStorage.setItem("runner.name", $("name").value);
$("record").checked = localStorage.getItem("runner.record") !== "0";
$("record").onchange = () => localStorage.setItem("runner.record", $("record").checked ? "1" : "0");

// an idle scene behind the menu
S.seed = 12345; S.ghost = new Game(S.seed); S.snap.ghost = snapshot(S.ghost); S.focus = "ghost";
loadInfo();
requestAnimationFrame(frame);

// "Played N times by visitors" under the title (plays.js)
window.Plays?.badge("runner", document.querySelector("header h1"));
