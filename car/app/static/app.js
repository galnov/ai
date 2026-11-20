// =============================================================
//  Self-Driving Car - client side
//  Draws the world, asks the server to "advance N steps" and receives the state of the cars.
// =============================================================
const $ = (id) => document.getElementById(id);
const world = $("world"), wctx = world.getContext("2d");
const brainCv = $("brain"), bctx = brainCv.getContext("2d");

const RACE_LAPS = 3;
const TRAIL_LEN = 260;

const S = {
  info: null,
  track: null,
  trackLayer: null,     // static drawing of the track (separate canvas)
  mode: "watch",
  cars: [],             // description of the cars from the server
  state: null,          // the latest state
  trails: [],
  crashes: [],          // crash counter for each car
  bursts: [],           // crash animations
  running: false,
  paused: false,
  speed: 1,
  keys: { up: false, down: false, left: false, right: false },
  drawing: null,        // points while drawing
  race: null,
  scale: 1,
  busy: false,
};

// ---------------------------------------------------------------- network
async function post(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

// ---------------------------------------------------------------- init
async function init() {
  S.info = await (await fetch("/api/info")).json();
  const hasAI = S.info.drivers.some((d) => d.best);
  if (S.info.error) {
    $("banner").innerHTML = S.info.error.replace(/(python -m [\w.]+)/, "<code>$1</code>") +
      (hasAI ? "" : "\nMeanwhile you can draw tracks and drive yourself in race mode.");
    $("banner").hidden = false;
  }
  if (!hasAI) document.querySelector('[data-mode="watch"]').textContent = "Watch (no driver)";
  if (S.info.drivers.length < 2) document.querySelector('[data-mode="stages"]').disabled = true;

  $("presetSelect").innerHTML = S.info.presets.map((p) => `<option value="${p.id}">${p.name}</option>`).join("");
  setTrack(S.info.track);
  resize();
  setMode("watch");
  requestAnimationFrame(loop);
}

function setTrack(t) {
  S.track = t;
  S.trackLayer = renderTrack(t);
}

// ---------------------------------------------------------------- modes
function setMode(mode) {
  S.mode = mode;
  document.querySelectorAll("#modes button").forEach((b) => b.setAttribute("aria-selected", b.dataset.mode === mode));
  document.querySelectorAll("[data-show]").forEach((el) => (el.hidden = el.dataset.show !== mode));
  $("stage").classList.toggle("drawing", mode === "draw");
  $("results").hidden = true;
  $("countdown").hidden = true;
  $("drawError").hidden = true;
  S.race = null;
  if (mode === "race") {
    startCars("race", { n_ai: 1 }, false);   // wait for a click on "Start race"
  } else if (mode === "stages") {
    startCars("stages", {});
  } else {
    startCars("watch", { n_ai: +$("nCars").value });
  }
}

async function startCars(mode, opts, run = true) {
  const r = await post("/api/start", { mode, ...opts });
  S.cars = r.cars;
  S.trails = S.cars.map(() => []);
  S.crashes = S.cars.map(() => 0);
  S.bursts = [];
  S.state = await post("/api/step", { n: 0 });
  S.running = run;
  S.paused = false;
  $("pauseBtn").textContent = "Pause";
  updateTower();
}

// ---------------------------------------------------------------- animation loop
let lastT = performance.now(), acc = 0;
async function loop(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;
  if (S.running && !S.paused && !S.busy) {
    acc += dt * S.speed;
    let n = Math.floor(acc / S.info.dt);
    if (n > 0) {
      acc -= n * S.info.dt;
      n = Math.min(n, 12);
      S.busy = true;
      try {
        applyState(await post("/api/step", { n, keys: S.keys }));
      } catch (e) {
        console.error(e);
      }
      S.busy = false;
    }
  }
  drawWorld(now);
  requestAnimationFrame(loop);
}

function applyState(st) {
  S.state = st;
  st.cars.forEach((c, i) => {
    const tr = S.trails[i];
    tr.push([c.x, c.y, c.v]);
    if (tr.length > TRAIL_LEN) tr.shift();
  });
  st.crashes.forEach((e) => {
    S.crashes[e.car]++;
    S.bursts.push({ x: e.x, y: e.y, t0: performance.now(), color: S.cars[e.car].color });
    S.trails[e.car].push(null); // break the trail
  });
  if (S.race) checkRace(st);
  updateTowerThrottled();
  if (st.brain) drawBrain(st.brain);
}

// ---------------------------------------------------------------- race
async function startRace() {
  $("results").hidden = true;
  await startCars("race", { n_ai: 1 }, false);
  S.race = { finished: {}, started: false };
  for (const n of ["3", "2", "1", "GO!"]) {
    $("countdown").textContent = n;
    $("countdown").hidden = false;
    await new Promise((r) => setTimeout(r, n === "GO!" ? 500 : 800));
    if (S.mode !== "race") return;
  }
  $("countdown").hidden = true;
  S.race.started = true;
  S.running = true;
  world.focus();
}

function checkRace(st) {
  st.cars.forEach((c, i) => {
    if (c.laps >= RACE_LAPS && !(i in S.race.finished)) S.race.finished[i] = st.t;
  });
  const human = S.cars.findIndex((c) => c.kind === "human");
  const done = Object.keys(S.race.finished).length;
  if ((human >= 0 && human in S.race.finished) || done === S.cars.length) {
    if (S.race.shown) return;
    S.race.shown = true;
    const order = Object.entries(S.race.finished).sort((a, b) => a[1] - b[1]);
    const humanWon = order.length && +order[0][0] === human;
    $("resultsTitle").textContent = humanWon ? "You beat the AI!" : "The AI won this time";
    $("resultsList").innerHTML = order.map(([i, t]) =>
      `<li>${S.cars[i].label}<span>${t.toFixed(2)} s</span></li>`).join("") +
      S.cars.map((c, i) => (i in S.race.finished ? "" : `<li>${c.label}<span>still driving…</span></li>`)).join("");
    $("results").hidden = false;
  }
}

// ---------------------------------------------------------------- drawing the world
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const [W, H] = S.info.world;
  const cssW = world.clientWidth;
  world.width = Math.round(cssW * dpr);
  world.height = Math.round(cssW * (H / W) * dpr);
  S.scale = world.width / W;
  if (S.track) S.trackLayer = renderTrack(S.track);
  const bw = brainCv.clientWidth;
  brainCv.width = Math.round(bw * dpr);
  brainCv.height = Math.round(brainCv.clientHeight * dpr);
  if (S.state && S.state.brain) drawBrain(S.state.brain);   // resizing clears the canvas
}

function pathOf(ctx, pts, close = true) {
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  if (close) ctx.closePath();
}

function renderTrack(t) {
  const [W, H] = S.info.world;
  const cv = document.createElement("canvas");
  cv.width = world.width || W;
  cv.height = world.height || H;
  const ctx = cv.getContext("2d");
  const k = cv.width / W;
  ctx.scale(k, k);

  // grass with mowing stripes
  ctx.fillStyle = "#5f8f3a";
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = "#679a40";
  for (let x = -H; x < W; x += 90) {
    ctx.beginPath();
    ctx.moveTo(x, 0); ctx.lineTo(x + 45, 0); ctx.lineTo(x + 45 + H * 0.5, H); ctx.lineTo(x + H * 0.5, H);
    ctx.fill();
  }
  if (!t) return cv;

  // asphalt: the area between the two walls
  ctx.beginPath();
  pathOf(ctx, t.left);
  pathOf(ctx, t.right);
  ctx.fillStyle = "#3a3c40";
  ctx.fill("evenodd");

  // red-and-white kerbs
  for (const edge of [t.left, t.right]) {
    ctx.beginPath(); pathOf(ctx, edge);
    ctx.lineWidth = 7; ctx.strokeStyle = "#f4f4f0"; ctx.setLineDash([]); ctx.stroke();
    ctx.beginPath(); pathOf(ctx, edge);
    ctx.strokeStyle = "#e63946"; ctx.setLineDash([16, 16]); ctx.stroke();
  }
  ctx.setLineDash([]);

  // subtle direction arrows on the center line
  ctx.fillStyle = "rgba(255,255,255,0.10)";
  const step = Math.max(1, Math.round(t.center.length / 9));
  for (let i = Math.round(step / 2); i < t.center.length; i += step) {
    const a = t.center[i], b = t.center[(i + 2) % t.center.length];
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    ctx.save(); ctx.translate(a[0], a[1]); ctx.rotate(ang);
    ctx.beginPath(); ctx.moveTo(10, 0); ctx.lineTo(-8, -9); ctx.lineTo(-3, 0); ctx.lineTo(-8, 9); ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  // checkered start line
  const L = t.left[0], R = t.right[0];
  const len = Math.hypot(R[0] - L[0], R[1] - L[1]);
  const ang = Math.atan2(R[1] - L[1], R[0] - L[0]);
  ctx.save(); ctx.translate(L[0], L[1]); ctx.rotate(ang);
  const sq = 7, n = Math.ceil(len / sq);
  for (let i = 0; i < n; i++) for (let j = 0; j < 2; j++) {
    ctx.fillStyle = (i + j) % 2 ? "#111" : "#fff";
    ctx.fillRect(i * sq, -sq + j * sq, Math.min(sq, len - i * sq), sq);
  }
  ctx.restore();
  return cv;
}

function speedColor(v) {
  // blue (slow) -> yellow -> orange-red (fast)
  const h = 210 - 195 * Math.min(1, Math.max(0, v));
  return `hsl(${h}, 95%, ${55 + 10 * (1 - v)}%)`;
}

function drawWorld(now) {
  const k = S.scale;
  wctx.setTransform(1, 0, 0, 1, 0, 0);
  if (S.trackLayer) wctx.drawImage(S.trackLayer, 0, 0);
  wctx.setTransform(k, 0, 0, k, 0, 0);

  if (S.mode === "draw" && S.drawing && S.drawing.length > 1) {
    wctx.lineCap = "round"; wctx.lineJoin = "round";
    wctx.strokeStyle = "rgba(58,60,64,0.85)";
    wctx.lineWidth = +$("trackWidth").value;
    wctx.beginPath(); pathOf(wctx, S.drawing, false); wctx.stroke();
    wctx.strokeStyle = "#ffd23f"; wctx.lineWidth = 2; wctx.setLineDash([8, 8]);
    wctx.beginPath(); pathOf(wctx, S.drawing, false); wctx.stroke();
    wctx.setLineDash([]);
    return;
  }
  const st = S.state;
  if (!st || !st.cars) return;
  const cars = st.cars;

  // trail
  if ($("showTrails").checked) {
    wctx.lineWidth = 3; wctx.lineCap = "round";
    S.trails.forEach((tr) => {
      for (let i = 1; i < tr.length; i++) {
        if (!tr[i] || !tr[i - 1]) continue;
        wctx.globalAlpha = (i / tr.length) * 0.85;
        wctx.strokeStyle = speedColor(tr[i][2]);
        wctx.beginPath(); wctx.moveTo(tr[i - 1][0], tr[i - 1][1]); wctx.lineTo(tr[i][0], tr[i][1]); wctx.stroke();
      }
    });
    wctx.globalAlpha = 1;
  }

  // sensors
  if ($("showRays").checked) {
    const focus = focusIndex();
    cars.forEach((c, i) => {
      if (S.cars[i].kind !== "ai") return;
      const strong = i === focus;
      S.info.ray_angles.forEach((a, r) => {
        const d = c.rays[r] * S.info.ray_length;
        const ex = c.x + Math.cos(c.h + a) * d, ey = c.y + Math.sin(c.h + a) * d;
        const near = 1 - c.rays[r];
        wctx.strokeStyle = `hsla(${120 - 120 * near}, 90%, 60%, ${strong ? 0.85 : 0.3})`;
        wctx.lineWidth = strong ? 1.5 : 1;
        wctx.beginPath(); wctx.moveTo(c.x, c.y); wctx.lineTo(ex, ey); wctx.stroke();
        if (strong && c.rays[r] < 0.999) {
          wctx.fillStyle = wctx.strokeStyle;
          wctx.beginPath(); wctx.arc(ex, ey, 3.5, 0, Math.PI * 2); wctx.fill();
        }
      });
    });
  }

  // cars
  const [CL, CW] = S.info.car;
  cars.forEach((c, i) => {
    const meta = S.cars[i];
    wctx.save();
    wctx.translate(c.x, c.y);
    wctx.rotate(c.h);
    wctx.fillStyle = "rgba(0,0,0,0.35)";
    roundRect(wctx, -CL / 2 + 2, -CW / 2 + 2, CL, CW, 4); wctx.fill();
    wctx.fillStyle = "#151515";                     // wheels
    for (const [wx, wy] of [[-CL * 0.3, -CW / 2 - 1.5], [CL * 0.28, -CW / 2 - 1.5], [-CL * 0.3, CW / 2 - 1.5], [CL * 0.28, CW / 2 - 1.5]]) {
      wctx.fillRect(wx - 3, wy, 6, 3);
    }
    wctx.fillStyle = meta.color;
    roundRect(wctx, -CL / 2, -CW / 2, CL, CW, 4); wctx.fill();
    wctx.fillStyle = "rgba(20,30,40,0.75)";         // windshield
    roundRect(wctx, CL * 0.05, -CW / 2 + 2, CL * 0.22, CW - 4, 2); wctx.fill();
    wctx.restore();
    if (S.cars.length > 1 || meta.kind === "human") {
      label(meta.label, c.x, c.y - 18, meta.color);
    }
  });

  // crashes
  S.bursts = S.bursts.filter((b) => now - b.t0 < 700);
  S.bursts.forEach((b) => {
    const p = (now - b.t0) / 700;
    wctx.strokeStyle = `rgba(255,210,63,${1 - p})`;
    wctx.lineWidth = 4 * (1 - p) + 1;
    wctx.beginPath(); wctx.arc(b.x, b.y, 8 + 34 * p, 0, Math.PI * 2); wctx.stroke();
    wctx.fillStyle = `rgba(230,57,70,${0.8 * (1 - p)})`;
    for (let j = 0; j < 8; j++) {
      const a = (j / 8) * Math.PI * 2;
      wctx.beginPath(); wctx.arc(b.x + Math.cos(a) * 30 * p, b.y + Math.sin(a) * 30 * p, 3, 0, Math.PI * 2); wctx.fill();
    }
  });
}

function label(text, x, y, color) {
  wctx.font = "500 12px Heebo, system-ui, sans-serif";
  wctx.direction = "ltr";
  const w = wctx.measureText(text).width + 12;
  wctx.fillStyle = "rgba(29,31,34,0.85)";
  roundRect(wctx, x - w / 2, y - 9, w, 18, 9); wctx.fill();
  wctx.fillStyle = color;
  wctx.textAlign = "center"; wctx.textBaseline = "middle";
  wctx.fillText(text, x, y + 0.5);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ---------------------------------------------------------------- timing table
let towerT = 0;
function updateTowerThrottled() {
  const now = performance.now();
  if (now - towerT > 150) { towerT = now; updateTower(); }
}
function fmt(t) { return t == null ? "—" : t.toFixed(2); }
function updateTower() {
  const st = S.state;
  if (!st || !st.cars) { $("tower").innerHTML = ""; return; }
  const rows = st.cars.map((c, i) => ({ c, i, meta: S.cars[i] }))
    .sort((a, b) => b.c.progress - a.c.progress);
  $("tower").innerHTML = rows.map(({ c, i, meta }) => `
    <tr><td class="name"><span class="chip" style="background:${meta.color}"></span>${meta.label}</td>
    <td>${c.laps}</td><td>${fmt(c.lap_time)}</td><td class="best">${fmt(c.best_lap)}</td><td>${S.crashes[i]}</td></tr>`).join("");
}

// ---------------------------------------------------------------- the brain (network visualization)
const IN_LABELS = ["Left 90°", "Left 60°", "Left 35°", "Left 15°", "Ahead", "Right 15°", "Right 35°", "Right 60°", "Right 90°", "Speed"];

function drawBrain(b) {
  const dpr = window.devicePixelRatio || 1;
  const W = brainCv.width / dpr, H = brainCv.height / dpr;
  bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  bctx.clearRect(0, 0, W, H);
  const net = S.info.brains[b.driver];
  if (!net) return;
  $("brainName").textContent = S.cars[focusIndex()].label;

  // the flow goes from left (input) to right (output)
  const xs = [62, W * 0.38, W * 0.57, W - 92];
  const layerY = (n) => Array.from({ length: n }, (_, i) => 14 + (i + 0.5) * ((H - 28) / n));
  const y0 = layerY(10), y1 = layerY(net.w1.length), y2 = layerY(net.w2.length), y3 = layerY(net.w3.length);
  const a0 = b.obs, a1 = norm(b.h1), a2 = norm(b.h2);

  const edges = (W8, xa, ya, act, xb, yb) => {
    for (let j = 0; j < W8.length; j++) for (let i = 0; i < W8[j].length; i++) {
      const w = W8[j][i];
      const s = Math.min(1, Math.abs(w) * act[i] * 1.6);
      if (s < 0.04) continue;
      bctx.strokeStyle = w > 0 ? `rgba(59,206,172,${s})` : `rgba(255,107,107,${s})`;
      bctx.lineWidth = 0.4 + s * 1.6;
      bctx.beginPath(); bctx.moveTo(xa, ya[i]); bctx.lineTo(xb, yb[j]); bctx.stroke();
    }
  };
  edges(net.w1, xs[0], y0, a0, xs[1], y1);
  edges(net.w2, xs[1], y1, a1, xs[2], y2);
  edges(net.w3, xs[2], y2, a2, xs[3], y3);

  const nodes = (x, ys, acts, r) => ys.forEach((y, i) => {
    const a = Math.max(0, Math.min(1, acts[i]));
    bctx.fillStyle = `rgba(244,244,240,${0.12 + 0.88 * a})`;
    bctx.strokeStyle = "#555a61"; bctx.lineWidth = 1;
    bctx.beginPath(); bctx.arc(x, y, r, 0, Math.PI * 2); bctx.fill(); bctx.stroke();
  });
  nodes(xs[0], y0, a0, 5.5);
  nodes(xs[1], y1, a1, 5);
  nodes(xs[2], y2, a2, 5);

  bctx.font = "11px Heebo, system-ui, sans-serif";
  bctx.textBaseline = "middle";
  bctx.textAlign = "right";
  bctx.fillStyle = "#a5a9ae";
  y0.forEach((y, i) => bctx.fillText(IN_LABELS[i], xs[0] - 10, y));

  // output: Q values
  const q = b.q, qmin = Math.min(...q), qmax = Math.max(...q);
  y3.forEach((y, i) => {
    const chosen = i === b.action;
    const t = qmax > qmin ? (q[i] - qmin) / (qmax - qmin) : 1;
    bctx.fillStyle = chosen ? "#ffd23f" : "#383c42";
    bctx.beginPath(); bctx.arc(xs[3], y, chosen ? 8 : 6, 0, Math.PI * 2); bctx.fill();
    bctx.fillStyle = chosen ? "#ffd23f" : "#6b7077";
    bctx.fillRect(xs[3] + 14, y - 4, 50 * t, 8);
    bctx.textAlign = "left";
    bctx.fillStyle = chosen ? "#f4f4f0" : "#a5a9ae";
    bctx.fillText(S.info.actions[i], xs[3] + 14, y - 12);
  });
}
function focusIndex() {
  const best = S.cars.findIndex((c) => c.driver === "best");
  return best >= 0 ? best : S.cars.findIndex((c) => c.kind === "ai");
}
function norm(a) {
  const m = Math.max(1e-6, ...a);
  return a.map((v) => v / m);
}

// ---------------------------------------------------------------- drawing a track
function worldPoint(e) {
  const r = world.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width * S.info.world[0], (e.clientY - r.top) / r.height * S.info.world[1]];
}
world.addEventListener("pointerdown", (e) => {
  if (S.mode !== "draw") return;
  world.setPointerCapture(e.pointerId);
  S.drawing = [worldPoint(e)];
  S.running = false;
  S.state = null;
  S.trackLayer = renderTrack(null);
  $("drawError").hidden = true;
});
world.addEventListener("pointermove", (e) => {
  if (S.mode !== "draw" || !S.drawing) return;
  const p = worldPoint(e), last = S.drawing[S.drawing.length - 1];
  if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 6) S.drawing.push(p);
});
world.addEventListener("pointerup", async () => {
  if (S.mode !== "draw" || !S.drawing) return;
  const pts = S.drawing;
  S.drawing = null;
  try {
    const r = await post("/api/track", { points: pts, width: +$("trackWidth").value });
    setTrack(r.track);
    await startCars("watch", { n_ai: +$("nCars").value });
  } catch (err) {
    $("drawError").textContent = err.message;
    $("drawError").hidden = false;
    setTrack(S.track);
  }
});

// ---------------------------------------------------------------- events
document.querySelectorAll("#modes button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
$("presetSelect").onchange = async (e) => {
  const r = await post("/api/track", { preset: e.target.value });
  setTrack(r.track);
  setMode(S.mode === "draw" ? "watch" : S.mode);
};
$("randomBtn").onclick = async () => {
  const r = await post("/api/track", { random: true });
  setTrack(r.track);
  setMode("watch");
};
$("nCars").oninput = (e) => ($("nCarsOut").textContent = e.target.value);
$("nCars").onchange = () => setMode("watch");
$("trackWidth").oninput = (e) => ($("widthOut").textContent = e.target.value);
$("raceBtn").onclick = startRace;
$("againBtn").onclick = startRace;
$("pauseBtn").onclick = () => {
  S.paused = !S.paused;
  $("pauseBtn").textContent = S.paused ? "Resume" : "Pause";
};
$("restartBtn").onclick = () => (S.mode === "race" ? startRace() : setMode(S.mode));
document.querySelectorAll(".seg button").forEach((b) => (b.onclick = () => {
  S.speed = +b.dataset.speed;
  document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", x === b));
}));

const KEYMAP = { ArrowUp: "up", KeyW: "up", ArrowDown: "down", KeyS: "down", ArrowLeft: "left", KeyA: "left", ArrowRight: "right", KeyD: "right" };
function onKey(e, down) {
  const k = KEYMAP[e.code];
  if (!k || S.mode !== "race") return;
  e.preventDefault();
  S.keys[k] = down;
}
window.addEventListener("keydown", (e) => onKey(e, true));
window.addEventListener("keyup", (e) => onKey(e, false));
window.addEventListener("blur", () => Object.keys(S.keys).forEach((k) => (S.keys[k] = false)));
window.addEventListener("resize", () => { resize(); });

init();
