// =============================================================
//  A drone that learned to fly - client side
// =============================================================
const $ = (id) => document.getElementById(id);
const sky = $("sky"), ctx = sky.getContext("2d");
const hz = $("horizon"), hctx = hz.getContext("2d");

const MODES = {
  free: ["Free flight", "Click anywhere in the sky and the drone will fly there. Change the wind and gusts, or load a package mid-flight, and see how the network copes."],
  delivery: ["Delivery route", "Click to add points to the route (up to 8). The drone passes between them in order and returns to the start. Watch out for the buildings: the network doesn't \"see\" them, so the route has to go around them."],
  stages: ["Learning stages", "Each drone flies with the \"brain\" from a different point in training. Click to set a shared target."],
  swarm: ["Swarm show", "40 drones, each with a copy of the same network, and each with its own destination point. Choose a shape."],
};
const SHAPES = [["heart", "Heart"], ["circle", "Circle"], ["star", "Star"], ["wave", "Wave"], ["grid", "Grid"]];

const S = {
  info: null, st: null, prev: null, tSt: 0, interval: 0.04,
  speed: 1, paused: false, busy: false, mode: "free",
  payload: false, compare: true,
  bursts: [], floats: [], particles: [], value: null, wind: 0,
};

// the "server" is backend.js, in this page: the same addresses, answered without a network
const post = (url, body) => Backend.request(url, body);

// ---------------------------------------------------------------- init
async function init() {
  S.info = await Backend.request("/api/info");
  if (S.info.error) {
    $("banner").innerHTML = S.info.error.replace(/(python -m [\w.]+)/, "<code>$1</code>");
    $("banner").hidden = false;
  }
  if (!S.info.stages) document.querySelector('[data-mode="stages"]').disabled = true;
  $("valueBox").hidden = !S.info.has_value;
  for (let i = 0; i < 160; i++) S.particles.push([Math.random() * 20, Math.random() * 12, Math.random()]);
  resize();
  await setMode("free");
  updateWindLabels();
  requestAnimationFrame(loop);
}

async function setMode(mode) {
  S.mode = mode;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.mode === mode));
  $("modeTitle").textContent = MODES[mode][0];
  $("modeText").textContent = MODES[mode][1];
  $("hintFloat").textContent = { free: "Click to set a target", delivery: "Click to add a point to the route", stages: "Click to set a target", swarm: "Choose a shape on the side" }[mode];
  let extra = "";
  if (mode === "free") extra = `<label class="check"><input type="checkbox" id="cmp" ${S.compare ? "checked" : ""}> Add a PID drone for comparison</label>`;
  if (mode === "delivery") extra = `<div class="legend-row"><span>Deliveries completed:</span> <b class="big-num" id="deliv">0</b></div><button class="btn" id="clearRoute">Clear route</button>`;
  if (mode === "swarm") extra = `<div class="shapes">${SHAPES.map(([k, n]) => `<button class="btn" data-shape="${k}">${n}</button>`).join("")}</div>`;
  $("modeExtra").innerHTML = extra;
  if ($("cmp")) $("cmp").onchange = async (e) => { S.compare = e.target.checked; await post("/api/config", { compare_pid: S.compare }); await setMode("free"); };
  if ($("clearRoute")) $("clearRoute").onclick = () => post("/api/clear_route");
  document.querySelectorAll("[data-shape]").forEach((b) => (b.onclick = () => {
    document.querySelectorAll("[data-shape]").forEach((x) => x.classList.toggle("on", x === b));
    post("/api/shape", { name: b.dataset.shape });
  }));
  S.prev = null;
  S.st = await post("/api/mode", { mode });
  S.tSt = performance.now();
  if ($("valueMap").checked) loadValue();
}

// ---------------------------------------------------------------- loop
let last = performance.now(), acc = 0;
async function loop(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (!S.paused && !S.busy) {
    acc += dt * S.speed;
    let n = Math.floor(acc / S.info.dt);
    if (n > 0) {
      acc -= n * S.info.dt;
      n = Math.min(n, 25);
      S.busy = true;
      try {
        const st = await post("/api/step", { n });
        S.prev = S.st;
        S.st = st;
        S.tSt = performance.now();
        S.interval = (n * S.info.dt) / S.speed;
        onState(st);
      } catch (e) { console.error(e); }
      S.busy = false;
    }
  }
  draw(now, dt);
  requestAnimationFrame(loop);
}

function onState(st) {
  S.wind = st.air;
  for (const ev of st.events) {
    if (ev.type === "crash") S.bursts.push({ x: ev.x, y: ev.y, t0: performance.now() });
    if (ev.type === "delivered") S.floats.push({ x: ev.x, y: ev.y, t0: performance.now(), text: "📦 Delivered!" });
  }
  if ($("deliv")) $("deliv").textContent = st.deliveries;
  telemetry(st);
}

// ---------------------------------------------------------------- coordinates
function resize() {
  const dpr = window.devicePixelRatio || 1;
  sky.width = Math.round(sky.clientWidth * dpr);
  sky.height = Math.round(sky.clientHeight * dpr);
}
const K = () => sky.width / S.info.world[0];
const X = (x) => x * K();
const Y = (y) => sky.height - y * K();

// ---------------------------------------------------------------- drawing
function lerpState(alpha) {
  const a = S.prev, b = S.st;
  if (!a || a.mode !== b.mode || a.drones.x.length !== b.drones.x.length) return b.drones;
  const L = (p, q) => p.map((v, i) => v + (q[i] - v) * alpha);
  const th = a.drones.th.map((v, i) => {
    let d = b.drones.th[i] - v;
    if (d > Math.PI) d -= 2 * Math.PI;
    if (d < -Math.PI) d += 2 * Math.PI;
    return v + d * alpha;
  });
  return { ...b.drones, x: L(a.drones.x, b.drones.x), y: L(a.drones.y, b.drones.y), th };
}

function draw(now, dt) {
  if (!S.st) return;
  const W = sky.width, H = sky.height, k = K();
  // sky
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, "#5aa7e6"); g.addColorStop(0.75, "#cfe7f7"); g.addColorStop(1, "#eaf4fb");
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  // meter grid
  ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 1;
  for (let x = 0; x <= 20; x += 2) { ctx.beginPath(); ctx.moveTo(X(x), 0); ctx.lineTo(X(x), H); ctx.stroke(); }
  for (let y = 0; y <= 12; y += 2) { ctx.beginPath(); ctx.moveTo(0, Y(y)); ctx.lineTo(W, Y(y)); ctx.stroke(); }
  // distant skyline
  ctx.fillStyle = "rgba(90,120,150,0.16)";
  for (let i = 0; i < 26; i++) {
    const bx = i * 0.8, bh = 0.5 + ((i * 37) % 11) / 10;
    ctx.fillRect(X(bx), Y(bh), 0.7 * k, bh * k);
  }
  if (S.value && $("valueMap").checked) drawValue();
  // wind
  drawWind(dt);
  // ground
  ctx.fillStyle = "#7fa46a"; ctx.fillRect(0, Y(0.15), W, H - Y(0.15));
  ctx.fillStyle = "#5f8a4e"; ctx.fillRect(0, Y(0.15), W, 0.06 * k);
  // buildings (delivery)
  for (const [bx, bw, bh] of S.st.buildings) {
    ctx.fillStyle = "#2f3e55"; ctx.fillRect(X(bx), Y(bh), bw * k, bh * k);
    ctx.fillStyle = "#ffd98a";
    for (let wy = 0.5; wy < bh - 0.3; wy += 0.7) for (let wx = 0.3; wx < bw - 0.3; wx += 0.55) {
      if (((wx * 13 + wy * 7) | 0) % 3) ctx.fillRect(X(bx + wx), Y(wy + 0.35), 0.22 * k, 0.3 * k);
    }
  }
  // route
  const route = S.st.route;
  if (S.st.mode === "delivery" && route.length) {
    ctx.setLineDash([0.25 * k, 0.2 * k]); ctx.strokeStyle = "#14213d88"; ctx.lineWidth = 0.05 * k;
    ctx.beginPath(); route.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    if (route.length > 2) ctx.lineTo(X(route[0][0]), Y(route[0][1]));
    ctx.stroke(); ctx.setLineDash([]);
    route.forEach(([x, y], i) => {
      ctx.fillStyle = i === S.st.route_i ? "#ff7a45" : "#14213d";
      ctx.beginPath(); ctx.arc(X(x), Y(y), 0.28 * k, 0, 7); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = `700 ${0.3 * k}px Rubik`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(String(i + 1), X(x), Y(y) + 1);
    });
  }
  const alpha = Math.min(1, (now - S.tSt) / 1000 / Math.max(0.02, S.interval));
  const d = lerpState(alpha);
  const meta = S.st.meta;
  // targets
  if (S.st.mode === "swarm") {
    ctx.fillStyle = "rgba(20,33,61,0.18)";
    d.tx.forEach((tx, i) => { ctx.beginPath(); ctx.arc(X(tx), Y(d.ty[i]), 0.12 * k, 0, 7); ctx.fill(); });
  } else if (S.st.mode !== "delivery") {
    const seen = new Set();
    d.tx.forEach((tx, i) => {
      const key = `${tx},${d.ty[i]}`; if (seen.has(key)) return; seen.add(key);
      const pulse = 0.35 + 0.08 * Math.sin(now / 250);
      ctx.strokeStyle = "#14213d"; ctx.lineWidth = 0.05 * k;
      ctx.beginPath(); ctx.arc(X(tx), Y(d.ty[i]), pulse * k, 0, 7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X(tx - 0.6), Y(d.ty[i])); ctx.lineTo(X(tx + 0.6), Y(d.ty[i]));
      ctx.moveTo(X(tx), Y(d.ty[i] - 0.6)); ctx.lineTo(X(tx), Y(d.ty[i] + 0.6)); ctx.stroke();
    });
  }
  // trail
  if ($("trails").checked) S.st.trails.forEach((tr, i) => {
    if (tr.length < 2) return;
    ctx.strokeStyle = meta[i].color; ctx.lineWidth = 0.05 * k; ctx.globalAlpha = 0.5;
    ctx.beginPath(); tr.forEach(([x, y], j) => (j ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)))); ctx.stroke();
    ctx.globalAlpha = 1;
  });
  // drones
  const small = S.st.mode === "swarm";
  for (let i = 0; i < d.x.length; i++) drawDrone(d.x[i], d.y[i], d.th[i], d.t1[i], d.t2[i], meta[i], now, small);
  // animations
  S.bursts = S.bursts.filter((b) => now - b.t0 < 900);
  for (const b of S.bursts) {
    const p = (now - b.t0) / 900;
    for (let j = 0; j < 14; j++) {
      const a = (j / 14) * 6.28, r = (0.3 + 1.6 * p) * k;
      ctx.fillStyle = `rgba(${j % 2 ? "255,122,69" : "255,209,102"},${1 - p})`;
      ctx.beginPath(); ctx.arc(X(b.x) + Math.cos(a) * r, Y(b.y) - Math.sin(a) * r + p * p * 0.8 * k, 0.1 * k * (1 - p) + 2, 0, 7); ctx.fill();
    }
  }
  S.floats = S.floats.filter((f) => now - f.t0 < 1400);
  for (const f of S.floats) {
    const p = (now - f.t0) / 1400;
    ctx.globalAlpha = 1 - p; ctx.fillStyle = "#14213d"; ctx.font = `700 ${0.45 * k}px Rubik`; ctx.textAlign = "center";
    ctx.fillText(f.text, X(f.x), Y(f.y + 0.6 + p)); ctx.globalAlpha = 1;
  }
}

function drawDrone(x, y, th, t1, t2, meta, now, small) {
  const k = K(), s = small ? 0.55 : 1;
  ctx.save();
  ctx.translate(X(x), Y(y));
  ctx.rotate(-th);                      // positive θ = counter-clockwise (and on screen y is flipped)
  const arm = 0.42 * s * k;
  // air jets from the propellers - length by thrust
  for (const [dx, t] of [[-arm, t1], [arm, t2]]) {
    const len = (0.15 + 1.1 * t) * k * s;
    const gr = ctx.createLinearGradient(0, 0, 0, len);
    gr.addColorStop(0, "rgba(255,170,60,0.85)"); gr.addColorStop(1, "rgba(255,170,60,0)");
    ctx.fillStyle = gr;
    ctx.beginPath(); ctx.moveTo(dx - 0.1 * k * s, 0.04 * k); ctx.lineTo(dx + 0.1 * k * s, 0.04 * k);
    ctx.lineTo(dx + 0.05 * k * s, len); ctx.lineTo(dx - 0.05 * k * s, len); ctx.closePath(); ctx.fill();
  }
  // arms and body
  ctx.strokeStyle = "#14213d"; ctx.lineWidth = 0.06 * k * s;
  ctx.beginPath(); ctx.moveTo(-arm, -0.06 * k); ctx.lineTo(arm, -0.06 * k); ctx.stroke();
  ctx.fillStyle = meta.color;
  ctx.beginPath(); ctx.roundRect(-0.2 * k * s, -0.16 * k * s, 0.4 * k * s, 0.2 * k * s, 0.06 * k * s); ctx.fill();
  ctx.strokeStyle = "#14213d"; ctx.lineWidth = 0.03 * k; ctx.stroke();
  // spinning propellers
  for (const [dx, t] of [[-arm, t1], [arm, t2]]) {
    ctx.fillStyle = "#14213d"; ctx.fillRect(dx - 0.03 * k, -0.16 * k * s, 0.06 * k, 0.12 * k * s);
    const w = (0.2 + 0.08 * Math.abs(Math.sin(now / 30 + dx))) * k * s;
    ctx.fillStyle = `rgba(20,33,61,${0.25 + 0.4 * t})`;
    ctx.beginPath(); ctx.ellipse(dx, -0.18 * k * s, w, 0.03 * k * s + 1, 0, 0, 7); ctx.fill();
  }
  if (meta.pid) { ctx.fillStyle = "#fff"; ctx.font = `700 ${0.12 * k}px Rubik`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("PID", 0, -0.06 * k); }
  ctx.restore();
  if (meta.label && !small) {
    ctx.font = `500 ${Math.max(11, 0.26 * k)}px Rubik`; ctx.textAlign = "center"; ctx.direction = "ltr";
    const tw = ctx.measureText(meta.label).width + 12;
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.beginPath(); ctx.roundRect(X(x) - tw / 2, Y(y + 0.55) - 0.2 * k, tw, 0.36 * k, 0.18 * k); ctx.fill();
    ctx.fillStyle = "#14213d"; ctx.textBaseline = "middle"; ctx.fillText(meta.label, X(x), Y(y + 0.55) - 0.02 * k);
  }
}

function drawWind(dt) {
  const k = K(), w = S.wind;
  ctx.strokeStyle = "rgba(255,255,255,0.75)"; ctx.lineWidth = 1.2;
  for (const p of S.particles) {
    p[0] += w * dt * S.speed * (0.6 + 0.8 * p[2]);
    if (p[0] > 20.5) p[0] -= 21; if (p[0] < -0.5) p[0] += 21;
    if (Math.abs(w) < 0.3) continue;
    const len = Math.min(1.2, Math.abs(w) * 0.18) * k * Math.sign(w);
    ctx.globalAlpha = 0.25 + 0.5 * p[2];
    ctx.beginPath(); ctx.moveTo(X(p[0]), Y(p[1])); ctx.lineTo(X(p[0]) - len, Y(p[1])); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawValue() {
  const v = S.value, k = K();
  const cw = sky.width / v.nx, ch = sky.height / v.ny;
  for (let r = 0; r < v.ny; r++) for (let c = 0; c < v.nx; c++) {
    const t = v.v[r][c];
    ctx.fillStyle = `hsla(${250 - 200 * t}, 85%, ${45 + 15 * t}%, 0.32)`;
    ctx.fillRect(c * cw, sky.height - (r + 1) * ch, cw + 1, ch + 1);
  }
}

async function loadValue() {
  if (!S.info.has_value) return;
  S.value = await Backend.request("/api/value");
}

// ---------------------------------------------------------------- flight instruments
function telemetry(st) {
  const d = st.drones;
  if (!d.x.length) return;
  const i = 0;
  const vx = S.prev && S.prev.drones.x.length === d.x.length ? (d.x[i] - S.prev.drones.x[i]) / Math.max(1e-3, S.interval * S.speed) : 0;
  const vy = S.prev && S.prev.drones.y.length === d.y.length ? (d.y[i] - S.prev.drones.y[i]) / Math.max(1e-3, S.interval * S.speed) : 0;
  $("telName").textContent = st.meta[i].label;
  $("alt").textContent = `${d.y[i].toFixed(1)} m`;
  $("spd").textContent = `${Math.hypot(vx, vy).toFixed(1)} m/s`;
  $("tilt").textContent = `${Math.round(-d.th[i] * 180 / Math.PI)}°`;
  $("dist").textContent = `${Math.hypot(d.tx[i] - d.x[i], d.ty[i] - d.y[i]).toFixed(2)} m`;
  $("thrL").style.width = `${d.t1[i] * 100}%`;
  $("thrR").style.width = `${d.t2[i] * 100}%`;
  // artificial horizon
  const R = 60;
  hctx.save(); hctx.clearRect(0, 0, 120, 120);
  hctx.beginPath(); hctx.arc(R, R, R, 0, 7); hctx.clip();
  hctx.translate(R, R); hctx.rotate(d.th[i]);
  hctx.fillStyle = "#5aa7e6"; hctx.fillRect(-90, -90, 180, 90);
  hctx.fillStyle = "#a07850"; hctx.fillRect(-90, 0, 180, 90);
  hctx.strokeStyle = "#fff"; hctx.lineWidth = 2; hctx.beginPath(); hctx.moveTo(-90, 0); hctx.lineTo(90, 0); hctx.stroke();
  hctx.restore();
  hctx.strokeStyle = "#14213d"; hctx.lineWidth = 3;
  hctx.beginPath(); hctx.moveTo(25, 60); hctx.lineTo(50, 60); hctx.moveTo(70, 60); hctx.lineTo(95, 60); hctx.stroke();
  hctx.beginPath(); hctx.arc(60, 60, 3, 0, 7); hctx.fillStyle = "#14213d"; hctx.fill();
}

// ---------------------------------------------------------------- events
function updateWindLabels() {
  const w = +$("wind").value;
  $("windOut").textContent = w === 0 ? "None" : `${Math.abs(w)} m/s ${w > 0 ? "→ east" : "← west"}`;
  $("gustOut").textContent = +$("gust").value === 0 ? "None" : `strength ${$("gust").value}`;
}
$("wind").oninput = $("gust").oninput = updateWindLabels;
$("wind").onchange = () => post("/api/config", { wind: +$("wind").value }).then(() => $("valueMap").checked && loadValue());
$("gust").onchange = () => post("/api/config", { gust: +$("gust").value });
$("payload").onclick = () => {
  S.payload = !S.payload;
  $("payload").classList.toggle("on", S.payload);
  $("payload").textContent = S.payload ? "Drop the package" : "Load a package (0.5 kg)";
  post("/api/config", { payload: S.payload ? 0.5 : 0 });
};
document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
document.querySelectorAll("#speed button").forEach((b) => (b.onclick = () => {
  S.speed = +b.dataset.v;
  document.querySelectorAll("#speed button").forEach((x) => x.setAttribute("aria-pressed", x === b));
}));
$("pause").onclick = () => { S.paused = !S.paused; $("pause").textContent = S.paused ? "Resume" : "Pause"; };
$("valueMap").onchange = (e) => { if (e.target.checked) loadValue(); };
sky.addEventListener("pointerdown", async (e) => {
  if (S.mode === "swarm") return;
  const r = sky.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width * S.info.world[0];
  const y = (1 - (e.clientY - r.top) / r.height) * S.info.world[1];
  await post("/api/click", { x, y });
  if ($("valueMap").checked) loadValue();
});
window.addEventListener("resize", resize);

init();
