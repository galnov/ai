// =============================================================
//  Smart Intersection - client side
//  Asks the server to "advance N steps", and draws both intersections, the cameras and the charts.
// =============================================================
const $ = (id) => document.getElementById(id);

const VIEW_M = 62;            // how many meters on each side of the intersection are shown
const STOP = 9, LANE_OFF = 1.75, ROAD_HALF = 4.5;
const CAR_COLORS = ["#ebebeb", "#c8cacd", "#2a2a2f", "#1c2448", "#a0161c", "#1e3c8c", "#6e7076", "#3c5a46", "#d2b478", "#e6781e"];
const BUS_COLORS = ["#f0c828", "#ebebeb", "#2878c8"];
const TRUCK_COLORS = ["#ebebeb", "#b4281e", "#3c6ea0"];
const WIDTHS = [1.8, 2.5, 2.5];
const LIGHT = ["#2fd27a", "#ffb020", "#ff5a4e"];

const S = {
  info: null, state: null, prev: { left: new Map(), right: new Map() },
  tState: 0, interval: 0.5,
  speed: 4, paused: false, busy: false,
  camSeq: -1, camImgs: [], camData: null,
  history: [],
};

async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  return r.json();
}

// ---------------------------------------------------------------- init
async function init() {
  const info = (S.info = await (await fetch("/api/info")).json());
  const msgs = [];
  if (!info.has_agent) msgs.push("There is no trained agent yet, so the AI light uses the simple rule. To train: <code>python -m src.train_agent</code>");
  if (!info.has_counter) msgs.push("There is no trained camera network yet, so the AI uses a perfect sensor. To train: <code>python -m src.make_camera_data</code> then <code>python -m src.train_counter</code>");
  if (msgs.length) { $("banner").innerHTML = msgs.join("\n"); $("banner").hidden = false; }

  $("preset").innerHTML = info.presets.map((p) => `<option value="${p.id}">${p.name}</option>`).join("");
  $("weather").innerHTML = info.conditions.map((c) => `<button data-v="${c.id}">${c.name}</button>`).join("");
  const st = info.settings;
  $("preset").value = st.preset;
  $("leftMode").value = st.left_mode;
  $("ns").value = Math.round(st.custom[0] * 3600 / 50) * 50;
  $("ew").value = Math.round(st.custom[2] * 3600 / 50) * 50;
  updateCustomLabels();
  setSeg("weather", st.condition);
  setSeg("sensor", st.sensor);
  setSeg("speed", String(S.speed));
  if (!info.has_counter) document.querySelector('#sensor [data-v="camera"]').disabled = true;
  $("customBox").hidden = st.preset !== "custom";

  $("camwall").innerHTML = info.lanes.map((name, i) => `
    <div class="cam"><canvas width="192" height="336" id="cam${i}"></canvas>
      <span class="tag">${name}</span><span class="rec"></span>
      <div class="num"><b id="camPred${i}">—</b><small id="camTrue${i}"></small></div></div>`).join("");

  resize();
  await doReset();
  requestAnimationFrame(loop);
}

function setSeg(id, v) {
  document.querySelectorAll(`#${id} button`).forEach((b) => b.setAttribute("aria-pressed", b.dataset.v === v));
}

// ---------------------------------------------------------------- loop
let last = performance.now(), acc = 0;
async function loop(now) {
  const dt = Math.min(0.25, (now - last) / 1000);
  last = now;
  if (!S.paused && !S.busy) {
    acc += dt * S.speed;
    let n = Math.floor(acc / S.info.dt);
    if (n > 0) {
      acc -= n * S.info.dt;
      n = Math.min(n, 40);
      S.busy = true;
      try {
        const st = await post("/api/step", { n, cam_seq: S.camSeq });
        S.interval = (n * S.info.dt) / S.speed;
        applyState(st);
      } catch (e) { console.error(e); }
      S.busy = false;
    }
  }
  draw(now);
  requestAnimationFrame(loop);
}

function applyState(st) {
  for (const side of ["left", "right"]) {           // remember each vehicle's previous position - for smooth motion
    const m = new Map();
    const old = S.state ? S.state[side].cars : [];
    old.forEach((c) => m.set(c[3], c[1]));
    S.prev[side] = m;
  }
  S.state = st;
  S.tState = performance.now();
  st.history.forEach((h) => S.history.push(h));
  if (S.history.length > 450) S.history.splice(0, S.history.length - 450);
  if (st.camera) onCamera(st.camera);
  updateStats(st);
  if (st.brain) drawBrain(st.brain);
}

// ---------------------------------------------------------------- statistics
function updateStats(st) {
  const t = Math.floor(st.t);
  $("clock").textContent = `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`;
  const L = st.left, R = st.right;
  $("baseWait").textContent = L.avg_wait.toFixed(1);
  $("aiWait").textContent = R.avg_wait.toFixed(1);
  $("baseNow").textContent = L.waiting; $("aiNow").textContent = R.waiting;
  $("baseServed").textContent = L.served; $("aiServed").textContent = R.served;
  $("baseSw").textContent = L.switches; $("aiSw").textContent = R.switches;
  if (st.t > 60 && L.avg_wait > 0.5) {
    const p = Math.round((1 - R.avg_wait / L.avg_wait) * 100);
    $("pct").textContent = `${Math.abs(p)}%`;
    $("pct").classList.toggle("worse", p < 0);
    $("pctLabel").textContent = p >= 0 ? "less waiting time with the AI light" : "more waiting time with the AI light";
  } else {
    $("pct").textContent = "—";
    $("pctLabel").textContent = "Collecting data in the first minute…";
  }
}

// ---------------------------------------------------------------- drawing the intersections
function resize() {
  const dpr = window.devicePixelRatio || 1;
  for (const id of ["baseCanvas", "aiCanvas", "chart"]) {
    const c = $(id);
    c.width = Math.round(c.clientWidth * dpr);
    c.height = Math.round(c.clientHeight * dpr);
  }
}

function carRect(lane, s, len, w) {
  const a = STOP + s, b = STOP + s + len, h = w / 2;
  if (lane === 0) return [[-LANE_OFF - h, -b, -LANE_OFF + h, -a], [0, 1]];
  if (lane === 1) return [[LANE_OFF - h, a, LANE_OFF + h, b], [0, -1]];
  if (lane === 2) return [[a, -LANE_OFF - h, b, -LANE_OFF + h], [-1, 0]];
  return [[-b, LANE_OFF - h, -a, LANE_OFF + h], [1, 0]];
}

function draw(now) {
  if (!S.state) return;
  const alpha = Math.min(1, (now - S.tState) / 1000 / Math.max(0.03, S.interval));
  drawSide($("baseCanvas"), S.state.left, S.prev.left, alpha);
  drawSide($("aiCanvas"), S.state.right, S.prev.right, alpha);
  drawChart();
}

function drawSide(cv, side, prev, alpha) {
  const ctx = cv.getContext("2d");
  const W = cv.width, k = W / (2 * VIEW_M);
  ctx.setTransform(k, 0, 0, k, W / 2, W / 2);       // from now on we draw in meters, (0,0) at the center of the intersection
  const E = VIEW_M;
  ctx.fillStyle = "#16313b"; ctx.fillRect(-E, -E, 2 * E, 2 * E);
  // building blocks
  ctx.fillStyle = "#1b3a46";
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
    ctx.fillRect(sx > 0 ? 12 : -E, sy > 0 ? 12 : -E, E - 12, E - 12);
  }
  // sidewalks and roads
  const sw = ROAD_HALF + 3;
  ctx.fillStyle = "#2a4a55";
  ctx.fillRect(-sw, -E, 2 * sw, 2 * E); ctx.fillRect(-E, -sw, 2 * E, 2 * sw);
  ctx.fillStyle = "#283036";
  ctx.fillRect(-ROAD_HALF, -E, 2 * ROAD_HALF, 2 * E); ctx.fillRect(-E, -ROAD_HALF, 2 * E, 2 * ROAD_HALF);
  // center line
  ctx.strokeStyle = "#7d8c92"; ctx.lineWidth = 0.25; ctx.setLineDash([3, 3.5]);
  ctx.beginPath();
  ctx.moveTo(0, -E); ctx.lineTo(0, -STOP); ctx.moveTo(0, STOP); ctx.lineTo(0, E);
  ctx.moveTo(-E, 0); ctx.lineTo(-STOP, 0); ctx.moveTo(STOP, 0); ctx.lineTo(E, 0);
  ctx.stroke(); ctx.setLineDash([]);
  // crosswalks
  ctx.fillStyle = "#b9c4c7";
  for (let t = -ROAD_HALF + 0.3; t < ROAD_HALF; t += 1.2) {
    for (const c of [-6.5, 6.5]) { ctx.fillRect(t, c - 1.5, 0.6, 3); ctx.fillRect(c - 1.5, t, 3, 0.6); }
  }
  // stop lines colored by the traffic light
  const stops = [[-ROAD_HALF, -STOP - 0.4, ROAD_HALF, 0.8], [0, STOP - 0.4, ROAD_HALF, 0.8],
                 [STOP - 0.4, -ROAD_HALF, 0.8, ROAD_HALF], [-STOP - 0.4, 0, 0.8, ROAD_HALF]];
  side.lights.forEach((l, i) => {
    ctx.fillStyle = LIGHT[l];
    ctx.shadowColor = LIGHT[l]; ctx.shadowBlur = 12;
    ctx.fillRect(...stops[i]);
  });
  ctx.shadowBlur = 0;

  // cars
  const lens = S.info.vehicle_lengths;
  for (const [lane, s, kind, cid, waiting] of side.cars) {
    const p = prev.has(cid) ? prev.get(cid) : s;
    const sd = p + (s - p) * alpha;
    const len = lens[kind], w = WIDTHS[kind];
    const [[x0, y0, x1, y1], [dx, dy]] = carRect(lane, sd, len, w);
    if (Math.max(Math.abs(x0), Math.abs(y0)) > E + 15) continue;
    const pal = kind === 2 ? BUS_COLORS : kind === 1 ? TRUCK_COLORS : CAR_COLORS;
    ctx.fillStyle = pal[(cid * 7919) % pal.length];
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    ctx.fillStyle = "rgba(15,25,35,0.8)";                // windshield
    const fx = cx + dx * (len / 2 - 1.2), fy = cy + dy * (len / 2 - 1.2), hw = w / 2 - 0.25;
    if (dx === 0) ctx.fillRect(fx - hw, fy - 0.4, 2 * hw, 0.8); else ctx.fillRect(fx - 0.4, fy - hw, 0.8, 2 * hw);
    if (waiting) {                                       // brake lights
      ctx.fillStyle = "#ff3b30"; ctx.shadowColor = "#ff3b30"; ctx.shadowBlur = 6;
      const rx = cx - dx * (len / 2 - 0.2), ry = cy - dy * (len / 2 - 0.2);
      if (dx === 0) { ctx.fillRect(rx - w / 2 + 0.1, ry - 0.2, 0.5, 0.4); ctx.fillRect(rx + w / 2 - 0.6, ry - 0.2, 0.5, 0.4); }
      else { ctx.fillRect(rx - 0.2, ry - w / 2 + 0.1, 0.4, 0.5); ctx.fillRect(rx - 0.2, ry + w / 2 - 0.6, 0.4, 0.5); }
      ctx.shadowBlur = 0;
    }
  }

  // queue labels
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const dpr = window.devicePixelRatio || 1;
  ctx.font = `500 ${12 * dpr}px "IBM Plex Sans", sans-serif`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.direction = "ltr";
  const pos = [[-14, -50], [14, 50], [50, -14], [-50, 14]];
  side.queue.forEach((q, i) => {
    const [x, y] = pos[i];
    const X = (x + VIEW_M) * k, Y = (y + VIEW_M) * k;
    const txt = `${q} waiting`;
    const tw = ctx.measureText(txt).width + 12 * dpr;
    ctx.fillStyle = q >= 12 ? "rgba(255,90,78,0.9)" : "rgba(7,17,24,0.85)";
    ctx.beginPath(); ctx.roundRect(X - tw / 2, Y - 10 * dpr, tw, 20 * dpr, 10 * dpr); ctx.fill();
    ctx.fillStyle = "#e6f1f2"; ctx.fillText(txt, X, Y + 0.5);
  });
}

// ---------------------------------------------------------------- chart
function drawChart() {
  const cv = $("chart"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height, dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const h = S.history;
  if (h.length < 2) return;
  const pad = 26 * dpr;
  const maxY = Math.max(10, ...h.map((p) => Math.max(p[1], p[2]))) * 1.1;
  const t0 = h[0][0], t1 = h[h.length - 1][0];
  const X = (t) => pad + ((t - t0) / Math.max(1, t1 - t0)) * (W - 2 * pad);   // time flows from left to right
  const Y = (v) => H - pad - (v / maxY) * (H - 2 * pad);
  ctx.strokeStyle = "#22424f"; ctx.lineWidth = 1;
  ctx.fillStyle = "#8fb0b8"; ctx.font = `${11 * dpr}px "IBM Plex Sans", sans-serif`;
  ctx.direction = "ltr"; ctx.textAlign = "left";
  for (let v = 0; v <= maxY; v += maxY > 40 ? 20 : 10) {
    ctx.beginPath(); ctx.moveTo(pad, Y(v)); ctx.lineTo(W - pad, Y(v)); ctx.stroke();
    ctx.fillText(String(v), 2, Y(v) + 4);
  }
  for (const [i, color] of [[1, "#ffb020"], [2, "#34d1bf"]]) {
    ctx.strokeStyle = color; ctx.lineWidth = 2 * dpr; ctx.beginPath();
    h.forEach((p, j) => (j ? ctx.lineTo(X(p[0]), Y(p[i])) : ctx.moveTo(X(p[0]), Y(p[i]))));
    ctx.stroke();
  }
}

// ---------------------------------------------------------------- cameras
function onCamera(cam) {
  S.camSeq = cam.seq;
  S.camData = cam;
  cam.images.forEach((b64, i) => {
    const img = new Image();
    img.onload = () => drawCam(i, img);
    img.src = "data:image/jpeg;base64," + b64;
  });
  cam.true.forEach((t, i) => {
    if (cam.pred) {
      const p = cam.pred[i];
      $(`camPred${i}`).textContent = p.toFixed(1);
      const off = Math.abs(Math.round(p) - t) > 1;
      $(`camTrue${i}`).textContent = `actual ${t}`;
      $(`camTrue${i}`).className = off ? "off" : "";
    } else {
      $(`camPred${i}`).textContent = "—";
      $(`camTrue${i}`).textContent = `actual ${t}`;
    }
  });
}

function drawCam(i, img) {
  const cv = $(`cam${i}`), ctx = cv.getContext("2d");
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  const d = S.camData && S.camData.density;
  if (!d || !$("heat").checked) return;
  const m = d[i], ch = cv.height / m.length, cw = cv.width / m[0].length;
  for (let r = 0; r < m.length; r++) for (let c = 0; c < m[0].length; c++) {
    const v = m[r][c];
    if (v < 0.06) continue;
    ctx.fillStyle = `rgba(52,209,191,${Math.min(0.75, v * 0.9)})`;
    ctx.fillRect(c * cw, r * ch, cw, ch);
  }
}

// ---------------------------------------------------------------- the brain
function drawBrain(b) {
  const [g1, g2, r1, r2] = b.obs.slice(0, 4).map((v) => Math.round(v * 20));
  $("obs").innerHTML = `
    <div class="g"><small>Direction with green</small><b>${g1 + g2}</b> cars</div>
    <div class="r"><small>Direction waiting at red</small><b>${r1 + r2}</b> cars</div>`;
  const q = b.q, lo = Math.min(...q), hi = Math.max(...q);
  const names = ["Keep", "Switch"];
  $("qbars").innerHTML = q.map((v, i) => {
    const w = hi > lo ? 25 + 75 * (v - lo) / (hi - lo) : 60;
    return `<div class="qrow ${i === b.action ? "on" : ""}"><span>${names[i]}</span>
      <span class="track"><span class="fill" style="display:block;width:${w}%"></span></span><span class="v">${v.toFixed(1)}</span></div>`;
  }).join("");
}

// ---------------------------------------------------------------- settings
function updateCustomLabels() {
  $("nsOut").textContent = `${$("ns").value} per hour`;
  $("ewOut").textContent = `${$("ew").value} per hour`;
}
function sendCustom() {
  const ns = +$("ns").value / 3600, ew = +$("ew").value / 3600;
  post("/api/config", { custom: [ns, ns * 0.9, ew, ew * 0.9] });
}
async function doReset() {
  S.history = [];
  S.state = null;
  S.camSeq = -1;
  applyState(await post("/api/reset"));
}

$("preset").onchange = (e) => {
  $("customBox").hidden = e.target.value !== "custom";
  post("/api/config", { preset: e.target.value });
  if (e.target.value === "custom") sendCustom();
};
$("ns").oninput = $("ew").oninput = updateCustomLabels;
$("ns").onchange = $("ew").onchange = sendCustom;
$("leftMode").onchange = (e) => { post("/api/config", { left_mode: e.target.value }); doReset(); };
$("weather").onclick = (e) => {
  const v = e.target.dataset && e.target.dataset.v; if (!v) return;
  setSeg("weather", v); post("/api/config", { condition: v });
};
$("sensor").onclick = (e) => {
  const v = e.target.dataset && e.target.dataset.v; if (!v || e.target.disabled) return;
  setSeg("sensor", v); post("/api/config", { sensor: v });
};
$("speed").onclick = (e) => {
  const v = e.target.dataset && e.target.dataset.v; if (!v) return;
  S.speed = +v; setSeg("speed", v);
};
$("pause").onclick = () => { S.paused = !S.paused; $("pause").textContent = S.paused ? "Resume" : "Pause"; };
$("reset").onclick = doReset;
$("heat").onchange = () => { if (S.camData) onCamera(S.camData); };
window.addEventListener("resize", resize);

init();
