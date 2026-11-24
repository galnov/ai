// =====================================================================================
//  Maze Pilot - the client side.
//  Watch: the AI's flight (computed in backend.js) is played back from the cockpit.
//  Fly it yourself: the ship's physics runs here (ship.js), and the AI's flight in the same maze
//  is shown as a ghost to race against.
// =====================================================================================
const { MazeData, Ship, V } = window.MazeSim;
const { CockpitView, MapView, M } = window.Neon;
const $ = (id) => document.getElementById(id);
const TOUCH = matchMedia("(pointer: coarse)").matches || "ontouchstart" in window;   // a phone or a tablet

const S = {
  info: null, P: null, maze: null, mode: "watch", pilot: null, rooms: 4, seed: 1,
  flight: null, flightKey: "", ptime: 0, speedMul: 1, phase: "idle", phaseT: 0,
  ship: null, prevPose: null, pose: null, acc: 0, act: [0, 0, 0], throttle: -0.2, keys: {},
  trail: [], result: null, shake: 0, lastDepth: null, mapAngle: 0.6, frameMs: 16, humanVmax: 6,
};

let cockpit, map;
try {
  cockpit = new CockpitView($("view"));
  map = new MapView($("map"));
} catch (e) {
  $("overlay").innerHTML = `<div class="card"><h2>3D graphics are not available</h2><p>${e.message}</p></div>`;
  throw e;
}

// ---------------------------------------------------------------- the "server" (backend.js, in this page)

async function init() {
  S.info = await Backend.info();
  S.P = S.info.physics;
  $("pilot").innerHTML = S.info.pilots.map((p) => `<option value="${p.id}">${p.label}</option>`).join("");
  S.pilot = S.info.pilots[0].id;
  $("banner").hidden = !S.info.message;
  $("banner").textContent = S.info.message || "";
  await newMaze();
  requestAnimationFrame(frame);
}

async function newMaze(seed, daily = false) {
  S.daily = daily;                                     // the maze of the day (it has a leaderboard)
  S.seed = seed ?? Math.floor(Math.random() * 1e9);
  const j = await Backend.maze(S.seed, S.rooms);
  S.maze = new MazeData(j);
  cockpit.setMaze(S.maze);
  map.setMaze(S.maze);
  $("mazeInfo").textContent = `${S.daily ? "Maze of the day" : `Maze #${S.seed}`} · ${S.rooms}×${S.rooms}×${S.rooms} rooms · ${Math.round(S.maze.pathLength())} m to the exit`;
  S.flight = null;
  start();
}

async function loadFlight() {
  const key = `${S.seed}/${S.rooms}/${S.pilot}`;
  if (S.flight && S.flightKey === key) return S.flight;
  $("status").textContent = "Computing the AI's flight…";
  const f = await Backend.fly({ seed: S.seed, rooms: S.rooms, pilot: S.pilot });
  if (key !== `${S.seed}/${S.rooms}/${S.pilot}`) return null;           // the settings changed meanwhile
  S.flight = f; S.flightKey = key;
  $("status").textContent = "";
  return f;
}

// ---------------------------------------------------------------- starting a flight
async function start() {
  S.result = null; S.trail = []; S.ptime = 0; S.acc = 0; S.phaseT = 0; S.shake = 0;
  $("overlay").innerHTML = "";
  if (S.mode === "watch") {
    S.phase = "loading";
    S.pose = startPose();
    const f = await loadFlight();
    if (!f) return;
    S.phase = "flying";
  } else {
    S.ship = new Ship(S.maze, { ...S.P, v_max: S.humanVmax });          // your ship can be slower, for practice
    S.act = [0, 0, 0]; S.throttle = -0.3;
    S.pose = S.prevPose = shipPose(S.ship);
    S.phase = "ready";
    showReady();
    loadFlight();                                                     // the AI ghost, in the background
  }
}

function showReady() {
  const ai = S.P.v_max;
  $("overlay").innerHTML = `<div class="card ready">
    <h2>${S.daily ? "The maze of the day" : "Fly the maze yourself"}</h2>
    ${S.daily ? `<p class="note">The same maze for everyone today. Your flight counts for today's leaderboard.</p>` : ""}
    ${TOUCH ? `<ul class="keys">
      <li><b>Left thumb:</b> the joystick steers the nose (up, down, left, right)</li>
      <li><b>Right thumb:</b> the throttle bar (up = faster) · ⟲ ⟳ roll</li>
    </ul>
    <p class="note">Tip: turn the phone sideways for a bigger view.</p>` : `<ul class="keys">
      <li><kbd>↑</kbd><kbd>↓</kbd> nose toward the top / bottom of the screen</li>
      <li><kbd>←</kbd><kbd>→</kbd> or <kbd>A</kbd><kbd>D</kbd> turn toward the left / right of the screen</li>
      <li><kbd>W</kbd><kbd>S</kbd> more / less throttle</li>
      <li><kbd>Q</kbd><kbd>E</kbd> roll (optional - in space there is no "up")</li>
    </ul>`}
    <p class="long">Follow the yellow beacon: steer toward it on the screen. The white ring shows where the ship is really
       drifting, so brake before the turns. The cyan triangle is the AI flying the same maze.</p>
    <label class="field">Your top speed
      <select id="vmax">
        ${[5, 6, 7.5, ai].map((v) => `<option value="${v}" ${v === S.humanVmax ? "selected" : ""}>${v} m/s${v === ai ? " (same as the AI)" : v === 5 ? " (easiest)" : ""}</option>`).join("")}
      </select>
    </label>
    <button class="btn primary" id="go">Start <kbd>Space</kbd></button>
  </div>`;
  $("vmax").onchange = (e) => { S.humanVmax = +e.target.value; e.target.blur(); start(); };
  $("go").onclick = launch;
}

function launch() {
  if (S.mode !== "fly" || S.phase !== "ready") return;
  $("overlay").innerHTML = "";
  S.moves = [];
  S.lb = null;
  if (S.daily && window.Leaderboard?.enabled) {
    // the leaderboard server remembers when this flight started; it checks the flight later by replaying it
    const seed = S.seed;
    window.Leaderboard.start("maze", { vmax: S.humanVmax }).then((r) => { if (r.id && r.seed === seed) S.lb = r.id; });
  }
  window.Plays?.played("maze");                                 // one more flight, for the counter everyone sees
  if (TOUCH) { document.querySelector(".stage").scrollIntoView({ behavior: "smooth", block: "start" }); setThrottle(S.throttle); }
  S.phase = "countdown"; S.phaseT = 0;
}

function startPose() {
  const m = S.maze, f = m.nextDir(m.start).map(Number);
  return { pos: m.center(m.start), f, u: Math.abs(f[1]) < 0.5 ? [0, 1, 0] : [1, 0, 0], speed: 0 };
}
const shipPose = (s) => ({ pos: s.pos.slice(), f: s.f.slice(), u: s.u.slice(), speed: V.len(s.vel), vel: s.vel.slice() });

function lerpPose(a, b, t) {
  const f = V.norm(V.lerp(a.f, b.f, t));
  let u = V.lerp(a.u, b.u, t);
  u = V.norm(V.sub(u, V.mul(f, V.dot(u, f))));
  return { pos: V.lerp(a.pos, b.pos, t), f, u, speed: a.speed + (b.speed - a.speed) * t };
}

// the AI's pose at time t (s) of its recorded flight
function flightAt(t) {
  const fr = S.flight.frames, k = Math.min(Math.floor(t / S.P.dt), fr.length - 1);
  const a = fr[k], b = fr[Math.min(k + 1, fr.length - 1)];
  const pose = lerpPose(a, b, Math.min(1, t / S.P.dt - k));
  return { pose, k, frame: a, action: S.flight.actions[Math.min(k, S.flight.actions.length - 1)], over: k >= fr.length - 1 };
}

// ---------------------------------------------------------------- the main loop
function frame(now) {
  const dt = Math.min(0.1, (now - (S.last || now)) / 1000);
  S.frameMs = S.frameMs * 0.9 + (now - (S.last || now)) * 0.1;
  S.last = now;
  S.phaseT += dt;
  S.shake = Math.max(0, S.shake - dt * 2);
  S.mapAngle += dt * 0.15;
  let ghost = null;

  if (S.mode === "watch" && S.flight && (S.phase === "flying" || S.phase === "over")) {
    if (S.phase === "flying") S.ptime += dt * S.speedMul;
    const at = flightAt(S.ptime);
    S.pose = at.pose;
    S.lastDepth = at.frame.depth;
    S.act = at.action || [0, 0, 0];
    if (!S.trail.length || V.len(V.sub(S.trail[S.trail.length - 1], S.pose.pos)) > 1) S.trail.push(S.pose.pos);
    if (at.over && S.phase === "flying") finishWatch();
    if (S.phase === "over" && S.phaseT > 4 && $("auto").checked) newMaze();
  }

  const showTouch = TOUCH && S.mode === "fly" && (S.phase === "countdown" || S.phase === "flying");
  if ($("touch").hidden === showTouch) { $("touch").hidden = !showTouch; if (!showTouch) resetStick(); }
  if (S.mode === "fly") {
    if (S.phase === "countdown" && S.phaseT > 3) { S.phase = "flying"; S.phaseT = 0; S.acc = 0; }
    if (S.phase === "flying") {
      S.acc += dt;
      while (S.acc >= S.P.dt && S.phase === "flying") {
        S.acc -= S.P.dt;
        S.prevPose = shipPose(S.ship);
        // rounded to steps of 0.02 and recorded: the leaderboard server replays the flight from these numbers
        const a = manualAction().map((v) => Math.round(v * 50));
        S.moves.push(...a);
        S.ship.step(a.map((v) => v / 50));
        S.ptime += S.P.dt;
        S.trail.push(S.ship.pos.slice());
        if (S.ship.crashed || S.ship.reached) finishFly();
      }
      S.lastDepth = S.ship.castRays();
    }
    const cur = shipPose(S.ship);
    S.pose = S.phase === "flying" ? { ...lerpPose(S.prevPose, cur, S.acc / S.P.dt), vel: cur.vel } : cur;
    if (S.flight && S.phase !== "countdown") ghost = flightAt(S.phase === "over" ? S.result.aiT : S.ptime + S.acc);
  }

  draw(now / 1000, ghost);
  requestAnimationFrame(frame);
}

function manualAction() {
  const k = S.keys;
  const clamp1 = (v) => Math.max(-1, Math.min(1, v));
  const pitch = clamp1((k.ArrowUp ? 1 : 0) - (k.ArrowDown ? 1 : 0) + T.pitch);         // keyboard + joystick
  const yaw = clamp1((k.ArrowRight || k.d ? 1 : 0) - (k.ArrowLeft || k.a ? 1 : 0) + T.yaw);
  const roll = clamp1((k.e ? 1 : 0) - (k.q ? 1 : 0) + T.roll);
  if (k.w) S.throttle = Math.min(1, S.throttle + 0.12);
  if (k.s) S.throttle = Math.max(-1, S.throttle - 0.12);
  if (k.w || k.s) setThrottle(S.throttle);
  S.act = [S.act[0] + (pitch - S.act[0]) * 0.6, S.act[1] + (yaw - S.act[1]) * 0.6, S.throttle];
  return [...S.act, roll];
}

// ---------------------------------------------------------------- endings
const END_TEXT = { exit: "Reached the exit", crash: "Crashed into a wall", timeout: "Out of time" };

function finishWatch() {
  S.phase = "over"; S.phaseT = 0;
  const f = S.flight;
  if (f.end === "crash") S.shake = 1;
  $("overlay").innerHTML = `<div class="card result ${f.end}">
    <h2>${END_TEXT[f.end]}</h2>
    <p>${f.end === "exit" ? `${f.time.toFixed(1)} s · average speed ${f.speed.toFixed(1)} m/s` : `after ${Math.round(f.progress * 100)}% of the way`}</p>
    <div class="row"><button class="btn" id="replay">Replay</button><button class="btn primary" id="next">New maze</button></div>
  </div>`;
  $("replay").onclick = () => start();
  $("next").onclick = () => newMaze();
}

function finishFly() {
  S.phase = "over"; S.phaseT = 0;
  const s = S.ship, f = S.flight;
  if (s.crashed) S.shake = 1;
  const mine = s.reached ? "exit" : "crash";
  const aiT = f ? f.time : 0;
  S.result = { aiT: Math.min(S.ptime, aiT) };
  const slower = S.humanVmax < S.P.v_max ? `<p class="note">Your top speed was ${S.humanVmax} m/s; the AI's is ${S.P.v_max} m/s.</p>` : "";
  let verdict = "";
  if (f) {
    if (mine === "exit" && (f.end !== "exit" || s.t < f.time)) verdict = "You beat the AI!";
    else if (mine === "exit") verdict = `The AI was ${(s.t - f.time).toFixed(1)} s faster`;
    else verdict = f.end === "exit" ? "The AI made it out" : "The AI didn't make it either";
  }
  $("overlay").innerHTML = `<div class="card result ${mine}">
    <h2>${verdict || END_TEXT[mine]}</h2>
    <div class="vs">
      <div class="you"><small>You</small><b>${mine === "exit" ? s.t.toFixed(1) + " s" : Math.round(s.progress() * 100) + "%"}</b>
        <span>${mine === "exit" ? "reached the exit" : "crashed - progress"}</span></div>
      <div class="ai"><small>${f ? f.label : "AI"}</small><b>${!f ? "-" : f.end === "exit" ? f.time.toFixed(1) + " s" : Math.round(f.progress * 100) + "%"}</b>
        <span>${!f ? "" : f.end === "exit" ? "reached the exit" : END_TEXT[f.end].toLowerCase()}</span></div>
    </div>
    ${slower}
    <div id="offer"></div>
    <div class="row"><button class="btn" id="again">Try again <kbd>R</kbd></button><button class="btn primary" id="next">New maze <kbd>N</kbd></button></div>
  </div>`;
  $("again").onclick = () => start();
  $("next").onclick = () => newMaze();
  if (S.lb) {                                        // a crash counts too: by how far it got
    const moves = S.moves.slice();
    window.Leaderboard.offer($("offer"), { id: S.lb, getMoves: () => moves, onDone: (r) => S.board?.set(r) });
  } else if (S.daily) {
    $("offer").innerHTML = `<p class="note">This flight couldn't be checked by the leaderboard (no connection).</p>`;
  }
}

// ---------------------------------------------------------------- drawing
function remainingAt(p) {
  const m = S.maze, c = m.cellOf(p);
  if (m.isSolidCell(c)) return null;
  return m.dist[m.idx(...c)] * m.cell - V.dot(V.sub(p, m.center(c)), m.nextDir(c));
}

function draw(time, ghost) {
  const pose = S.pose;
  if (!pose) return;
  let cam = pose;
  if (S.shake > 0) {
    const j = S.shake * 0.08;
    cam = { ...pose, pos: V.add(pose.pos, [Math.sin(time * 61) * j, Math.cos(time * 53) * j, Math.sin(time * 47) * j]) };
  }
  // a tall picture (a phone held upright) gets a wider view, so the sides of the corridor stay visible
  const vw = $("view").clientWidth, vh = Math.max(1, $("view").clientHeight), aspect = vw / vh;
  const fov = aspect >= 1.3 ? 78 : (2 * Math.atan(Math.tan((39 * Math.PI) / 180) * Math.min(1.7, 1.3 / aspect)) * 180) / Math.PI;
  cockpit.render(cam, time, S.frameMs, fov);
  drawHud(pose, ghost, time);
  const ships = [{ pos: pose.pos, color: [0.3, 0.95, 1, 1], size: 11 }];
  if (ghost) ships.push({ pos: ghost.pose.pos, color: [1, 1, 1, 0.8], size: 8 });
  map.render(S.mapAngle, S.trail, ships);
  drawDepth();
  drawControls();
  drawStats(pose);
}

function drawHud(pose, ghost, time) {
  const cv = $("hud"), ctx = cv.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.round(cv.clientWidth * dpr), H = Math.round(cv.clientHeight * dpr);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  ctx.clearRect(0, 0, W, H);
  ctx.save(); ctx.scale(dpr, dpr);
  const w = W / dpr, h = H / dpr, cx = w / 2, cy = h / 2;
  const vp = cockpit.vp;
  const proj = (p) => { const q = M.project(vp, p, W, H); return q && [q[0] / dpr, q[1] / dpr]; };
  const green = "rgba(120,255,110,0.9)";

  // the reticle: where the nose points
  ctx.strokeStyle = green; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(cx, cy, 14, 0, Math.PI * 2);
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.moveTo(cx + dx * 20, cy + dy * 20); ctx.lineTo(cx + dx * 30, cy + dy * 30); }
  ctx.stroke();

  // the drift marker: where the ship is actually going
  const vel = pose.vel || (S.flight && S.mode === "watch" ? V.mul(pose.f, pose.speed) : null);
  if (vel && V.len(vel) > 0.5) {
    const q = proj(V.add(pose.pos, V.mul(V.norm(vel), 6)));
    if (q) {
      ctx.strokeStyle = "rgba(255,255,255,0.75)";
      ctx.beginPath(); ctx.arc(q[0], q[1], 6, 0, Math.PI * 2);
      ctx.moveTo(q[0] - 12, q[1]); ctx.lineTo(q[0] - 6, q[1]); ctx.moveTo(q[0] + 6, q[1]); ctx.lineTo(q[0] + 12, q[1]);
      ctx.moveTo(q[0], q[1] - 6); ctx.lineTo(q[0], q[1] - 11); ctx.stroke();
    }
  }

  // the beacon: the next cell on the way out
  const m = S.maze, c = m.cellOf(pose.pos);
  if (!m.isSolidCell(c)) {
    const b = m.center(V.add(c, m.nextDir(c)));
    marker(ctx, proj(b), w, h, cx, cy, "#ffe14d", "next", true);
  }
  if (ghost && !ghost.over) marker(ctx, proj(ghost.pose.pos), w, h, cx, cy, "#4df2ff", "AI", false);

  // speed and throttle
  const sp = pose.speed || 0, thr = (S.act[2] + 1) / 2;
  ctx.fillStyle = "rgba(0,0,0,0.45)"; ctx.fillRect(18, h - 128, 22, 110); ctx.fillRect(48, h - 128, 10, 110);
  ctx.fillStyle = green; ctx.fillRect(18, h - 18 - 110 * Math.min(1, sp / S.P.v_max), 22, 110 * Math.min(1, sp / S.P.v_max));
  ctx.fillStyle = "rgba(255,225,77,0.85)"; ctx.fillRect(48, h - 18 - 110 * thr, 10, 110 * thr);
  ctx.fillStyle = green; ctx.font = "600 13px system-ui"; ctx.textAlign = "left";
  ctx.fillText(`${sp.toFixed(1)} m/s`, 66, h - 22);
  ctx.fillStyle = "rgba(200,255,200,0.6)"; ctx.font = "11px system-ui";
  ctx.fillText("speed · throttle", 66, h - 40);

  if (S.phase === "countdown") {
    ctx.fillStyle = "#fff"; ctx.font = "800 96px system-ui"; ctx.textAlign = "center";
    ctx.fillText(String(Math.max(1, 3 - Math.floor(S.phaseT))), cx, cy + 34);
    if (S.daily) {
      ctx.shadowColor = "rgba(0,0,0,0.95)"; ctx.shadowBlur = 10;      // readable over the bright walls
      ctx.fillStyle = "#ffe14d"; ctx.font = "800 30px system-ui";
      ctx.fillText("Maze of the day", cx, cy - 70);
      ctx.fillStyle = "rgba(255,255,255,0.8)"; ctx.font = "15px system-ui";
      ctx.fillText("this flight counts for today's leaderboard", cx, cy + 74);
      ctx.shadowBlur = 0;
    }
  }
  if (S.daily && S.mode === "fly" && (S.phase === "flying" || S.phase === "over")) {
    ctx.shadowColor = "rgba(0,0,0,0.95)"; ctx.shadowBlur = 6;
    ctx.fillStyle = "#ffe14d"; ctx.font = "700 13px system-ui"; ctx.textAlign = "right";
    ctx.fillText("Maze of the day", w - 18, 28);
    ctx.shadowBlur = 0;
  }
  ctx.restore();
}

// a marker for a point in space; if it is behind or off screen, an arrow at the edge points to it
function marker(ctx, q, w, h, cx, cy, color, label, diamond) {
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
  const pad = 30;
  if (q && q[0] > pad && q[0] < w - pad && q[1] > pad && q[1] < h - pad) {
    ctx.beginPath();
    if (diamond) { ctx.moveTo(q[0], q[1] - 9); ctx.lineTo(q[0] + 9, q[1]); ctx.lineTo(q[0], q[1] + 9); ctx.lineTo(q[0] - 9, q[1]); ctx.closePath(); }
    else { ctx.moveTo(q[0], q[1] - 10); ctx.lineTo(q[0] + 8, q[1] + 6); ctx.lineTo(q[0] - 8, q[1] + 6); ctx.closePath(); }
    ctx.stroke();
    ctx.font = "600 11px system-ui"; ctx.textAlign = "center"; ctx.fillText(label, q[0], q[1] - 14);
    return;
  }
  // off screen: arrow at the edge
  let dx = q ? q[0] - cx : 0, dy = q ? q[1] - cy : 1;
  if (!q) { dx = 0; dy = h; }
  const a = Math.atan2(dy, dx), r = Math.min(w, h) / 2 - 40;
  const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r;
  ctx.save(); ctx.translate(x, y); ctx.rotate(a);
  ctx.beginPath(); ctx.moveTo(12, 0); ctx.lineTo(-6, -8); ctx.lineTo(-6, 8); ctx.closePath(); ctx.fill();
  ctx.restore();
}

function drawDepth() {
  const cv = $("depth"), ctx = cv.getContext("2d"), P = S.P;
  const d = S.lastDepth, cw = cv.width / P.rays_x, ch = cv.height / P.rays_y;
  ctx.fillStyle = "#020402"; ctx.fillRect(0, 0, cv.width, cv.height);
  if (!d) return;
  for (let i = 0; i < d.length; i++) {
    const near = Math.min(1, 3 / Math.max(0.1, d[i] * P.ray_range));       // 1 at 3 m or closer, 0.125 at 24 m
    ctx.fillStyle = `rgb(${Math.round(40 + 120 * near)},${Math.round(30 + 225 * near)},${Math.round(20 + 60 * near)})`;
    ctx.fillRect((i % P.rays_x) * cw + 1, Math.floor(i / P.rays_x) * ch + 1, cw - 2, ch - 2);
  }
}

function drawControls() {
  const names = ["Pitch", "Yaw", "Throttle"];
  const hints = [["down", "up"], ["left", "right"], ["slow", "fast"]];
  $("controls").innerHTML = S.act.map((v, i) => {
    const x = Math.max(-1, Math.min(1, v));
    const left = x < 0 ? 50 + x * 50 : 50, width = Math.abs(x) * 50;
    return `<div class="crow"><span>${names[i]}</span><div class="ctrack"><i style="left:${left}%;width:${width}%"></i><em></em></div>
      <small>${hints[i][0]}</small><small>${hints[i][1]}</small></div>`;
  }).join("");
}

function drawStats(pose) {
  const L = remainingAt(pose.pos), total = S.maze.pathLength();
  const t = S.mode === "watch" ? Math.min(S.ptime, S.flight ? S.flight.time : 0) : S.ptime;
  $("sTime").textContent = `${t.toFixed(1)} s`;
  $("sSpeed").textContent = `${(pose.speed || 0).toFixed(1)} m/s`;
  if (L !== null) {
    $("sLeft").textContent = `${Math.max(0, Math.round(L))} m`;
    $("sProg").style.width = `${Math.max(0, Math.min(100, (1 - L / total) * 100))}%`;
  }
}

// ---------------------------------------------------------------- controls
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT") return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "].includes(e.key)) e.preventDefault();
  S.keys[k] = true;
  if (S.mode === "fly" && S.phase === "ready" && (k === " " || k === "Enter")) launch();
  if (S.mode === "fly" && S.phase === "over") {
    if (k === "r") start();
    if (k === "n") newMaze();
  }
});
window.addEventListener("keyup", (e) => { const k = e.key.length === 1 ? e.key.toLowerCase() : e.key; S.keys[k] = false; });
window.addEventListener("blur", () => { S.keys = {}; });

// ---------------------------------------------------------------- touch controls
// the joystick: put the left thumb anywhere on the left part of the picture and push. The farther from where
// the thumb landed (up to 50 px), the harder the ship turns: up / down = the nose, left / right = turning.
const T = { pitch: 0, yaw: 0, roll: 0, stickId: null, thrId: null, x0: 0, y0: 0 };
const STICK_R = 50;
function moveKnob(dx, dy) { $("knob").style.transform = `translate(${dx}px, ${dy}px)`; }
function resetStick() {
  T.pitch = T.yaw = T.roll = 0; T.stickId = T.thrId = null;
  const st = $("stick"); st.classList.remove("active"); st.style.left = ""; st.style.top = ""; st.style.bottom = "";
  moveKnob(0, 0);
  $("rollL").classList.remove("on"); $("rollR").classList.remove("on");
}
$("stickZone").addEventListener("pointerdown", (e) => {
  e.preventDefault();
  T.stickId = e.pointerId; T.x0 = e.clientX; T.y0 = e.clientY;
  $("stickZone").setPointerCapture(e.pointerId);
  const r = $("stickZone").getBoundingClientRect(), st = $("stick");
  st.style.left = `${e.clientX - r.left - 60}px`; st.style.top = `${e.clientY - r.top - 60}px`; st.style.bottom = "auto";
  st.classList.add("active");
});
$("stickZone").addEventListener("pointermove", (e) => {
  if (e.pointerId !== T.stickId) return;
  let dx = e.clientX - T.x0, dy = e.clientY - T.y0;
  const d = Math.hypot(dx, dy);
  if (d > STICK_R) { dx *= STICK_R / d; dy *= STICK_R / d; }
  moveKnob(dx, dy);
  const dead = (v) => (Math.abs(v) < 0.12 ? 0 : v);         // a small dead zone, so "straight" is easy to hold
  T.yaw = dead(dx / STICK_R); T.pitch = dead(-dy / STICK_R);
});
for (const ev of ["pointerup", "pointercancel"]) {
  $("stickZone").addEventListener(ev, (e) => {
    if (e.pointerId !== T.stickId) return;
    T.stickId = null; T.pitch = T.yaw = 0;
    const st = $("stick"); st.classList.remove("active"); st.style.left = ""; st.style.top = ""; st.style.bottom = "";
    moveKnob(0, 0);
  });
}
// the throttle bar: the height of the right thumb on the bar sets the throttle
function setThrottle(v) { S.throttle = v; $("thrFill").style.height = `${((v + 1) / 2) * 100}%`; }
function thrFrom(e) {
  const r = $("thr").getBoundingClientRect();
  setThrottle(Math.max(-1, Math.min(1, 1 - (2 * (e.clientY - r.top)) / r.height)));
}
$("thr").addEventListener("pointerdown", (e) => { e.preventDefault(); T.thrId = e.pointerId; $("thr").setPointerCapture(e.pointerId); thrFrom(e); });
$("thr").addEventListener("pointermove", (e) => { if (e.pointerId === T.thrId) thrFrom(e); });
for (const ev of ["pointerup", "pointercancel"]) $("thr").addEventListener(ev, (e) => { if (e.pointerId === T.thrId) T.thrId = null; });
// roll: hold a button
for (const [id, dir] of [["rollL", -1], ["rollR", 1]]) {
  const b = $(id);
  b.addEventListener("pointerdown", (e) => { e.preventDefault(); b.setPointerCapture(e.pointerId); T.roll = dir; b.classList.add("on"); });
  for (const ev of ["pointerup", "pointercancel"]) b.addEventListener(ev, () => { if (T.roll === dir) T.roll = 0; b.classList.remove("on"); });
}

for (const b of document.querySelectorAll("#modes button")) {
  b.onclick = () => {
    S.mode = b.dataset.mode;
    document.querySelectorAll("#modes button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    $("watchOnly").hidden = S.mode !== "watch";
    $("flyHelp").hidden = S.mode !== "fly";
    b.blur();
    start();
    if (TOUCH && S.mode === "fly") document.querySelector(".stage").scrollIntoView({ behavior: "smooth", block: "start" });
  };
}
if (TOUCH) {                                                  // the help under the menus, for touch screens
  $("flyHelp").innerHTML = `<p>Left thumb: the joystick steers the nose. Right thumb: the throttle bar (up = faster). ⟲ ⟳ roll.</p>
    <p>Follow the yellow beacon. The white ring shows where you are really drifting. Race the AI ghost (cyan).</p>`;
}
for (const b of document.querySelectorAll("#speed button")) {
  b.onclick = () => {
    S.speedMul = +b.dataset.v;
    document.querySelectorAll("#speed button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  };
}
$("pilot").onchange = (e) => { S.pilot = e.target.value; e.target.blur(); S.flight = null; start(); };
$("rooms").onchange = (e) => { S.rooms = +e.target.value; e.target.blur(); newMaze(); };
$("newMaze").onclick = () => newMaze();

// the maze of the day: the same maze for everyone (4×4×4 rooms), with a leaderboard of the fastest flights
async function dailyMaze() {
  const r = await window.Leaderboard.top("maze");
  if (r.error || r.seed === undefined) { $("status").textContent = r.error || "No connection to the leaderboard"; return; }
  S.rooms = 4;
  $("rooms").value = "4";
  if (S.mode !== "fly") {                              // switch to flying yourself (newMaze starts the flight)
    S.mode = "fly";
    document.querySelectorAll("#modes button").forEach((x) => x.setAttribute("aria-pressed", x.dataset.mode === "fly"));
    $("watchOnly").hidden = true;
    $("flyHelp").hidden = false;
  }
  await newMaze(r.seed, true);
  S.board?.show(r.board, `Maze of ${r.board} · fastest to the exit, then the farthest crashes`);
}
if (window.Leaderboard) {
  S.board = window.Leaderboard.panel($("lbPanel"), { game: "maze", title: "Leaderboard: maze of the day", sub: "Fastest to the exit, then the farthest crashes" });
  $("daily").onclick = dailyMaze;
}

init();

// "Played N times by visitors" under the title (plays.js)
window.Plays?.badge("maze", document.querySelector("header h1"));
