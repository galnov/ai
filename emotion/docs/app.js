// =============================================================
//  Emotion Mirror - client side
//  Captures frames from the webcam, analyzes them right here in the browser (backend.js), and draws the results.
// =============================================================

const $ = (id) => document.getElementById(id);
const video = $("video"), photo = $("photo"), overlay = $("overlay"), capture = $("capture");
const octx = overlay.getContext("2d"), cctx = capture.getContext("2d");

const CAPTURE_WIDTH = 640;      // width of the frame that is analyzed
const OVERLAY_SCALE = 2;        // resolution of the drawing over the video
const SMOOTHING = 0.45;         // smoothing: how much weight the new frame gets
const HISTORY_SECONDS = 30;

const state = {
  emotions: [],                 // [{key, he, emoji, color}]
  byKey: {},
  ready: false,
  mode: "idle",                 // idle | camera | photo
  stream: null,
  faces: [],                    // latest result of the analysis
  srcW: 1, srcH: 1,             // size of the image that was analyzed
  capW: CAPTURE_WIDTH, capH: 360,
  heatImgs: [],
  smooth: null,                 // smoothed probabilities (for the main face)
  rawTop: null,                 // latest unsmoothed probabilities
  history: [],                  // [{t, probs}]
  frameTimes: [],
  lastTop: null,
  challenge: null,
};

// ---------------------------------------------------------------- init
async function init() {
  try {
    const info = await Backend.info();
    state.emotions = info.emotions;
    state.emotions.forEach((e) => (state.byKey[e.key] = e));
    state.ready = info.ready;
    state.detector = info.detector;           // faces can be found even before a network is exported
    buildBars();
    buildLegend();
    if (info.ready) {
      setStatus("ok", `Model ready (runs ${info.device})`);
      if (info.val_acc) $("valAcc").textContent = `${(info.val_acc * 100).toFixed(1)}%`;
      const weak = (info.train_samples && info.train_samples < 10000) || (info.val_acc && info.val_acc < 0.5);
      if (weak) {
        const n = info.train_samples ? info.train_samples.toLocaleString() : "?";
        const acc = info.val_acc ? (info.val_acc * 100).toFixed(1) + "%" : "?";
        showBanner(`The current model is weak: trained on ${n} images for ${info.epoch ?? "?"} epochs, with validation accuracy ${acc}.\n` +
          `For full training: <code>python -m src.train</code>. FER-2013 has about 28,000 training images, and the expected accuracy is 63% to 68%.`);
      }
    } else {
      setStatus("err", info.detector ? "No trained network - finding faces only" : "Model not loaded");
      showBanner(info.error.replace(/(python -m [\w.]+)/g, "<code>$1</code>"));
    }
  } catch (e) {
    setStatus("err", "Could not load the model");
  }
  requestAnimationFrame(render);
}

function setStatus(kind, text) {
  $("status").className = "status " + kind;
  $("statusText").textContent = text;
}
function showBanner(html) {
  $("banner").innerHTML = html;
  $("banner").hidden = false;
}

function buildBars() {
  $("bars").innerHTML = state.emotions.map((e) => `
    <li class="bar" data-k="${e.key}">
      <span class="e">${e.emoji}</span><span class="n">${e.name}</span>
      <span class="track"><span class="fill" style="background:${e.color}"></span></span>
      <span class="v">0%</span>
    </li>`).join("");
}
function buildLegend() {
  $("legend").innerHTML = state.emotions.map((e) => `<span><i style="background:${e.color}"></i>${e.name}</span>`).join("");
}

// ---------------------------------------------------------------- camera
async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showBanner("This browser does not support camera access. Open this page in Chrome, Edge or Firefox (the camera needs an https:// or http://localhost address).");
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false,
    });
  } catch (e) {
    showBanner("No camera access. Allow the browser to use the camera and try again.");
    return;
  }
  video.srcObject = state.stream;
  await video.play();
  setMode("camera");
  setSize(video.videoWidth, video.videoHeight);
  captureLoop();
}

function stopCamera() {
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
  if (state.challenge) endChallenge(true);
  setMode("idle");
}

function setMode(mode) {
  state.mode = mode;
  state.faces = [];
  state.heatImgs = [];
  video.hidden = mode !== "camera";
  photo.hidden = mode !== "photo";
  $("empty").hidden = mode !== "idle";
  $("camBtn").textContent = mode === "camera" ? "Stop camera" : "Start camera";
  $("camBtn").classList.toggle("primary", mode !== "camera");
  $("snapBtn").disabled = mode === "idle";
  $("challengeBtn").disabled = mode !== "camera";
  applyMirror();
  if (mode === "idle") updatePanel(null);
}

function setSize(w, h) {
  state.capW = CAPTURE_WIDTH;
  state.capH = Math.round((CAPTURE_WIDTH * h) / w);
  capture.width = state.capW;
  capture.height = state.capH;
  overlay.width = state.capW * OVERLAY_SCALE;
  overlay.height = state.capH * OVERLAY_SCALE;
  $("stage").style.aspectRatio = `${w} / ${h}`;
  $("stage").style.maxWidth = `calc(74vh * ${(w / h).toFixed(4)})`;  // tall images won't overflow the screen
}

function mirrored() {
  return state.mode === "camera" && $("mirrorToggle").checked;
}
function applyMirror() {
  video.classList.toggle("mirrored", mirrored());
}

async function captureLoop() {
  while (state.mode === "camera") {
    // a phone that is turned changes the picture's shape: follow it, so faces are not squeezed
    if (video.videoWidth && Math.abs(video.videoHeight / video.videoWidth - state.capH / state.capW) > 0.02) setSize(video.videoWidth, video.videoHeight);
    cctx.drawImage(video, 0, 0, state.capW, state.capH);
    await analyze(capture, true);
    await new Promise((r) => setTimeout(r, 30));
  }
}

// ---------------------------------------------------------------- analysis (backend.js, in a background thread)
async function analyze(canvas, live = false) {
  if (!state.ready && !state.detector) return;
  const t0 = performance.now();
  try {
    const data = await Backend.predict(canvas, { gradcam: $("camToggle").checked, live });
    handleResult(data, performance.now() - t0);
  } catch (e) {
    setStatus("err", "Error: " + e.message);
    await new Promise((r) => setTimeout(r, 1000));
  }
}

function handleResult(data, ms) {
  state.faces = data.faces;
  state.srcW = data.width;
  state.srcH = data.height;
  state.heatImgs = data.faces.map((f) => {
    if (!f.heatmap) return null;
    const img = new Image();
    img.src = f.heatmap;
    return img;
  });

  const now = performance.now();
  state.frameTimes.push(now);
  state.frameTimes = state.frameTimes.filter((t) => now - t < 1000);
  $("fps").textContent = state.frameTimes.length;
  $("latency").textContent = Math.round(ms);
  $("faceCount").textContent = data.faces.length;

  const main = data.faces[0] && data.faces[0].probs ? data.faces[0] : null;
  if (!main && data.faces.length) {                    // a face was found, but there is no network to read it
    updatePanel(null);
    $("moodEmoji").textContent = "🙂";
    $("moodName").textContent = data.faces.length > 1 ? `${data.faces.length} faces found` : "Face found";
    $("moodConf").textContent = "Reading the emotion needs a trained network (see the note above)";
    return;
  }
  if (main) {
    const p = state.emotions.map((e) => main.probs[e.key]);
    state.rawTop = p;
    state.smooth = state.smooth && state.mode === "camera"
      ? state.smooth.map((v, i) => v * (1 - SMOOTHING) + p[i] * SMOOTHING)
      : p;
  } else {
    state.rawTop = null;
    if (state.smooth) state.smooth = state.smooth.map((v) => v * 0.8);
  }
  if (state.mode === "camera") {
    state.history.push({ t: now, probs: main ? state.smooth.slice() : null });
    state.history = state.history.filter((h) => now - h.t < HISTORY_SECONDS * 1000);
  }
  updatePanel(main ? state.smooth : null);
  if (state.challenge) challengeTick();
}

// ---------------------------------------------------------------- panel
function updatePanel(probs) {
  const bars = document.querySelectorAll(".bar");
  if (!probs) {
    $("moodEmoji").textContent = state.mode === "idle" ? "·" : "🔍";
    $("moodName").textContent = state.mode === "idle" ? "Waiting for a face" : "No face detected";
    $("moodConf").textContent = state.mode === "idle" ? "\u00a0" : "Look straight at the camera, in good light";
    document.documentElement.style.setProperty("--mood", "#8c9bab");
    bars.forEach((b) => {
      b.classList.remove("top");
      b.querySelector(".fill").style.width = "0%";
      b.querySelector(".v").textContent = "0%";
    });
    state.lastTop = null;
    return;
  }
  const top = probs.indexOf(Math.max(...probs));
  const e = state.emotions[top];
  if (state.lastTop !== e.key) {
    $("moodEmoji").classList.add("pop");
    setTimeout(() => $("moodEmoji").classList.remove("pop"), 180);
    state.lastTop = e.key;
  }
  $("moodEmoji").textContent = e.emoji;
  $("moodName").textContent = e.name;
  $("moodConf").textContent = `Confidence ${(probs[top] * 100).toFixed(0)}%`;
  document.documentElement.style.setProperty("--mood", e.color);
  bars.forEach((b, i) => {
    b.classList.toggle("top", i === top);
    b.querySelector(".fill").style.width = `${(probs[i] * 100).toFixed(1)}%`;
    b.querySelector(".v").textContent = `${Math.round(probs[i] * 100)}%`;
  });
}

// ---------------------------------------------------------------- drawing
function render() {
  drawOverlay();
  drawTimeline();
  requestAnimationFrame(render);
}

function drawOverlay() {
  octx.setTransform(1, 0, 0, 1, 0, 0);
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (state.mode === "idle") return;
  const k = OVERLAY_SCALE * (state.capW / state.srcW);
  octx.setTransform(k, 0, 0, k, 0, 0);
  const W = state.srcW, mir = mirrored();
  const lw = 1 / k;  // line width in canvas pixels

  state.faces.forEach((f, i) => {
    const e = f.top ? state.byKey[f.top] : { color: "#8c9bab", emoji: "🙂", name: "Face" };
    let [x, y, w, h] = f.box;
    if (mir) x = W - x - w;

    // heat map
    const img = state.heatImgs[i];
    if (img && img.complete) {
      let [cx, cy, cw, ch] = f.crop_box;
      if (mir) {
        octx.save();
        octx.translate(W - cx, cy);
        octx.scale(-1, 1);
        octx.drawImage(img, 0, 0, cw, ch);
        octx.restore();
      } else {
        octx.drawImage(img, cx, cy, cw, ch);
      }
    }

    // corner brackets
    const c = Math.min(w, h) * 0.22;
    octx.strokeStyle = e.color;
    octx.lineWidth = 5 * lw;
    octx.lineCap = "round";
    octx.beginPath();
    [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]].forEach(([px, py, sx, sy]) => {
      octx.moveTo(px + sx * c, py);
      octx.lineTo(px, py);
      octx.lineTo(px, py + sy * c);
    });
    octx.stroke();
    octx.lineWidth = 1.5 * lw;
    octx.globalAlpha = 0.35;
    octx.strokeRect(x, y, w, h);
    octx.globalAlpha = 1;

    // label
    const fs = Math.min(Math.max(w * 0.11, 15 * OVERLAY_SCALE * lw), 30 * OVERLAY_SCALE * lw);
    octx.font = `700 ${fs}px Assistant, system-ui, sans-serif`;
    octx.direction = "ltr";
    const label = f.top ? `${e.emoji} ${e.name} ${Math.round(f.confidence * 100)}%` : "Face found";
    const tw = octx.measureText(label).width;
    const padX = fs * 0.5, bh = fs * 1.55;
    const bx = x + w / 2 - tw / 2 - padX;
    const by = Math.max(0, y - bh - 8 * lw);
    octx.fillStyle = e.color;
    roundRect(octx, bx, by, tw + padX * 2, bh, bh / 2);
    octx.fill();
    octx.fillStyle = "#17153b";
    octx.textAlign = "center";
    octx.textBaseline = "middle";
    octx.fillText(label, x + w / 2, by + bh / 2 + fs * 0.05);

    // small meter under the face: the top 3 emotions
    if (!f.probs) return;
    const top3 = Object.entries(f.probs).sort((a, b) => b[1] - a[1]).slice(0, 3);
    const barH = Math.max(4, h * 0.035), gap = barH * 0.8;
    top3.forEach(([key, p], j) => {
      const yy = y + h + 10 * lw + j * (barH + gap);
      octx.fillStyle = "rgba(255,255,255,0.15)";
      roundRect(octx, x, yy, w, barH, barH / 2); octx.fill();
      octx.fillStyle = state.byKey[key].color;
      const bw = w * p;
      roundRect(octx, x + w - bw, yy, bw, barH, barH / 2); octx.fill();
    });
  });
}

function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, h / 2, w / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// stacked area chart: the share of each emotion at every moment
function drawTimeline() {
  const cv = $("timeline");
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  // grid lines
  ctx.strokeStyle = "rgba(163,159,208,0.15)";
  ctx.lineWidth = 1;
  for (let s = 0; s <= HISTORY_SECONDS; s += 5) {
    const x = W - (s / HISTORY_SECONDS) * W;  // the present is on the right, time flows to the left
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  ctx.fillStyle = "rgba(163,159,208,0.7)";
  ctx.font = "12px Assistant, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`${HISTORY_SECONDS} s ago`, 4, H - 4);
  ctx.textAlign = "right";
  ctx.fillText("now", W - 4, H - 4);

  const now = performance.now();
  const pts = state.history.filter((h) => h.probs);
  if (pts.length < 2) return;
  const xOf = (t) => W - ((now - t) / (HISTORY_SECONDS * 1000)) * W;
  const n = state.emotions.length;
  const plotH = H - 18;

  // when there's no face (a gap in time) - split into segments
  const segments = [];
  let seg = [];
  state.history.forEach((h) => {
    if (h.probs) seg.push(h); else if (seg.length) { segments.push(seg); seg = []; }
  });
  if (seg.length) segments.push(seg);

  segments.forEach((sg) => {
    if (sg.length < 2) return;
    let lower = sg.map(() => 0);
    for (let i = 0; i < n; i++) {
      const upper = sg.map((h, j) => lower[j] + h.probs[i]);
      ctx.beginPath();
      sg.forEach((h, j) => { const x = xOf(h.t), y = plotH - upper[j] * plotH; j ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      for (let j = sg.length - 1; j >= 0; j--) ctx.lineTo(xOf(sg[j].t), plotH - lower[j] * plotH);
      ctx.closePath();
      ctx.fillStyle = state.emotions[i].color;
      ctx.globalAlpha = 0.85;
      ctx.fill();
      lower = upper;
    }
    ctx.globalAlpha = 1;
  });
}

// ---------------------------------------------------------------- photo upload
function loadPhoto(file) {
  const img = new Image();
  img.onload = async () => {
    if (state.mode === "camera") stopCamera();
    const s = Math.min(1, 1280 / Math.max(img.width, img.height));
    photo.width = Math.round(img.width * s);
    photo.height = Math.round(img.height * s);
    photo.getContext("2d").drawImage(img, 0, 0, photo.width, photo.height);
    setMode("photo");
    setSize(photo.width, photo.height);
    state.smooth = null;
    await analyzePhoto();
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(file);
}
async function analyzePhoto() {
  await analyze(photo);
  if (!state.faces.length) $("moodConf").textContent = "Try a photo where the faces are large and clear";
}

// ---------------------------------------------------------------- snapshot
function snapshot() {
  const src = state.mode === "camera" ? video : photo;
  const w = state.mode === "camera" ? video.videoWidth : photo.width;
  const h = state.mode === "camera" ? video.videoHeight : photo.height;
  const out = document.createElement("canvas");
  out.width = w; out.height = h;
  const ctx = out.getContext("2d");
  if (mirrored()) { ctx.translate(w, 0); ctx.scale(-1, 1); }
  ctx.drawImage(src, 0, 0, w, h);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(overlay, 0, 0, w, h);
  const a = document.createElement("a");
  a.download = `emotion-${Date.now()}.png`;
  a.href = out.toDataURL("image/png");
  a.click();
}

// ---------------------------------------------------------------- challenge game
const CH_EMOTIONS = ["happy", "surprise", "angry", "sad", "fear", "neutral"];
const CH_ROUNDS = 5, CH_ROUND_MS = 5000, CH_PAUSE_MS = 1300;

function startChallenge() {
  window.Plays?.played("emotion");                             // one more game, for the counter everyone sees
  const pool = CH_EMOTIONS.slice().sort(() => Math.random() - 0.5).slice(0, CH_ROUNDS);
  const ch = (state.challenge = { targets: pool, round: -1, best: 0, scores: [], start: 0, pause: false, lb: null });
  // a leaderboard id for this game (this score can't be checked by the server - see counter/README.md)
  window.Leaderboard?.enabled && window.Leaderboard.start("emotion").then((r) => { if (r.id) ch.lb = r.id; });
  $("challenge").hidden = false;
  $("challengeBtn").disabled = true;
  nextRound();
}

function nextRound() {
  const ch = state.challenge;
  ch.round++;
  if (ch.round >= ch.targets.length) return endChallenge(false);
  ch.best = 0;
  ch.pause = false;
  ch.start = performance.now();
  const e = state.byKey[ch.targets[ch.round]];
  $("challenge").classList.remove("win");
  $("chRound").textContent = `Round ${ch.round + 1} of ${ch.targets.length}`;
  $("chEmoji").textContent = e.emoji;
  $("chText").textContent = `Show: ${e.name}!`;
  $("chMeter").style.width = "0%";
  $("chMeter").style.background = e.color;
  animateTimer();
}

function animateTimer() {
  const ch = state.challenge;
  if (!ch || ch.pause) return;
  const left = 1 - (performance.now() - ch.start) / CH_ROUND_MS;
  $("chTimer").style.width = `${Math.max(0, left) * 100}%`;
  if (left <= 0) return finishRound();
  requestAnimationFrame(animateTimer);
}

function challengeTick() {
  const ch = state.challenge;
  if (ch.pause || !state.rawTop) return;
  const idx = state.emotions.findIndex((e) => e.key === ch.targets[ch.round]);
  ch.best = Math.max(ch.best, state.rawTop[idx]);
  $("chMeter").style.width = `${ch.best * 100}%`;
}

function finishRound() {
  const ch = state.challenge;
  ch.pause = true;
  const pts = Math.round(ch.best * 100);
  ch.scores.push(pts);
  $("chText").textContent = pts >= 60 ? `Great! +${pts}` : pts >= 30 ? `Not bad, +${pts}` : `Tough… +${pts}`;
  $("challenge").classList.toggle("win", pts >= 60);
  setTimeout(() => state.challenge && nextRound(), CH_PAUSE_MS);
}

function endChallenge(aborted) {
  const ch = state.challenge;
  state.challenge = null;
  $("challenge").hidden = true;
  $("challengeBtn").disabled = state.mode !== "camera";
  if (aborted) return;
  const total = ch.scores.reduce((a, b) => a + b, 0);
  $("resultScore").textContent = `${total} / ${ch.targets.length * 100}`;
  $("resultList").innerHTML = ch.targets.map((k, i) => {
    const e = state.byKey[k];
    return `<li><span>${e.emoji} ${e.name}</span><strong>${ch.scores[i]}</strong></li>`;
  }).join("");
  $("offer").innerHTML = "";
  window.Leaderboard?.offer($("offer"), { id: ch.lb, getMoves: () => ({ score: total }), onDone: (r) => state.board?.set(r) });
  $("result").hidden = false;
  $("resultClose").focus();
}

// ---------------------------------------------------------------- events
$("camBtn").onclick = () => (state.mode === "camera" ? stopCamera() : startCamera());
// (the camera starts from the "Start camera" button under the picture)
$("fileInput").onchange = (e) => { if (e.target.files[0]) loadPhoto(e.target.files[0]); e.target.value = ""; };
$("snapBtn").onclick = snapshot;
$("challengeBtn").onclick = startChallenge;
$("resultClose").onclick = () => ($("result").hidden = true);
$("mirrorToggle").onchange = applyMirror;
$("camToggle").onchange = () => { if (state.mode === "photo") analyzePhoto(); };
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("result").hidden = true; });

// drag & drop a photo onto the stage
$("stage").addEventListener("dragover", (e) => e.preventDefault());
$("stage").addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) loadPhoto(f);
});

init();

// "Played N times by visitors" under the title (plays.js)
window.Plays?.badge("emotion", document.querySelector("header h1"));

// the leaderboard of the challenge (plays.js)
if (window.Leaderboard) state.board = window.Leaderboard.panel($("lbPanel"), { game: "emotion", title: "Leaderboard", sub: "Challenge points (5 faces)" });
