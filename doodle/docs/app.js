// =============================================================
//  Guess What I'm Drawing - client side
//  Collects the drawing's strokes, sends them to the network (running in this browser - see backend.js),
//  and shows the network's guesses. The static twin of app/static/app.js, for GitHub Pages.
// =============================================================
const $ = (id) => document.getElementById(id);
const pad = $("pad"), pctx = pad.getContext("2d");
const netView = $("netView"), nctx = netView.getContext("2d");

const GAME_ROUNDS = 6;
const ROUND_MS = 20000;
const SUCCESS_P = 0.35;      // enough if the top guess is right with reasonable confidence

const S = {
  cats: [], byKey: {},
  ready: false,
  strokes: [],               // each stroke: [[x...], [y...]]
  current: null,
  mode: "free",
  inflight: false, pending: false, lastSend: 0,
  lastTop: null,
  game: null,
};

// ---------------------------------------------------------------- init
async function init() {
  const info = await Backend.info();
  S.cats = info.categories;
  S.cats.forEach((c) => (S.byKey[c.key] = c));
  S.ready = info.ready;
  S.hasSamples = info.has_samples;
  if (!info.ready) {
    showBanner(`The network isn't ready yet: ${info.error}`);
  } else if (info.quick) {
    showBanner("The current model was trained in quick-test mode, so recognition is weak. For full training: <code>python -m src.train</code>");
  } else if (info.val_acc && info.val_acc < 0.6) {
    showBanner(`The model's accuracy is low (${(info.val_acc * 100).toFixed(0)}%). It's worth training it again, with more epochs or more data.`);
  }
  $("saveLabel").innerHTML = S.cats.map((c) => `<option value="${c.key}">${c.emoji} ${c.name}</option>`).join("");
  $("cats").innerHTML = S.cats.map((c, i) =>
    `<button class="cat" data-i="${i}" aria-pressed="false"><span class="em">${c.emoji}</span>${c.name}</button>`).join("");
  document.querySelectorAll(".cat").forEach((b) => (b.onclick = () => showSamples(+b.dataset.i)));
  resizePad();
  renderGuesses([]);
  drawNet(null);
}

function showBanner(html) {
  $("banner").innerHTML = html;
  $("banner").hidden = false;
}

// ---------------------------------------------------------------- drawing
function resizePad() {
  const dpr = window.devicePixelRatio || 1;
  const r = pad.getBoundingClientRect();
  pad.width = Math.round(r.width * dpr);
  pad.height = Math.round(r.height * dpr);
  pctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  redraw();
}

function redraw() {
  const r = pad.getBoundingClientRect();
  pctx.clearRect(0, 0, r.width, r.height);
  pctx.strokeStyle = "#1f2d5c";
  pctx.lineWidth = Math.max(4, r.width / 110);
  pctx.lineCap = "round";
  pctx.lineJoin = "round";
  const all = S.current ? [...S.strokes, S.current] : S.strokes;
  for (const [xs, ys] of all) {
    pctx.beginPath();
    pctx.moveTo(xs[0], ys[0]);
    if (xs.length === 1) pctx.lineTo(xs[0] + 0.1, ys[0]);
    for (let i = 1; i < xs.length; i++) pctx.lineTo(xs[i], ys[i]);
    pctx.stroke();
  }
}

function pos(e) {
  const r = pad.getBoundingClientRect();
  return [Math.round((e.clientX - r.left) * 10) / 10, Math.round((e.clientY - r.top) * 10) / 10];
}

pad.addEventListener("pointerdown", (e) => {
  if (S.game && S.game.locked) return;
  pad.setPointerCapture(e.pointerId);
  const [x, y] = pos(e);
  S.current = [[x], [y]];
  redraw();
});
pad.addEventListener("pointermove", (e) => {
  if (!S.current) return;
  const [x, y] = pos(e);
  const xs = S.current[0], ys = S.current[1];
  if (Math.hypot(x - xs[xs.length - 1], y - ys[ys.length - 1]) < 2) return;
  xs.push(x); ys.push(y);
  redraw();
  if (performance.now() - S.lastSend > 350) requestPredict();
});
const endStroke = () => {
  if (!S.current) return;
  S.strokes.push(S.current);
  S.current = null;
  redraw();
  requestPredict();
};
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);

function undo() {
  S.strokes.pop();
  redraw();
  requestPredict();
}
function clearPad() {
  S.strokes = [];
  S.current = null;
  S.lastTop = null;
  redraw();
  renderGuesses([]);
  drawNet(null);
  setBubble("✏️", "Draw something, and I'll guess what it is!", "");
}

// ---------------------------------------------------------------- guessing
async function requestPredict() {
  if (!S.ready) return;
  if (S.inflight) { S.pending = true; return; }
  const strokes = S.current ? [...S.strokes, S.current] : S.strokes;
  if (!strokes.length) { clearPad(); return; }
  S.inflight = true;
  S.lastSend = performance.now();
  try {
    const data = await Backend.predict(strokes);
    if (S.strokes.length || S.current) showPrediction(data);
  } catch (e) {
    setBubble("⚠️", "Something went wrong", e.message);
  }
  S.inflight = false;
  if (S.pending) { S.pending = false; requestPredict(); }
}

function showPrediction(data) {
  drawNet(data.image);
  renderGuesses(data.top);
  const [a, b] = data.top;
  if (!a) return;
  const ca = S.byKey[a.key], cb = b && S.byKey[b.key];
  if (a.p >= 0.6) {
    setBubble(ca.emoji, `It's ${art(ca)}!`, `I'm ${Math.round(a.p * 100)}% sure`);
  } else if (a.p >= 0.3) {
    setBubble(ca.emoji, `Maybe ${art(ca)}?`, cb ? `or maybe ${cb.emoji} ${art(cb)}…` : "");
  } else {
    setBubble("🤔", "Hmm… not sure yet", `Maybe ${ca.emoji} ${art(ca)}? Keep drawing`);
  }
  if (a.key !== S.lastTop && a.p >= 0.5) {
    S.lastTop = a.key;
    speak(a.p >= 0.6 ? `Oh, I know! It's ${art(ca)}!` : `Is it ${art(ca)}?`);
  }
  if (S.mode === "free") $("saveLabel").value = a.key;
  if (S.game) checkGame(data.top);
}

const low = (s) => s.toLowerCase();
const NO_ARTICLE = new Set(["scissors", "eyeglasses", "ice cream", "lightning"]);
// "a cat", "an apple", "scissors" - for natural sentences
const art = (c) => {
  const n = low(c.name);
  return NO_ARTICLE.has(n) ? n : (/^[aeiou]/.test(n) ? "an " : "a ") + n;
};

function setBubble(emoji, text, sub) {
  const em = $("bubbleEmoji");
  if (em.textContent !== emoji) {
    em.textContent = emoji;
    em.classList.remove("pop"); void em.offsetWidth; em.classList.add("pop");
  }
  $("bubbleText").textContent = text;
  $("bubbleSub").textContent = sub;
}

function renderGuesses(top) {
  if (!top.length) {
    $("guesses").innerHTML = `<li class="empty">The top 5 guesses will appear here, with the network's confidence.</li>`;
    return;
  }
  $("guesses").innerHTML = top.map((g) => {
    const c = S.byKey[g.key];
    return `<li class="guess"><span class="em">${c.emoji}</span>
      <span class="bar"><span class="fill" style="width:${(g.p * 100).toFixed(1)}%"></span><span class="name">${c.name}</span></span>
      <span class="pct">${Math.round(g.p * 100)}%</span></li>`;
  }).join("");
}

function drawNet(pixels) {
  const img = nctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = pixels ? pixels[i] : 0;
    img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
  }
  nctx.putImageData(img, 0, 0);
}

function speak(text) {
  if (!$("voice").checked || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-US";
  const v = speechSynthesis.getVoices().find((x) => x.lang && x.lang.startsWith("en"));
  if (v) u.voice = v;
  speechSynthesis.speak(u);
}

// ---------------------------------------------------------------- saving an example
async function saveSample() {
  if (!S.strokes.length) { $("savedMsg").textContent = "Draw something first"; return; }
  const label = $("saveLabel").value;
  try {
    const d = await Backend.save(S.strokes, label);
    $("savedCount").textContent = d.total;
    $("savedMsg").textContent = `Saved! You have ${d.saved} drawing${d.saved === 1 ? "" : "s"} labeled "${low(S.byKey[label].name)}", and ${d.total} in total.`;
    clearPad();
  } catch (e) {
    $("savedMsg").textContent = e.message;
  }
}

// ---------------------------------------------------------------- game
function startGame() {
  window.Plays?.played("doodle");                              // one more game, for the counter everyone sees
  $("gameStart").hidden = true;
  $("summary").hidden = true;
  const keys = S.cats.map((c) => c.key).sort(() => Math.random() - 0.5).slice(0, GAME_ROUNDS);
  S.game = { targets: keys, round: -1, results: [], locked: false };
  nextRound();
}

function nextRound() {
  const g = S.game;
  g.round++;
  clearPad();
  if (g.round >= g.targets.length) return endGame();
  const c = S.byKey[g.targets[g.round]];
  $("task").hidden = false;
  $("taskEmoji").textContent = c.emoji;
  $("taskName").textContent = c.name;
  $("taskRound").textContent = `${g.round + 1} of ${g.targets.length}`;
  g.locked = false;
  g.start = performance.now();
  g.lastGuess = null;
  tickTimer();
}

function tickTimer() {
  const g = S.game;
  if (!g || g.locked) return;
  const left = 1 - (performance.now() - g.start) / ROUND_MS;
  $("timerFill").style.width = `${Math.max(0, left) * 100}%`;
  $("timerFill").classList.toggle("low", left < 0.25);
  if (left <= 0) return finishRound(false);
  requestAnimationFrame(tickTimer);
}

function checkGame(top) {
  const g = S.game;
  if (!g || g.locked || !top.length) return;
  g.lastGuess = top[0].key;
  if (top[0].key === g.targets[g.round] && top[0].p >= SUCCESS_P) finishRound(true);
}

function finishRound(ok) {
  const g = S.game;
  g.locked = true;
  const c = S.byKey[g.targets[g.round]];
  g.results.push({
    key: c.key, ok, secs: (performance.now() - g.start) / 1000,
    guess: g.lastGuess, img: snapshot(),
  });
  const f = $("flash");
  f.className = "flash" + (ok ? "" : " fail");
  const guess = g.lastGuess && S.byKey[g.lastGuess];
  f.innerHTML = ok ? `Got it! It's ${art(c)} ${c.emoji}` :
    `Time's up…<br><small>${guess ? `I thought it was ${art(guess)} ${guess.emoji}` : "I didn't manage to guess"}</small>`;
  f.hidden = false;
  if (ok) speak(`Oh, I know! It's ${art(c)}!`);
  setTimeout(() => { f.hidden = true; if (S.game) nextRound(); }, 1600);
}

function snapshot() {
  const out = document.createElement("canvas");
  out.width = 240; out.height = 240;
  const ctx = out.getContext("2d");
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, 240, 240);
  ctx.drawImage(pad, 0, 0, 240, 240);
  return out.toDataURL("image/png");
}

function endGame() {
  const g = S.game;
  S.game = null;
  $("task").hidden = true;
  const score = g.results.filter((r) => r.ok).length;
  $("summaryTitle").textContent = `I recognized ${score} of ${g.results.length}!`;
  $("summaryGrid").innerHTML = g.results.map((r) => {
    const c = S.byKey[r.key], guess = r.guess && S.byKey[r.guess];
    const res = r.ok ? `recognized in ${r.secs.toFixed(1)} seconds` : (guess ? `I thought it was ${art(guess)}` : "not recognized");
    return `<div class="shot ${r.ok ? "ok" : "no"}"><img src="${r.img}" alt="Your drawing of ${art(c)}">
      <div class="what">${c.emoji} ${c.name}</div><div class="res">${r.ok ? "✓" : "✗"} ${res}</div></div>`;
  }).join("");
  $("summary").hidden = false;
  $("againBtn").focus();
}

function stopGame() {
  S.game = null;
  $("task").hidden = true;
  $("flash").hidden = true;
}

// ---------------------------------------------------------------- gallery
async function showSamples(i) {
  const c = S.cats[i];
  document.querySelectorAll(".cat").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.i === i));
  $("samples").hidden = false;
  $("samplesTitle").textContent = `${c.emoji} ${c.name}: how people drew it`;
  if (!S.hasSamples) {
    $("sampleGrid").innerHTML = `<p class="hint">No example drawings were published with this page.</p>`;
    return;
  }
  const d = await Backend.samples(i);
  $("sampleGrid").innerHTML = d.images.map(() => `<canvas width="28" height="28"></canvas>`).join("");
  $("sampleGrid").querySelectorAll("canvas").forEach((cv, k) => {
    const ctx = cv.getContext("2d"), img = ctx.createImageData(28, 28), px = d.images[k];
    for (let j = 0; j < 784; j++) {   // blue ink on white paper
      const v = px[j] / 255;
      img.data[j * 4] = 255 - v * 224; img.data[j * 4 + 1] = 255 - v * 210; img.data[j * 4 + 2] = 255 - v * 163;
      img.data[j * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  });
}

// ---------------------------------------------------------------- modes and events
function setMode(mode) {
  S.mode = mode;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.mode === mode));
  $("drawLayout").hidden = mode === "gallery";
  $("gallery").hidden = mode !== "gallery";
  $("teach").hidden = mode !== "free";
  stopGame();
  clearPad();
  if (mode === "game") $("gameStart").hidden = false;
  if (mode !== "gallery") requestAnimationFrame(resizePad);
}

document.querySelectorAll(".tabs button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
$("undoBtn").onclick = undo;
$("clearBtn").onclick = clearPad;
$("saveBtn").onclick = saveSample;
$("downloadBtn").onclick = () => Backend.downloadDrawings();
$("savedCount").textContent = Backend.savedCount();
$("goBtn").onclick = startGame;
$("againBtn").onclick = startGame;
$("closeSummary").onclick = () => { $("summary").hidden = true; setMode("free"); };
window.addEventListener("resize", resizePad);
window.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); }
  if (e.key === "Escape") { $("summary").hidden = true; $("gameStart").hidden = true; }
});

init();

// "Played N times by visitors" under the title (plays.js)
window.Plays?.badge("doodle", document.querySelector("header h1"));
