// =====================================================================================
//  Pixel Breakout AI - the web page.
//  game.js: the game (a copy of src/game.py).  cnn.js: the trained convolutional networks.
// =====================================================================================
(() => {
  const { C, Board, seeded } = window.Breakout;
  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const COLORS = ["#e5484d", "#f08c2e", "#f2c94c", "#4cb782", "#3e8ed0"];   // one color per brick row
  const MOVES = [0, -1, 1];                                                 // stay, left, right

  const S = {
    mode: "race",            // "race" (you vs AI) | "watch" (the AI alone)
    phase: "menu",           // "menu" | "live" | "over"
    paused: false,
    you: null, ai: null,
    prevYou: null, prevAi: null,
    acc: 0,
    nets: null, net: null,
    info: null,              // the AI's last decision: { action, probs, cam }
    target: null,            // where you want your paddle (screen pixels)
    keys: { left: false, right: false },
    sound: true,
    sparks: [],              // little pieces of broken bricks
    restartAt: 0,
  };

  // ---------------------------------------------------------------- the networks
  async function loadModels() {
    let m = null;
    try { m = await window.CNN.load("model"); } catch (e) { /* no model yet */ }
    const sel = $("stage");
    if (m) {
      S.nets = m.nets;
      const list = m.meta.networks;
      const order = [...list.filter((n) => n.id === "final"), ...list.filter((n) => n.id !== "final").reverse()];
      sel.innerHTML = order.map((n) => `<option value="${n.id}">${n.label}</option>`).join("");
      sel.value = "final" in m.nets ? "final" : order[0].id;
    } else {
      sel.innerHTML = `<option value="none">Scripted player (no network yet)</option>`;
      showBanner("No trained network has been exported yet, so the AI is a simple scripted player. " +
        "Train in Colab, run  python -m src.export_web  and add docs/model/.");
    }
    pickNet();
  }

  function pickNet() {
    S.net = S.nets ? S.nets[$("stage").value] : null;
    const label = $("stage").options[$("stage").selectedIndex]?.text || "AI";
    $("aiName").textContent = S.net ? "AI: " + label.split(" (")[0] : "Scripted player";
  }

  function showBanner(t) { const b = $("banner"); b.textContent = t; b.hidden = false; }

  // the AI's decision for the board it plays
  function aiMove(board) {
    if (!S.net) {                                      // without a network: follow the ball
      const d = board.bx - board.px;
      S.info = null;
      return d < -1 ? -1 : d > 1 ? 1 : 0;
    }
    const r = S.net.act(board.observe());
    S.info = { action: r.action, probs: r.probs, cam: $("showCam").checked ? S.net.gradcam(r) : null };
    return MOVES[r.action];
  }

  // your move: the arrow keys, or toward the pointer (without overshooting it)
  function yourMove() {
    if (S.keys.left !== S.keys.right) return S.keys.left ? -1 : 1;
    if (S.target == null) return 0;
    return clamp((S.target - S.you.px) / (C.PADDLE_SPEED * C.DT), -1, 1);
  }

  // ---------------------------------------------------------------- the game
  const snap = (b) => ({ px: b.px, bx: b.bx, by: b.by });

  function newGame() {
    const seed = (Math.random() * 2 ** 31) | 0;          // both boards get the same serves
    S.you = new Board(seeded(seed));
    S.ai = new Board(seeded(seed));
    S.prevYou = snap(S.you);
    S.prevAi = snap(S.ai);
    S.sparks = [];
    S.acc = 0;
    S.target = null;
    S.phase = "live";
    S.paused = false;
    setPauseButton();
    hideOverlay();
    updateStats();
    if (S.mode === "race") window.Plays?.played("breakout");
  }

  function tick() {
    const race = S.mode === "race";
    if (race && !S.you.over) {
      S.prevYou = snap(S.you);
      const ev = S.you.step(yourMove());
      effects(ev, "you", true);
    }
    if (!S.ai.over) {
      S.prevAi = snap(S.ai);
      const ev = S.ai.step(aiMove(S.ai));
      effects(ev, "ai", !race);
    }
    updateStats();
    if (race) {
      const y = S.you, a = S.ai;
      // the race is decided when both are out, or one is out and the other already has more bricks
      if ((y.over && a.over) || (y.over && a.score > y.score) || (a.over && y.score > a.score)) endRace();
    } else if (S.ai.over && !S.restartAt) {
      S.restartAt = performance.now() + 2500;
      showOverlay(`<div class="card"><h2>Game over</h2><div class="final"><span class="a">${S.ai.score}</span></div>
        <p>bricks with 3 balls. A new game starts in a moment.</p></div>`, false);
    }
  }

  function effects(ev, who, loud) {
    for (const [r, c] of ev.bricks) {
      for (let i = 0; i < 6; i++) {
        S.sparks.push({ who, x: c * C.BRICK_W + 2.5, y: C.BRICK_TOP + r * C.BRICK_H + 1, vx: (Math.random() - 0.5) * 30,
          vy: (Math.random() - 0.2) * 25, life: 0.5, color: COLORS[r] });
      }
      if (loud) sfx("brick", r);
    }
    if (loud && ev.paddle) sfx("paddle");
    if (loud && ev.lost) sfx("lost");
  }

  function endRace() {
    S.phase = "over";
    const y = S.you.score, a = S.ai.score;
    const title = y > a ? "You win!" : y < a ? "The AI wins" : "A tie";
    const text = y > a ? "You broke more bricks than the network. Try a stronger stage?"
      : "It learned this from pixels alone. Another round?";
    setTimeout(() => showOverlay(`<div class="card"><h2>${title}</h2>
      <div class="final"><span class="y">${y}</span> : <span class="a">${a}</span></div>
      <p>${text}</p><button class="btn primary" id="again">Play again</button></div>`), 600);
  }

  function updateStats() {
    const dots = (n) => "●".repeat(Math.max(0, n)) + "○".repeat(Math.max(0, C.LIVES - n));
    const bricks = (n) => `${n} ${n === 1 ? "brick" : "bricks"}`;
    $("youScore").textContent = bricks(S.you ? S.you.score : 0);
    $("aiScore").textContent = bricks(S.ai ? S.ai.score : 0);
    $("youLives").textContent = dots(S.you ? S.you.lives : C.LIVES);
    $("aiLives").textContent = dots(S.ai ? S.ai.lives : C.LIVES);
  }

  function showOverlay(html, focus = true) {
    const o = $("overlay");
    o.innerHTML = html;
    o.classList.add("on");
    const b = o.querySelector("#again, #start");
    if (b) {
      b.addEventListener("click", () => { unlockAudio(); newGame(); });
      if (focus) b.focus({ preventScroll: true });
    }
  }
  function hideOverlay() { $("overlay").classList.remove("on"); S.restartAt = 0; }

  function menu() {
    S.phase = "menu";
    S.you = new Board(seeded(1));
    S.ai = new Board(seeded(1));
    S.prevYou = snap(S.you);
    S.prevAi = snap(S.ai);
    updateStats();
    showOverlay(`<div class="card"><h2>Race a network that plays from pixels</h2>
      <p>You both get the same bricks and the same serves, and 3 balls each. Whoever breaks more bricks wins.</p>
      <button class="btn primary" id="start">Start</button></div>`);
  }

  // ---------------------------------------------------------------- drawing
  const boards = { you: $("youCanvas"), ai: $("aiCanvas") };
  function fit(cv) {
    const r = cv.getBoundingClientRect(), d = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(r.width * d));
    if (cv.width !== w) { cv.width = w; cv.height = w; }
    return w;
  }

  function drawBoard(cv, b, prev, alpha, who) {
    const w = fit(cv), g = cv.getContext("2d"), k = w / C.SCREEN;
    g.clearRect(0, 0, w, w);
    // a faint pixel grid: the network's 48 x 48 screen
    g.fillStyle = "rgba(255,255,255,0.035)";
    for (let i = 0; i < C.SCREEN; i += 2) g.fillRect(i * k, 0, k, w);
    if (!b) return;
    // bricks
    for (let r = 0; r < C.BRICK_ROWS; r++) for (let c = 0; c < C.BRICK_COLS; c++) {
      if (!b.bricks[r][c]) continue;
      const x = c * C.BRICK_W * k, y = (C.BRICK_TOP + r * C.BRICK_H) * k;
      g.fillStyle = COLORS[r];
      roundRect(g, x + 0.5 * k * 0.2, y, (C.BRICK_W - 1) * k, (C.BRICK_H - 1) * k, k * 0.5);
      g.fillStyle = "rgba(255,255,255,0.22)";
      g.fillRect(x + k * 0.4, y + k * 0.25, (C.BRICK_W - 1.8) * k, k * 0.35);
    }
    // sparks
    for (const s of S.sparks) {
      if (s.who !== who) continue;
      g.globalAlpha = clamp(s.life * 2, 0, 1);
      g.fillStyle = s.color;
      g.fillRect(s.x * k, s.y * k, k * 0.7, k * 0.7);
    }
    g.globalAlpha = 1;
    // paddle and ball, drawn between the last two physics steps
    const lerp = (p, q) => p + (q - p) * alpha;
    const px = lerp(prev.px, b.px);
    g.fillStyle = who === "you" ? "#ff8a6e" : "#7fb8f0";
    roundRect(g, (px - C.PADDLE_W / 2) * k, C.PADDLE_Y * k, C.PADDLE_W * k, 2 * k, k);
    if (!b.over) {
      const jumped = Math.abs(b.by - prev.by) > 6;           // a new ball was served: don't draw a streak
      const bx = jumped ? b.bx : lerp(prev.bx, b.bx), by = jumped ? b.by : lerp(prev.by, b.by);
      g.fillStyle = "#ffffff";
      g.shadowColor = "rgba(255,255,255,0.8)";
      g.shadowBlur = k * 1.5;
      g.beginPath();
      g.arc(bx * k, by * k, k * 1.05, 0, Math.PI * 2);
      g.fill();
      g.shadowBlur = 0;
    } else {
      g.fillStyle = "rgba(255,255,255,0.75)";
      g.font = `700 ${Math.round(k * 4)}px ui-rounded, system-ui, sans-serif`;
      g.textAlign = "center";
      g.fillText("Out of balls", w / 2, w * 0.62);
    }
  }

  function roundRect(g, x, y, w, h, r) {
    g.beginPath();
    g.moveTo(x + r, y);
    g.arcTo(x + w, y, x + w, y + h, r);
    g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r);
    g.arcTo(x, y, x + w, y, r);
    g.closePath();
    g.fill();
  }

  // the AI's input: its two screens, exactly the pixels the network gets, and Grad-CAM on top
  const camSmall = document.createElement("canvas");
  camSmall.width = camSmall.height = 7;
  function drawSees() {
    const b = S.ai;
    if (!b) return;
    ["f0", "f1"].forEach((id, i) => {
      const g = $(id).getContext("2d"), img = g.createImageData(C.SCREEN, C.SCREEN), f = b.frames[i];
      for (let p = 0; p < f.length; p++) {
        img.data[p * 4] = img.data[p * 4 + 1] = img.data[p * 4 + 2] = f[p];
        img.data[p * 4 + 3] = 255;
      }
      g.putImageData(img, 0, 0);
    });
    const cg = $("cam").getContext("2d");
    cg.clearRect(0, 0, 48, 48);
    const info = S.info;
    if (info && info.cam) {
      const sg = camSmall.getContext("2d"), im = sg.createImageData(7, 7);
      for (let i = 0; i < 49; i++) {
        const v = info.cam[i];
        im.data[i * 4] = 255;
        im.data[i * 4 + 1] = Math.round(60 + 170 * v);
        im.data[i * 4 + 2] = 40;
        im.data[i * 4 + 3] = Math.round(200 * v);
      }
      sg.putImageData(im, 0, 0);
      cg.imageSmoothingEnabled = true;
      // each cell of the 7x7 grid looks at a 12-pixel patch of the screen, 6 pixels apart (centers 5.5 .. 41.5)
      cg.drawImage(camSmall, 2.5, 2.5, 42, 42);
    }
    const pr = info ? info.probs : null;
    $("pStay").style.width = pr ? `${pr[0] * 100}%` : "0";
    $("pLeft").style.width = pr ? `${pr[1] * 100}%` : "0";
    $("pRight").style.width = pr ? `${pr[2] * 100}%` : "0";
  }

  // ---------------------------------------------------------------- the main loop
  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (S.phase === "live" && !S.paused) {
      S.acc += dt;
      while (S.acc >= C.DT && S.phase === "live") { S.acc -= C.DT; tick(); }
      for (const s of S.sparks) { s.x += s.vx * dt; s.y += s.vy * dt; s.vy += 60 * dt; s.life -= dt; }
      S.sparks = S.sparks.filter((s) => s.life > 0);
      if (S.restartAt && now > S.restartAt) newGame();
    }
    const alpha = S.phase === "live" && !S.paused ? clamp(S.acc / C.DT, 0, 1) : 1;
    if (S.mode === "race") drawBoard(boards.you, S.you, S.prevYou, alpha, "you");
    drawBoard(boards.ai, S.ai, S.prevAi, alpha, "ai");
    drawSees();
    requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------- input
  const arena = $("arena");
  function aim(e) {
    if (S.mode !== "race") return;
    const r = boards.you.getBoundingClientRect();
    S.target = ((e.clientX - r.left) / r.width) * C.SCREEN;
  }
  arena.addEventListener("pointermove", aim);
  arena.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".overlay.on")) return;
    aim(e);
    try { arena.setPointerCapture(e.pointerId); } catch (_) {}
    unlockAudio();
  });
  const keyName = (e) => (e.key === "ArrowLeft" || e.key === "a" || e.key === "A" ? "left"
    : e.key === "ArrowRight" || e.key === "d" || e.key === "D" ? "right" : null);
  document.addEventListener("keydown", (e) => {
    const k = keyName(e);
    if (k && S.mode === "race") { S.keys[k] = true; e.preventDefault(); }
    if ((e.key === "p" || e.key === "P") && S.phase === "live") { S.paused = !S.paused; setPauseButton(); }
  });
  document.addEventListener("keyup", (e) => { const k = keyName(e); if (k) S.keys[k] = false; });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && S.phase === "live" && S.mode === "race") { S.paused = true; setPauseButton(); }
  });

  // ---------------------------------------------------------------- sound (made on the fly)
  let audio = null;
  function unlockAudio() {
    if (!S.sound) return;
    try {
      if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
      if (audio.state === "suspended") audio.resume();
    } catch (_) { audio = null; }
  }
  function sfx(kind, v = 0) {
    if (!S.sound || !audio || audio.state !== "running") return;
    const t = audio.currentTime, o = audio.createOscillator(), g = audio.createGain();
    o.connect(g); g.connect(audio.destination);
    if (kind === "brick") {
      o.type = "square"; o.frequency.setValueAtTime(880 - v * 90, t);
      g.gain.setValueAtTime(0.05, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.07);
      o.start(t); o.stop(t + 0.08);
    } else if (kind === "paddle") {
      o.type = "triangle"; o.frequency.setValueAtTime(300, t);
      g.gain.setValueAtTime(0.12, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
      o.start(t); o.stop(t + 0.07);
    } else {
      o.type = "sawtooth"; o.frequency.setValueAtTime(220, t); o.frequency.exponentialRampToValueAtTime(70, t + 0.35);
      g.gain.setValueAtTime(0.08, t); g.gain.exponentialRampToValueAtTime(0.001, t + 0.4);
      o.start(t); o.stop(t + 0.42);
    }
  }

  // ---------------------------------------------------------------- controls
  function setMode(mode) {
    S.mode = mode;
    document.querySelectorAll("#modes button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === mode)));
    arena.classList.toggle("watch", mode === "watch");
    $("hint").textContent = mode === "race"
      ? "Move your paddle with the mouse, your finger or the arrow keys. Same bricks, same serves: whoever breaks more bricks with 3 balls wins."
      : "The network plays on its own. Switch between the learning stages to see how its play changed.";
    if (mode === "watch") newGame(); else menu();
  }
  document.querySelectorAll("#modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("restart").addEventListener("click", () => { unlockAudio(); newGame(); });
  $("stage").addEventListener("change", () => { pickNet(); if (S.phase !== "menu") newGame(); });
  function setPauseButton() {
    $("pause").textContent = S.paused ? "Resume" : "Pause";
    $("pause").setAttribute("aria-pressed", String(S.paused));
  }
  $("pause").addEventListener("click", () => { if (S.phase === "live") { S.paused = !S.paused; setPauseButton(); } });
  $("sound").addEventListener("click", () => {
    S.sound = !S.sound;
    $("sound").setAttribute("aria-pressed", String(S.sound));
    $("sound").textContent = S.sound ? "🔊" : "🔈";
    if (S.sound) unlockAudio();
  });

  // ---------------------------------------------------------------- start
  loadModels().then(() => {
    menu();
    requestAnimationFrame(frame);
  });
  window.BreakoutApp = S;               // for testing
  window.Plays?.badge("breakout", document.querySelector("header h1"));
})();
