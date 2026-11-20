// =====================================================================================
//  The "server" of the static version: everything app/server.py did, done in the browser.
//    info       - the AI players (the exported networks + the rule-based bot) and the leaderboard
//    ghost      - the AI plays the same track (same seed) in its own copy of the game (game.js),
//                 and its key presses are handed to the page in chunks, like the server did
//    recording  - your runs are kept in this browser (they are the training data for imitation learning).
//                 "Download my runs" saves them to a file, and  python -m src.import_recordings  adds them
//                 to data/recordings/ for the next training run
//  The networks come from docs/model (python -m src.export_web) and run in mlp.js.
// =====================================================================================
(function (root) {
  const { C, Game, jumpHeight } = root.Runner;
  const CHUNK = 200;                 // ticks of AI play per request (10 seconds)
  const GHOST_TICKS = 12000;         // the AI runs for up to 10 minutes
  const STORE = "runner.recordings";
  const LABELS = { combo: "Combo (imitation + RL)", dqn: "DQN (RL from scratch)",
                   imitation: "Imitation (learned from humans)", bot: "Rule-based bot" };
  const CRASH_NAMES = { 0: null, 1: "hurdle", 2: "overhead bar", 3: "train" };

  // ---------------------------------------------------------------- what the network sees (src/game.py observe)
  function observe(g) {
    const o = new Float32Array(3 * C.N_BINS * C.N_LANES + 9);
    for (const ob of g.obs) {
      if (ob.typ === 0) continue;
      const r0 = ob.z0 - g.pz, r1 = ob.z1 - g.pz;
      for (let b = 0; b < C.N_BINS; b++) {
        const lo = b * C.BIN_CM;
        if (r0 < lo + C.BIN_CM && r1 >= lo) o[((ob.typ - 1) * C.N_BINS + b) * C.N_LANES + ob.lane] = 1;
      }
    }
    const f = 3 * C.N_BINS * C.N_LANES;
    o[f + (g.lcT > 0 ? g.lcTo : g.lane)] = 1;                 // 0-2: the lane it is in / heading to
    o[f + 3] = g.lcT / C.LANE_TICKS;                          // 3: lane change still to go
    o[f + 4] = g.pending;                                      // 4: a queued lane change (-1/0/1)
    o[f + 5] = jumpHeight(g.jt) / C.JUMP_HEIGHT;               // 5: height
    o[f + 6] = g.jt / C.JUMP_TICKS;                            // 6: how far into the jump
    o[f + 7] = g.st / C.SLIDE_TICKS;                           // 7: slide time left
    o[f + 8] = (g.speed() - C.V0) / (C.VMAX - C.V0);           // 8: speed (0 = start, 1 = top)
    return o;
  }

  // ---------------------------------------------------------------- the rule-based bot (src/bot.py, skill = 1)
  function firstOverlap(pz, v, z0, z1) {
    const first = Math.max(1, Math.ceil((z0 - C.HALF_LEN - pz) / v));
    const last = Math.floor((z1 + C.HALF_LEN - pz) / v) + 1;
    return [first, last];
  }

  function bot(g) {
    const v = g.speed(), pz = g.pz;
    const tgt = g.lcT > 0 ? g.lcTo : g.lane, eff = g.effectiveLane();
    const obs = g.obs.filter((o) => o.typ > 0 && o.z1 + C.HALF_LEN >= pz);
    const free = [1e9, 1e9, 1e9];                              // how far is the next train in each lane
    for (const o of obs) if (o.typ === C.TRAIN) free[o.lane] = Math.min(free[o.lane], Math.max(0, o.z0 - C.HALF_LEN - pz));

    if (g.lcT === 0 && free[tgt] < v * 16) {                   // trains: change lanes in good time
      let best = null, bestScore = free[tgt];
      for (let l = 0; l < C.N_LANES; l++) {
        if (l === tgt) continue;
        let pathOk = true;
        for (let k = Math.min(l, tgt); k <= Math.max(l, tgt); k++) {
          if (k !== tgt && !(free[k] > v * 3 * Math.abs(k - tgt) + v * 2)) pathOk = false;
        }
        const score = free[l] - 50 * Math.abs(l - tgt);
        if (pathOk && score > bestScore) { best = l; bestScore = score; }
      }
      if (best !== null) return best < tgt ? C.LEFT : C.RIGHT;
    }
    // hurdles and bars: jump / slide at the right moment
    for (const [typ, act, w0, w1] of [[C.LOW, C.JUMP, 2, 11], [C.HIGH, C.SLIDE, 1, 12]]) {
      for (const o of obs) {
        if (o.typ !== typ || (o.lane !== tgt && o.lane !== eff)) continue;
        const [first, last] = firstOverlap(pz, v, o.z0, o.z1);
        if (typ === C.LOW && g.jt > 0 && first + g.jt - 1 >= 2 && last + g.jt - 1 <= 11) continue;
        if (typ === C.HIGH && g.st >= last) continue;
        if (first <= w0 + 1 && last <= w1) return act;
      }
    }
    return C.NOOP;
  }

  // ---------------------------------------------------------------- the players
  let players = null, message = null;

  async function loadPlayers() {
    players = new Map();
    try {
      const model = await root.MLP.load("model");
      if (model) {
        for (const id of ["combo", "dqn", "imitation"]) {
          const spec = model.meta.networks.find((n) => n.id === id);
          if (!spec) continue;
          const net = model.nets[id];
          players.set(id, {
            kind: spec.kind,
            act(g) {
              const out = net.forward(observe(g)).out;
              let scores = out;
              if (spec.kind === "imitation") {                 // probabilities: how likely a human would press each key
                const m = Math.max(...out), e = out.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0);
                scores = e.map((x) => x / s);
              }
              let a = 0;
              for (let i = 1; i < scores.length; i++) if (scores[i] > scores[a]) a = i;
              return [a, scores.map((x) => Math.round(x * 1000) / 1000)];
            },
          });
        }
      }
      if (!players.size) {
        message = "No trained network has been exported to this page yet, so the AI ghost is the rule-based bot. " +
                  "Play a few runs, download them, train (python -m src.train_bc), then run  python -m src.export_web  and publish again.";
      }
    } catch (e) {
      message = `Could not load a network: ${e.message}`;
    }
    players.set("bot", { kind: "bot", act: (g) => [bot(g), null] });
  }

  // ---------------------------------------------------------------- the leaderboard (this browser's runs)
  const localTime = () => { const d = new Date(), p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`; };
  const stored = () => { try { return JSON.parse(localStorage.getItem(STORE) || "[]"); } catch { return []; } };

  function leaderboard() {
    const recs = stored();
    const ticks = recs.reduce((s, r) => s + r.ticks, 0);
    const h2h = recs.filter((r) => r.ai);
    return {
      top: [...recs].sort((a, b) => b.distance - a.distance).slice(0, 10)
        .map((r) => ({ player: r.player || "anonymous", distance: r.distance, ai: r.ai })),
      summary: { runs: recs.length, minutes: (ticks * C.TICK_MS) / 60000 },
      head_to_head: { races: h2h.length, human_wins: h2h.filter((r) => r.distance > r.ai.distance).length },
    };
  }

  // ---------------------------------------------------------------- ghost sessions
  const sessions = new Map();
  let nextSid = 1;

  function runGhost(s, n, keep = true) {
    const actions = [], scores = [];
    for (let i = 0; i < n; i++) {
      const g = s.game;
      if (!g.alive || g.t >= GHOST_TICKS) break;
      const [a, sc] = players.get(s.model).act(g);
      g.step(a);
      if (keep) { actions.push(a); scores.push(sc); }
    }
    return { actions, scores };
  }

  function ghostState(s) {
    const g = s.game;
    return { ticks: g.t, over: !g.alive || g.t >= GHOST_TICKS, distance: g.distance(), crash: CRASH_NAMES[g.crashType] };
  }

  root.Backend = {
    async info() {
      if (!players) await loadPlayers();
      return {
        players: [...players].map(([id, p]) => ({ id, label: LABELS[id], kind: p.kind })),
        message, leaderboard: leaderboard(),
      };
    },

    async ghostStart({ model, seed }) {
      if (!players) await loadPlayers();
      const id = players.has(model) ? model : players.keys().next().value;
      const sid = String(nextSid++);
      const s = { game: new Game(seed & 0x7fffffff), model: id };
      sessions.set(sid, s);
      if (sessions.size > 30) sessions.delete(sessions.keys().next().value);
      return { sid, model: id, ...runGhost(s, CHUNK), ...ghostState(s) };
    },

    async ghostMore({ sid }) {
      const s = sessions.get(sid);
      if (!s) return { error: "unknown session" };
      return { ...runGhost(s, CHUNK), ...ghostState(s) };
    },

    // the human has crashed: let the AI play to the end, a slice at a time so the page stays responsive
    async ghostFinish({ sid }) {
      const s = sessions.get(sid);
      if (!s) return { error: "unknown session" };
      while (!ghostState(s).over) {
        runGhost(s, 400, false);
        await new Promise((r) => setTimeout(r, 0));
      }
      return ghostState(s);
    },

    async recording(d) {
      // replay it: the run is fully described by the seed and the key presses
      const g = new Game(d.seed);
      const acts = new Array(d.ticks).fill(0);
      for (const [t, a] of d.presses) if (t >= 0 && t < d.ticks) acts[t] = a;
      let played = 0;
      for (const a of acts) { g.step(a); played++; if (!g.alive) break; }
      if (Math.abs(g.distance() - d.distance) > 0.011 || played !== d.ticks) {
        return { saved: false, error: `the replay did not match (${g.distance().toFixed(1)} m)` };
      }
      const rec = { player: String(d.player || "anonymous").slice(0, 24), source: "human", seed: d.seed, ticks: d.ticks,
                    distance: g.distance(), crash: CRASH_NAMES[g.crashType], presses: d.presses,
                    date: localTime() };
      if (d.ai && LABELS[d.ai.model]) rec.ai = { model: d.ai.model, distance: d.ai.distance };
      const all = stored();
      all.push(rec);
      try { localStorage.setItem(STORE, JSON.stringify(all)); } catch (e) { return { saved: false, error: "this browser's storage is full - download your runs" }; }
      return { saved: true, leaderboard: leaderboard() };
    },

    savedCount: () => stored().length,

    downloadRecordings() {
      const blob = new Blob([JSON.stringify({ format: "runner-recordings-v1", recordings: stored() })], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "my_runs.json";
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    },

    clearRecordings() { localStorage.removeItem(STORE); return leaderboard(); },
  };
  root.RunnerBackend = { observe, bot, LABELS };
})(typeof window !== "undefined" ? window : globalThis);
