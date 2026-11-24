// =====================================================================================
//  The Worker: play counters + leaderboards. A Cloudflare Worker with a D1 database (binding: DB).
//
//  Play counters (the same addresses as Abacus, used by plays.js):
//      GET /hit/<namespace>/<game>   adds 1 and returns {"value": n}
//      GET /get/<namespace>/<game>   returns {"value": n}  (404 if the game was never played)
//
//  Leaderboards - for Breakout, Runner, Maze and the car race a score can't simply be sent; the server
//  checks every game itself (Doodle and Emotion are the exception - see unchecked() below):
//      POST /lb/start   {game, track?, n?, index?, vmax?}  -> {id, board, seed}
//            when a game starts. The server chooses the game's random seed (the bricks' serves, today's
//            track or maze) and remembers when the game started. Each id can be used once.
//      POST /lb/submit  {id, name, moves}                  -> {score, text, rank, top}
//            when the game ends: the player's moves (keys, paddle positions) - NOT the score. The server
//            replays the whole game from the moves with the same game code the page uses (bundled below)
//            and computes the score itself. It also checks that the game took real time (a program can't
//            play a 3-minute game in one second) and that the id wasn't used before.
//      GET  /lb/top?game=..&board=..                       -> {board, seed?, top: [{name, score, text}]}
//
//  worker.js is made from this file and the games' code by build_worker.py - edit this file, not worker.js.
//  Setup: see README.md in this folder.
// =====================================================================================
const NAMESPACE = "galnov_github_io_ai";                  // the site's namespace (see plays.js)
const GAMES = ["doodle", "emotion", "car", "runner", "maze", "hockey", "breakout"];

const HEADERS = {
  "Access-Control-Allow-Origin": "*",                     // the pages on github.io may call it
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
  "Cache-Control": "no-store",
  "Content-Type": "application/json",
};
const reply = (status, body) => new Response(JSON.stringify(body), { status, headers: HEADERS });

// ---------------------------------------------------------------- the leaderboards
const TOP_N = 10;
const SESSION_MAX_MS = 2 * 60 * 60 * 1000;                // a game must be submitted within 2 hours of its start
const REAL_TIME = 0.9;                                    // a game must take at least 90% of its game time

// today's date (UTC) - the board of the daily games
const today = () => new Date().toISOString().slice(0, 10);

// the same seed for everyone on the same day: a hash (FNV-1a) of the game's name and the date
function dailySeed(game, date) {
  let h = 2166136261;
  for (const ch of `${game}:${date}`) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; }
  return h % 1000000000;
}

const fmtTime = (s) => `${s.toFixed(s < 100 ? 2 : 1)} s`;

// For every game: how it is ranked, how its score is written, how a game is started and replayed.
const LB = {
  breakout: {
    higherIsBetter: true,
    board: "all",                                         // one leaderboard for all games
    text: (s) => `${s} ${s === 1 ? "brick" : "bricks"}`,
    start: () => ({ board: "all", seed: (Math.random() * 2 ** 31) | 0, opts: {} }),
    // moves: one letter per decision, "a" (full left) .. "k" (stay) .. "u" (full right)
    replay(s, moves) {
      const { Board, seeded, C } = globalThis.Breakout;
      if (typeof moves !== "string" || moves.length > C.MAX_STEPS || !/^[a-u]*$/.test(moves)) return { error: "bad moves" };
      const b = new Board(seeded(s.seed), true);
      for (const ch of moves) {
        if (b.over) break;
        b.step((ch.charCodeAt(0) - 107) / 10);
      }
      if (!b.over) return { error: "the game didn't end" };
      return { score: b.score, seconds: b.steps * C.DT };
    },
  },

  runner: {
    higherIsBetter: true,
    daily: true,
    text: (s) => `${s.toFixed(1)} m`,
    seedOf: (board) => dailySeed("runner", board) % 2147483647,
    start() { const d = today(); return { board: d, seed: this.seedOf(d), opts: {} }; },
    // moves: one digit per tick (0 nothing, 1 left, 2 right, 3 jump, 4 slide)
    replay(s, moves) {
      const { Game, C } = globalThis.Runner;
      const MAX_TICKS = 12000;                            // 10 minutes
      if (typeof moves !== "string" || moves.length > MAX_TICKS || !/^[0-4]*$/.test(moves)) return { error: "bad moves" };
      const g = new Game(s.seed);
      for (const ch of moves) {
        if (!g.alive) break;
        g.step(ch.charCodeAt(0) - 48);
      }
      if (g.alive && moves.length < MAX_TICKS) return { error: "the run didn't end" };
      return { score: Math.round(g.distance() * 10) / 10, seconds: (g.t * C.TICK_MS) / 1000 };
    },
  },

  maze: {
    higherIsBetter: false,
    daily: true,
    text: fmtTime,
    PHYSICS: {
      cell: 4.0, radius: 0.6, v_min: 2.0, v_max: 11.0, accel_k: 2.5, pitch_rate: 2.2, yaw_rate: 2.2, dt: 0.1, substeps: 2,
      rays_x: 9, rays_y: 9, fov: 100.0, ray_range: 24.0,
    },
    ROOMS: 4,
    seedOf: (board) => dailySeed("maze", board),
    start(body) {
      const d = today(), vmax = +body.vmax;
      if (!(vmax >= 3 && vmax <= this.PHYSICS.v_max)) return { error: "bad top speed" };
      return { board: d, seed: this.seedOf(d), opts: { vmax } };
    },
    // moves: [pitch, yaw, throttle, roll] for every 0.1 s step, each a whole number from -50 to 50 (= -1 .. 1)
    replay(s, moves) {
      const { MazeData, Ship, makeMaze } = globalThis.MazeSim;
      const P = { ...this.PHYSICS, v_max: s.opts.vmax };
      const MAX = 2400;                                   // flights of up to 4 minutes are ranked
      if (!Array.isArray(moves) || moves.length % 4) return { error: "bad moves" };
      if (moves.length > MAX * 4) return { error: "flights longer than 4 minutes are not ranked" };
      if (!moves.every((v) => Number.isInteger(v) && v >= -50 && v <= 50)) return { error: "bad moves" };
      const ship = new Ship(new MazeData(makeMaze(s.seed, this.ROOMS, P.cell)), P);
      for (let i = 0; i < moves.length; i += 4) {
        ship.step([moves[i] / 50, moves[i + 1] / 50, moves[i + 2] / 50, moves[i + 3] / 50]);
        if (ship.crashed) return { error: "the ship crashed" };
        if (ship.reached) return { score: Math.round(ship.t * 10) / 10, seconds: ship.t };
      }
      return { error: "the ship didn't reach the exit" };
    },
  },

  car: {
    higherIsBetter: false,
    text: fmtTime,
    LAPS: 3,
    start(body) {
      const n = body.n | 0, index = body.index | 0;
      if (!TRACKS[body.track]) return { error: "unknown track" };
      if (n < 1 || n > 9 || index < 0 || index >= n) return { error: "bad start" };
      return { board: body.track, seed: 0, opts: { n, index } };
    },
    // moves: one digit per 0.05 s step: (steer + 1) * 3 + (throttle + 1)
    replay(s, moves) {
      const { Track, CarEnv, startOffset, C } = globalThis.CarSim;
      const MAX = 12000;                                  // 10 minutes
      if (typeof moves !== "string" || moves.length > MAX || !/^[0-8]*$/.test(moves)) return { error: "bad moves" };
      const T = TRACKS[s.board], track = new Track(T.center, T.width, T.name);
      const env = new CarEnv(track, 1), car = env.cars[0];
      const [ox, oy] = startOffset(track, s.opts.n, s.opts.index);
      car.x += ox; car.y += oy;
      for (const ch of moves) {
        const k = ch.charCodeAt(0) - 48;
        const { crashed } = env.step([[Math.floor(k / 3) - 1, (k % 3) - 1]]);
        if (crashed[0]) env.respawn(0);
        if (car.laps >= this.LAPS) {
          const t = Math.round(car.steps * C.DT * 100) / 100;
          return { score: t, seconds: t };
        }
      }
      return { error: "the race wasn't finished" };
    },
  },
};

// Doodle and Emotion: their scores CAN'T be checked (they come from the camera, or from the network judging
// drawings in the browser), so the page sends its score. The server only checks that the number is possible,
// that the game took at least its shortest possible time, and that each game id is used once.
// Someone who knows how could still post a made-up score here.
function unchecked(max, minSeconds, text) {
  return {
    higherIsBetter: true,
    unchecked: true,
    board: "all",
    text,
    start: () => ({ board: "all", seed: 0, opts: {} }),
    replay(s, moves) {
      const v = moves && moves.score;
      if (!Number.isInteger(v) || v < 0 || v > max) return { error: "impossible score" };
      return { score: v, seconds: minSeconds };
    },
  };
}
// Doodle: 6 drawings; a recognized drawing gives 100 points + 5 for every second left (of 20)
LB.doodle = unchecked(1200, 6, (s) => `${s} points`);
// Emotion: 5 faces, up to 100 points each; the rounds take 5 s each
LB.emotion = unchecked(500, 25, (s) => `${s} points`);

// a player's name: 1-16 letters, digits, spaces and a few signs
function cleanName(name) {
  const n = String(name || "").normalize("NFC").replace(/\s+/g, " ").trim();
  if (!n || n.length > 16 || !/^[\p{L}\p{N} ._'-]+$/u.test(n)) return null;
  return n;
}

async function top(env, game, board) {
  const L = LB[game];
  const rows = await env.DB.prepare(
    `SELECT name, ${L.higherIsBetter ? "MAX" : "MIN"}(score) AS s, MIN(created) AS t FROM scores
     WHERE game = ?1 AND board = ?2 GROUP BY name ORDER BY s ${L.higherIsBetter ? "DESC" : "ASC"}, t ASC LIMIT ?3`
  ).bind(game, board, TOP_N).all();
  return rows.results.map((r) => ({ name: r.name, score: r.s, text: L.text(r.s) }));
}

async function leaderboard(request, env, url) {
  const path = url.pathname;
  if (path === "/lb/top" && request.method === "GET") {
    const game = url.searchParams.get("game"), L = LB[game];
    if (!L) return reply(404, { error: "Unknown game" });
    const board = url.searchParams.get("board") || (L.daily ? today() : L.board || null);
    if (!board || board.length > 40) return reply(400, { error: "Which board?" });
    // a daily board also tells its seed (the same for everyone), so the page can show today's track or maze
    return reply(200, { board, ...(L.seedOf ? { seed: L.seedOf(board) } : {}), top: await top(env, game, board) });
  }
  if (request.method !== "POST") return reply(405, { error: "Use POST" });
  let body;
  try { body = await request.json(); } catch (_) { return reply(400, { error: "Bad request" }); }

  if (path === "/lb/start") {
    const game = body.game, L = LB[game];
    if (!L) return reply(404, { error: "Unknown game" });
    const st = L.start(body);
    if (st.error) return reply(400, { error: st.error });
    const id = crypto.randomUUID(), now = Date.now();
    await env.DB.batch([
      env.DB.prepare("DELETE FROM sessions WHERE created < ?1").bind(now - SESSION_MAX_MS),   // forget old games
      env.DB.prepare("INSERT INTO sessions (id, game, board, seed, opts, created) VALUES (?1, ?2, ?3, ?4, ?5, ?6)")
        .bind(id, game, st.board, st.seed, JSON.stringify(st.opts), now),
    ]);
    return reply(200, { id, board: st.board, seed: st.seed });
  }

  if (path === "/lb/submit") {
    const name = cleanName(body.name);                    // (checked first, so a bad name can be fixed and sent again)
    if (!name) return reply(400, { error: "A name of 1-16 letters or digits, please" });
    // take the session out of the database: an id can be used only once
    const s = await env.DB.prepare("DELETE FROM sessions WHERE id = ?1 RETURNING *").bind(String(body.id || "")).first();
    if (!s) return reply(400, { error: "This game was already submitted, or it is too old" });
    const L = LB[s.game];
    s.opts = JSON.parse(s.opts || "{}");
    let r;
    try { r = L.replay(s, body.moves); } catch (e) { r = { error: "the game could not be replayed" }; }
    if (r.error) return reply(400, { error: `Not accepted: ${r.error}` });
    const elapsed = (Date.now() - s.created) / 1000;
    if (elapsed > SESSION_MAX_MS / 1000) return reply(400, { error: "Not accepted: the game is too old" });
    if (elapsed < r.seconds * REAL_TIME) return reply(400, { error: "Not accepted: the game was played faster than real time" });
    await env.DB.prepare("INSERT INTO scores (game, board, name, score, created) VALUES (?1, ?2, ?3, ?4, ?5)")
      .bind(s.game, s.board, name, r.score, Date.now()).run();
    // the rank of this score among the players' best scores
    const better = await env.DB.prepare(
      `SELECT COUNT(*) AS c FROM (SELECT name, ${L.higherIsBetter ? "MAX" : "MIN"}(score) AS b FROM scores
       WHERE game = ?1 AND board = ?2 AND name <> ?3 GROUP BY name) WHERE b ${L.higherIsBetter ? ">" : "<"} ?4`
    ).bind(s.game, s.board, name, r.score).first();
    return reply(200, { score: r.score, text: L.text(r.score), rank: better.c + 1, board: s.board, top: await top(env, s.game, s.board) });
  }
  return reply(404, { error: "Unknown address" });
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: HEADERS });
    const url = new URL(request.url);
    try {
      if (url.pathname.startsWith("/lb/")) return await leaderboard(request, env, url);
      const m = url.pathname.match(/^\/(hit|get)\/([A-Za-z0-9_]{3,64})\/([A-Za-z0-9_]{3,64})$/);
      if (!m) return reply(404, { error: "Use /hit/<namespace>/<game> or /get/<namespace>/<game>" });
      const [, op, ns, game] = m;
      if (ns !== NAMESPACE || !GAMES.includes(game)) return reply(404, { error: "Unknown counter" });
      if (op === "hit") {
        // one statement, so two visitors at the same moment can't lose a play
        const row = await env.DB.prepare(
          "INSERT INTO plays (game, n) VALUES (?1, 1) ON CONFLICT (game) DO UPDATE SET n = n + 1 RETURNING n"
        ).bind(game).first();
        return reply(200, { value: row.n });
      }
      const row = await env.DB.prepare("SELECT n FROM plays WHERE game = ?1").bind(game).first();
      return row ? reply(200, { value: row.n }) : reply(404, { error: "Key not found" });
    } catch (e) {
      return reply(500, { error: String(e) });
    }
  },
};
