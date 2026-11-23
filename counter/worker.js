// =====================================================================================
//  Your own counting service, for the "Played N times" counters - a Cloudflare Worker with a D1 database.
//  It answers the same addresses as Abacus, so plays.js only needs its address:
//      GET /hit/<namespace>/<game>   adds 1 and returns {"value": n}
//      GET /get/<namespace>/<game>   returns {"value": n}  (404 if the game was never played)
//  Only the namespace and the games listed below are accepted, so nobody can fill your database with
//  other counters. Setup: see README.md in this folder.
// =====================================================================================
const NAMESPACE = "galnov_github_io_ai";                  // the site's namespace (see plays.js)
const GAMES = ["doodle", "emotion", "car", "runner", "maze", "hockey", "breakout"];

const HEADERS = {
  "Access-Control-Allow-Origin": "*",                     // the pages on github.io may call it
  "Cache-Control": "no-store",
  "Content-Type": "application/json",
};
const reply = (status, body) => new Response(JSON.stringify(body), { status, headers: HEADERS });

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: HEADERS });
    const m = new URL(request.url).pathname.match(/^\/(hit|get)\/([A-Za-z0-9_]{3,64})\/([A-Za-z0-9_]{3,64})$/);
    if (!m) return reply(404, { error: "Use /hit/<namespace>/<game> or /get/<namespace>/<game>" });
    const [, op, ns, game] = m;
    if (ns !== NAMESPACE || !GAMES.includes(game)) return reply(404, { error: "Unknown counter" });
    try {
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
