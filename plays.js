// =====================================================================================
//  "Played N times": one counter per game, shared by everyone who opens the page.
//
//  GitHub Pages only serves files, so it can't keep a number that grows. The counters are kept by a small
//  counting service on the internet, with two addresses:
//      GET <API>/hit/<namespace>/<game>   adds 1 and returns {"value": n}
//      GET <API>/get/<namespace>/<game>   returns {"value": n} without changing it (404 = never played)
//  By default that is Abacus (https://abacus.jasoncameron.dev), a free service that needs no account.
//  Your own counter (counter/ in the repository, on Cloudflare) answers the same addresses: put its
//  address in API below. This one file serves all the games (they load it as ../plays.js).
//
//  The internet is not always reliable, so this file does not give up:
//    * every request has a time limit, and is tried again a few times (waiting longer each time)
//    * a play that could not be sent is kept in this browser (localStorage) and sent later - on the next
//      page this visitor opens - so it is not lost
//    * the last number seen is kept too, so the line appears right away, and is updated when the answer comes
//  The namespace is made from the site's address (galnov.github.io/ai -> galnov_github_io_ai), so every copy of
//  the site has its own counters. On a local test (localhost, file://) nothing is counted.
// =====================================================================================
(function (root) {
  const API = root.PLAYS_API || "https://plays-counter.galnvk.workers.dev";
  const host = location.hostname;
  const LOCAL = !root.PLAYS_FORCE && (!host || host === "localhost" || host === "127.0.0.1" || host.endsWith(".local") || location.protocol === "file:");
  const TIMEOUT_MS = 8000, TRIES = 4;
  const PENDING = "plays.pending", LAST = "plays.last";

  function namespace() {
    if (root.PLAYS_NAMESPACE) return root.PLAYS_NAMESPACE;
    const repo = host.endsWith(".github.io") ? location.pathname.split("/").filter(Boolean)[0] || "" : "";
    return `${host}_${repo}`.replace(/[^A-Za-z0-9_]/g, "_").replace(/_+$/, "").slice(0, 64);
  }

  // remembered in localStorage; if it is missing or full (private browsing), at least for this page
  const mem = {};
  const load = (k) => {
    try { const s = localStorage.getItem(k); if (s !== null) return JSON.parse(s); } catch {}
    return JSON.parse(mem[k] || "{}");
  };
  const save = (k, v) => { mem[k] = JSON.stringify(v); try { localStorage.setItem(k, mem[k]); } catch {} };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // one request, tried again on a network error, a time-out, "too many requests" (429) or a server error (5xx)
  async function call(op, game) {
    if (LOCAL) return null;
    for (let i = 0; i < TRIES; i++) {
      const ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
      const timer = ctl && setTimeout(() => ctl.abort(), TIMEOUT_MS);
      let wait = 1000 * 2 ** i;
      try {
        const r = await fetch(`${API}/${op}/${namespace()}/${game}`, { cache: "no-store", keepalive: op === "hit", signal: ctl?.signal });
        if (op === "get" && r.status === 404) return 0;
        if (r.ok) {
          const v = (await r.json()).value;
          if (Number.isFinite(v)) return v;
        } else if (r.status !== 429 && r.status < 500) {
          return null;                                    // a mistake in the request: trying again won't help
        } else {
          const ra = +r.headers.get("Retry-After");       // the service may say how long to wait
          if (ra > 0) wait = Math.max(wait, ra > 100 ? ra : ra * 1000);
        }
      } catch (e) {
        // no connection, blocked, or timed out: try again
      } finally {
        if (timer) clearTimeout(timer);
      }
      if (i < TRIES - 1) await sleep(wait + Math.random() * 500);
    }
    return null;
  }

  // ---------------------------------------------------------------- the badges
  let styled = false;
  function style() {
    if (styled) return;
    styled = true;
    const s = document.createElement("style");
    s.textContent = `.plays { display: inline-flex; align-items: center; gap: 6px; margin: 6px 0 0; padding: 3px 10px;
      border-radius: 999px; border: 1px solid currentColor; font-size: 13px; line-height: 1.5; opacity: 0.75; white-space: nowrap; }
      .plays[hidden] { display: none; }
      .plays b { font-variant-numeric: tabular-nums; }`;
    document.head.appendChild(s);
  }

  const shown = new Map();                     // game -> the elements that show its number
  const best = load(LAST);                     // game -> the largest number seen so far (the counter only grows)
  const sending = {};                          // game -> plays on their way to the service right now

  function show(game, v) {
    if (v === null || v === undefined) return;
    best[game] = Math.max(best[game] || 0, v);
    save(LAST, best);
    const n = best[game] + (load(PENDING)[game] || 0) + (sending[game] || 0);   // plays not sent yet are counted too
    for (const set of shown.get(game) || []) set(n);
  }

  function watch(game, set) {
    if (!shown.has(game)) shown.set(game, []);
    shown.get(game).push(set);
    if (game in best || load(PENDING)[game]) show(game, best[game] || 0);   // the last number seen, right away
    call("get", game).then((v) => show(game, v));                          // then the current one
  }

  // ---------------------------------------------------------------- sending the plays
  let flushing = null;
  function flush() {
    if (flushing || LOCAL) return flushing;
    flushing = (async () => {
      for (const game of Object.keys(load(PENDING))) {
        while ((load(PENDING)[game] || 0) > 0) {
          // taken off the list before sending (so another open tab doesn't send it too), put back if it fails
          const change = (d) => { const p = load(PENDING); p[game] = Math.max(0, (p[game] || 0) + d); if (!p[game]) delete p[game]; save(PENDING, p); };
          change(-1);
          sending[game] = (sending[game] || 0) + 1;
          const v = await call("hit", game);
          sending[game] -= 1;
          if (v === null) { change(+1); return; }           // still no connection: try again later
          show(game, v);
        }
      }
    })().finally(() => { flushing = null; });
    return flushing;
  }

  root.Plays = {
    // a badge "🎮 Played N times by visitors", placed after the element `after`
    badge(game, after) {
      style();
      const el = document.createElement("div");
      el.className = "plays";
      el.hidden = true;
      el.title = "How many games were played on this page, by all visitors together";
      el.innerHTML = `🎮 Played <b>0</b> <i style="font-style:normal">times</i> by visitors`;
      after.insertAdjacentElement("afterend", el);
      watch(game, (n) => {
        el.hidden = false;
        el.querySelector("b").textContent = n.toLocaleString("en-US");
        el.querySelector("i").textContent = n === 1 ? "time" : "times";
      });
      return el;
    },
    // any element: set(n) is called with the number whenever it is known (used by the home page)
    watch,
    // a game has started: remember it, and send it (now if possible, or later)
    played(game) {
      if (LOCAL) return;
      const p = load(PENDING);
      p[game] = (p[game] || 0) + 1;
      save(PENDING, p);
      show(game, best[game] || 0);
      flush();
    },
    get: (game) => call("get", game),
    namespace, LOCAL,
  };
  // ================================================================= the leaderboards
  // The page never sends a score. When a game starts it asks the server for a game id (and the game's seed);
  // when it ends it sends the player's moves, and the server replays the game to find the score
  // (counter/worker_src.js). Shown with Leaderboard.panel(); a finished game is posted with Leaderboard.offer().
  async function lbFetch(path, body, tries = 2) {
    if (LOCAL) return { error: "The leaderboard works on the published site" };
    for (let i = 0; i < tries; i++) {
      const ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
      const timer = ctl && setTimeout(() => ctl.abort(), 10000);
      try {
        const r = await fetch(`${API}${path}`, body ? { method: "POST", body: JSON.stringify(body), signal: ctl?.signal,
          headers: { "Content-Type": "application/json" } } : { cache: "no-store", signal: ctl?.signal });
        const j = await r.json().catch(() => ({}));
        if (r.ok) return j;
        if (r.status < 500) return { error: j.error || `Error ${r.status}` };
      } catch (_) { /* no connection: try again */ } finally { if (timer) clearTimeout(timer); }
      if (i < tries - 1) await sleep(800);
    }
    return { error: "No connection to the leaderboard" };
  }

  let lbStyled = false;
  function lbStyle() {
    if (lbStyled) return;
    lbStyled = true;
    const s = document.createElement("style");
    s.textContent = `.lb h3 { margin: 0 0 6px; font-size: 1em; }
      .lb .lb-sub { margin: -2px 0 8px; font-size: 0.82em; opacity: 0.7; }
      .lb ol { margin: 0; padding: 0; list-style: none; font-variant-numeric: tabular-nums; }
      .lb li { display: grid; grid-template-columns: 1.8em 1fr auto; gap: 6px; padding: 3px 0; font-size: 0.92em;
        border-top: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
      .lb li:first-child { border-top: 0; }
      .lb li span:first-child { opacity: 0.6; }
      .lb li span:nth-child(2) { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .lb li.me { font-weight: 700; }
      .lb .lb-empty { font-size: 0.88em; opacity: 0.7; margin: 0; }
      .lb-offer { margin: 10px 0 0; display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; align-items: center; }
      .lb-offer input { font: inherit; font-size: 0.95em; padding: 7px 9px; border-radius: 8px; width: 11em; max-width: 100%;
        border: 1px solid color-mix(in srgb, currentColor 35%, transparent); background: transparent; color: inherit; }
      .lb-offer button { font: inherit; font-size: 0.95em; font-weight: 700; padding: 7px 12px; border-radius: 8px; cursor: pointer;
        border: 1px solid currentColor; background: transparent; color: inherit; }
      .lb-offer button:disabled { opacity: 0.5; cursor: default; }
      .lb-offer .lb-msg { flex-basis: 100%; font-size: 0.88em; margin: 2px 0 0; }`;
    document.head.appendChild(s);
  }
  const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const lbName = () => { try { return localStorage.getItem("lb.name") || ""; } catch { return ""; } };

  root.Leaderboard = {
    enabled: !LOCAL,
    // a game starts: {id, board, seed} (or {error}). extra: the game's settings (track, top speed...)
    start: (game, extra = {}) => lbFetch("/lb/start", { game, ...extra }),
    // the game ended: send the moves; the server answers with the score it found, the rank and the top 10
    submit: (id, name, moves) => lbFetch("/lb/submit", { id, name, moves }, 1),
    top: (game, board) => lbFetch(`/lb/top?game=${encodeURIComponent(game)}${board ? `&board=${encodeURIComponent(board)}` : ""}`),

    // a leaderboard in the element el: {game, title, sub, board?, autoload?}. Returns {show(board, sub), set(result)}
    panel(el, opt) {
      lbStyle();
      el.classList.add("lb");
      let board = opt.board || null, sub = opt.sub || "";
      const render = (rows, error) => {
        const me = lbName();
        el.innerHTML = `<h3>${esc(opt.title || "Leaderboard")}</h3>${sub ? `<p class="lb-sub">${esc(sub)}</p>` : ""}` +
          (error ? `<p class="lb-empty">${esc(error)}</p>`
            : !rows.length ? `<p class="lb-empty">No scores yet. Be the first!</p>`
            : `<ol>${rows.map((r, i) => `<li class="${r.name === me ? "me" : ""}"><span>${i + 1}</span><span>${esc(r.name)}</span><span>${esc(r.text)}</span></li>`).join("")}</ol>`);
      };
      const api = {
        async show(b, s) {
          if (b !== undefined) board = b;
          if (s !== undefined) sub = s;
          if (LOCAL) return render([], "The leaderboard works on the published site.");
          const r = await lbFetch(`/lb/top?game=${encodeURIComponent(opt.game)}${board ? `&board=${encodeURIComponent(board)}` : ""}`);
          render(r.top || [], r.error);
        },
        set(r) { if (r && r.top) render(r.top); },
      };
      if (opt.autoload === false) render([], "Loading…"); else api.show();
      return api;
    },

    // a "post my score" form in the element el (e.g. the result card). getMoves() gives the moves to send.
    // onDone(result) is called with the server's answer.
    offer(el, { id, getMoves, onDone }) {
      lbStyle();
      if (!id || LOCAL) return null;
      const box = document.createElement("div");
      box.className = "lb-offer";
      box.innerHTML = `<input maxlength="16" placeholder="Your name" aria-label="Your name for the leaderboard" value="${esc(lbName())}">
        <button type="button">Post my score</button><p class="lb-msg" aria-live="polite"></p>`;
      el.appendChild(box);
      const [input, btn, msg] = [box.querySelector("input"), box.querySelector("button"), box.querySelector(".lb-msg")];
      input.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Enter") btn.click(); });
      btn.addEventListener("click", async () => {
        const name = input.value.trim();
        if (!name) { msg.textContent = "Type your name first."; input.focus(); return; }
        if (!/^[\p{L}\p{N} ._'-]{1,16}$/u.test(name)) { msg.textContent = "Letters, digits and spaces only, please."; input.focus(); return; }
        try { localStorage.setItem("lb.name", name); } catch {}
        btn.disabled = input.disabled = true;
        msg.textContent = "Checking your game…";
        const r = await root.Leaderboard.submit(id, name, await getMoves());
        if (r.error) { msg.textContent = r.error; return; }
        msg.textContent = `Posted: ${r.text}, #${r.rank} on the leaderboard.`;
        onDone?.(r);
      });
      return box;
    },
  };

  setTimeout(flush, 1500);                       // plays from an earlier visit that were not sent yet
  setInterval(() => { if (Object.keys(load(PENDING)).length) flush(); }, 20000);   // and again every 20 seconds
})(typeof window !== "undefined" ? window : globalThis);
