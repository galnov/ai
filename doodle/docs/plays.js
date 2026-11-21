// =====================================================================================
//  "Played N times": one counter per game, shared by everyone who opens the page.
//
//  GitHub Pages only serves files, so it can't keep a number that grows. The counter is kept by Abacus
//  (https://abacus.jasoncameron.dev), a free counting service that needs no account:
//      GET /hit/<namespace>/<key>   adds 1 and returns {"value": n}
//      GET /get/<namespace>/<key>   returns {"value": n} without changing it (404 = never played yet)
//  The namespace is made from the site's address (e.g. galnov.github.io/ai -> galnov_github_io_ai), so every
//  copy of the site has its own counters. On a local test (localhost, file://) nothing is counted.
//  If the service can't be reached, the page works as usual and the counter just isn't shown.
// =====================================================================================
(function (root) {
  const API = "https://abacus.jasoncameron.dev";
  const host = location.hostname;
  const LOCAL = !host || host === "localhost" || host === "127.0.0.1" || host.endsWith(".local") || location.protocol === "file:";

  function namespace() {
    if (root.PLAYS_NAMESPACE) return root.PLAYS_NAMESPACE;
    const repo = location.hostname.endsWith(".github.io") ? location.pathname.split("/").filter(Boolean)[0] || "" : "";
    return `${host}_${repo}`.replace(/[^A-Za-z0-9_]/g, "_").replace(/_+$/, "").slice(0, 64);
  }

  async function call(op, game) {
    if (LOCAL) return null;
    try {
      const r = await fetch(`${API}/${op}/${namespace()}/${game}`, { cache: "no-store" });
      if (op === "get" && r.status === 404) return 0;
      if (!r.ok) return null;
      const v = (await r.json()).value;
      return Number.isFinite(v) ? v : null;
    } catch (e) {
      return null;
    }
  }

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

  const shown = new Map();   // game -> the badge elements on this page
  function show(game, v) {
    for (const el of shown.get(game) || []) {
      if (v === null) continue;
      el.hidden = false;
      el.querySelector("b").textContent = v.toLocaleString("en-US");
      el.querySelector("i").textContent = v === 1 ? "time" : "times";
    }
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
      if (!shown.has(game)) shown.set(game, []);
      shown.get(game).push(el);
      call("get", game).then((v) => show(game, v));
      return el;
    },
    // a game has started: add 1 (and update the badges)
    async played(game) {
      const v = await call("hit", game);
      show(game, v);
      return v;
    },
    get: (game) => call("get", game),
    namespace, LOCAL,
  };
})(typeof window !== "undefined" ? window : globalThis);
