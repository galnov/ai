# Your own counter (optional)

The "Played N times" counters started out on Abacus, a free counting service that anyone can use. It was
not reliable enough (it is shared by many websites, often answered "Try again later", and allows only 30
requests per 10 seconds from one internet address - a whole class at school shares one address). So the
site now uses its own counter, set up as described here: a **Cloudflare Worker** (a small program on
Cloudflare's servers) with a **D1 database** (where the numbers are kept). It's free (the free plan
allows 100,000 requests a day), takes about 10 minutes in the browser, and needs no programs on your computer.

| File | What it is |
|---|---|
| `worker.js` | the Worker: `/hit/<namespace>/<game>` adds 1, `/get/<namespace>/<game>` reads (the same addresses as Abacus) |
| `schema.sql` | the database table: one row per game |

The Worker only accepts your site's namespace and the games in its `GAMES` list (doodle, emotion, car,
runner, maze, hockey, breakout), so nobody can use it for other counters. When you add a game, add its folder
name to `GAMES` and deploy again.
Adding 1 is a single database statement, so two players finishing at the same moment can't lose a play.

## Setup (in the browser)

1. **An account:** sign up at https://dash.cloudflare.com (free, no credit card).
2. **The database:** in the dashboard, go to **Storage & Databases → D1 SQL database → Create database**.
   Name it `plays`. Open it, go to the **Console** tab, paste the contents of `schema.sql` and run it.
3. **The Worker:** go to **Compute → Workers & Pages → Create → Start with Hello World**. Name it
   `plays-counter` and select **Deploy**.
4. **Connect them:** open the Worker, go to **Bindings** (or **Settings → Bindings**) **→ Add → D1 database**.
   Variable name: `DB`; database: `plays`. Save.
5. **The code:** select **Edit code**, replace everything with the contents of `worker.js`, and select
   **Deploy**. If your GitHub user or repository name is different, change `NAMESPACE` at the top first
   (it's your site's address with `_` instead of `.` and `/`, e.g. `galnov_github_io_ai`).
6. **Check it:** open `https://plays-counter.<your-subdomain>.workers.dev/hit/galnov_github_io_ai/maze` in
   the browser (the address is shown on the Worker's page). It should show `{"value":1}`, and `2` when you
   reload. (Those were test plays: in the D1 console, `UPDATE plays SET n = 0 WHERE game = 'maze';` resets it.)
7. **Use it** (already done for this site - `plays.js` points to `plays-counter.galnvk.workers.dev`): in
   `plays.js` at the top of the repository, change the line
   ```js
   const API = root.PLAYS_API || "https://abacus.jasoncameron.dev";
   ```
   to your Worker's address (no `/` at the end):
   ```js
   const API = root.PLAYS_API || "https://plays-counter.<your-subdomain>.workers.dev";
   ```
   and push. The counters start again from 0 (they are new counters), unless you copy the numbers from
   the old ones: `UPDATE plays SET n = 1234 WHERE game = 'maze';` in the D1 console.

To see all the numbers: `SELECT * FROM plays;` in the D1 console.
