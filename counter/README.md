# The Worker: play counters and leaderboards

The "Played N times" counters started out on Abacus, a free counting service that anyone can use. It was
not reliable enough (it is shared by many websites, often answered "Try again later", and allows only 30
requests per 10 seconds from one internet address - a whole class at school shares one address). So the
site now uses its own counter, set up as described here: a **Cloudflare Worker** (a small program on
Cloudflare's servers) with a **D1 database** (where the numbers are kept). It's free (the free plan
allows 100,000 requests a day), takes about 10 minutes in the browser, and needs no programs on your computer.

The same Worker also keeps the **leaderboards** (see below).

| File | What it is |
|---|---|
| `worker.js` | the file to paste into Cloudflare. It is **generated** by `build_worker.py` - don't edit it |
| `worker_src.js` | the Worker's own code: the counters (`/hit/...`, `/get/...`) and the leaderboards (`/lb/...`) |
| `build_worker.py` | builds `worker.js` from `worker_src.js` and the games' code (`python counter/build_worker.py`) |
| `schema.sql` | the database tables: the counters, and the leaderboards' games and scores |

The Worker only accepts your site's namespace and the games in its `GAMES` list (doodle, emotion, car,
runner, maze, hockey, breakout), so nobody can use it for other counters. When you add a game, add its folder
name to `GAMES` in `worker_src.js`, build `worker.js` again and deploy it.
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

## The leaderboards

Six games have leaderboards. Four of them are **checked by the server** (it replays every game):

| Game | Leaderboard | Ranked by |
|---|---|---|
| Breakout | one for all games (a race against the AI, 3 balls) | bricks |
| Runner | **today's track**: the same track for everyone, a new one every day (UTC) | distance |
| Maze | **the maze of the day**: the same 4x4x4 maze for everyone, a new one every day (UTC) | time to the exit |
| Car race | one per preset track (tracks you draw yourself have none) | the time of 3 laps |

Each player appears once on a board, with their best score.

Two more have leaderboards that **can't be checked**, because their scores come from things the server can't
replay (the camera, and the network judging drawings in the browser):

| Game | Leaderboard | Ranked by |
|---|---|---|
| Doodle | one for all games (6 drawings) | points: 100 for each recognized drawing + 5 for every second left |
| Emotion | one for all challenges (5 faces) | the challenge's points (up to 500) |

For these the page sends its score. The server only checks that the score is possible, that the game took at
least its shortest possible time (6 s for Doodle, 25 s for Emotion), and that each game id is used once. So
someone who knows how could post a made-up score there. Hockey has no leaderboard.

### Why a score can't be faked (Breakout, Runner, Maze, car race)
A page never sends a score. Instead:
1. When a game starts, the page asks the Worker for a game id (`POST /lb/start`). The Worker chooses the game's
   random seed (Breakout's serves; today's track and maze come from the date) and writes down the id and the
   time in the `sessions` table.
2. When the game ends, the page sends the id, a name and the player's **moves**: the paddle position of every
   step, the key of every tick, the joystick of every 0.1 s (`POST /lb/submit`).
3. The Worker **replays the whole game** from the moves, with the same game code the page runs (the games'
   files are bundled into `worker.js` by `build_worker.py`), and computes the score itself. The physics of
   these games uses its own `sin`, `cos` and `exp` (`DMath` in the games' code), because the built-in ones may
   differ in the last digit between browsers, and the replay must come out exactly the same.
4. It checks that the game took real time (at least 90% of the game's own time since the id was given), so a
   program can't play a game in a second, and deletes the id, so a game can be submitted only once.

So sending made-up moves gives exactly the score those moves really earn, and changing the game in the
browser (a wider paddle, a slower ball) doesn't help either: the server replays with the real rules.
What no website can stop: a program that plays the real game at normal speed, as a person would.

Replaying is fast: about 1-3 ms for a typical game (the free plan allows 10 ms of CPU per request). The
longest games allowed (a 4-minute maze flight, a 10-minute run) take up to about 6 ms.

### Setting them up (once)
1. **The tables:** in the D1 database's **Console**, paste the contents of `schema.sql` and run it again (the
   counters' table is kept; the two new tables are added).
2. **The code:** in the Worker, **Edit code**, replace everything with the new `worker.js`, and **Deploy**.

Nothing changes in `plays.js` or the pages' addresses: the leaderboards use the same Worker.

**Whenever a game's rules change** (`breakout/docs/game.js`, `runner/docs/game.js`, `maze/docs/ship.js`,
`car/docs/carsim.js`, `car/docs/tracks.json`), run `python counter/build_worker.py` and paste the new
`worker.js` again. Otherwise the server's replays won't match the games and scores will be rejected.

### Looking after them (in the D1 console)
* All the scores of a board: `SELECT * FROM scores WHERE game = 'maze' AND board = '2026-11-24' ORDER BY score;`
* Remove a rude name: `DELETE FROM scores WHERE name = 'the name';`
* Start a board again: `DELETE FROM scores WHERE game = 'breakout';`
