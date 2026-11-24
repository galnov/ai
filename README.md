# Machines That Learn

Final projects in Software Engineering, Artificial Intelligence track: a collection of projects that each
train a neural network from scratch. Every project lives in its own folder, with its own README.

| Folder | Project | How it learns |
|---|---|---|
| [`doodle/`](doodle/) | Guess What I'm Drawing: a CNN recognizes drawings of 42 things | supervised learning |
| [`emotion/`](emotion/) | Emotion Mirror: a CNN reads facial expressions from the webcam | supervised learning |
| [`car/`](car/) | Self-Driving Car: a car learns to drive on tracks it has never seen (DQN) | reinforcement learning |
| [`traffic/`](traffic/) | Smart Intersection: one network counts cars in camera images, another runs the light | supervised + reinforcement |
| [`drone/`](drone/) | A Drone That Learned to Fly: two propellers, wind and packages (PPO) | reinforcement learning |
| [`runner/`](runner/) | Runner AI: race networks in a 3D endless runner | imitation + reinforcement |
| [`maze/`](maze/) | Maze Pilot: a spaceship flies through narrow 3D mazes (PPO) | reinforcement learning |
| [`hockey/`](hockey/) | Air Hockey AI: play 3D air hockey against a network that learned by playing itself (PPO + self-play) | reinforcement learning |
| [`breakout/`](breakout/) | Pixel Breakout AI: race a CNN that learned Breakout from the screen's pixels alone (PPO) | reinforcement learning |

---

## The website (GitHub Pages)

Every project that has a `docs/` folder gets a web page that runs its trained network in the visitor's
browser, plus a home page that lists all the projects:

```
https://<your-user>.github.io/<repository>/            the home page
https://<your-user>.github.io/<repository>/doodle/     a project's page
```

The workflow `.github/workflows/pages.yml` builds the site on every push: it copies `index.html`,
`projects.json` and `plays.js`, and the `docs/` folder of every project, into one site and publishes it.

### Setting it up (once)
1. Create a **public** repository on GitHub (for example `ai-projects`), empty - no README, no .gitignore
2. From this folder (the one with this README):
   ```bash
   git init
   git add .
   git status          # check that no data/raw/ or venv/ folders are listed
   git commit -m "AI projects"
   git branch -M main
   git remote add origin https://github.com/<your-user>/<repository>.git
   git push -u origin main
   ```
3. On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**
4. Open the **Actions** tab: the "Publish the web pages" run takes about a minute. If it ran before step 3
   and failed, click it and choose **Re-run all jobs**
5. The site is at `https://<your-user>.github.io/<repository>/`

### Adding a project
1. Copy the project folder into this repository, next to the others (for example `chess/`)
2. If it has a web version, export its network into its `docs/` folder (see that project's README)
3. Check that its entry in `projects.json` uses the same folder name, and edit the texts if you like. A game
   gets `"game": true`, so the home page shows its play counter
4. If it is a game with a counter, add its folder name to `GAMES` in the Cloudflare Worker (see the counters
   below), or its plays won't be saved
5. `git add .`, `git commit -m "Add chess"`, `git push` - the site updates by itself

The home page shows only the projects whose folders are in the repository: with an **Open** button if the
project has a web page, and "web version coming soon" if it doesn't yet. The games come first, sorted by how
many times they were played, then the other projects.

### Putting the trained networks on the pages
GitHub Pages can't run Python, so every project's page runs its network in the browser, in JavaScript. A page
works without a network too (with the project's hand-written baseline: the autopilot, the PID controller, the
rule-based bot...), and shows a note saying how to add it. To add the trained network, run the export inside
the project's folder, after training, and push:

| Folder | After training | What the export writes |
|---|---|---|
| `doodle/` | `python -m src.export_web` | `docs/model/` |
| `emotion/` | `python -m src.export_web` | `docs/model/` |
| `car/` | `python -m src.export_web` | `docs/model/`, `docs/tracks.json` |
| `traffic/` | `python -m src.export_web` | `docs/model/` (both networks), `docs/scene.png`, `docs/scene.json` |
| `drone/` | `python -m src.export_web` | `docs/model/` |
| `runner/` | `python -m src.export_web` | `docs/model/` (the networks that exist: combo, DQN, imitation) |
| `maze/` | `python -m src.export_web` | `docs/model/` |
| `hockey/` | `python -m src.export_web` | `docs/model/` (the final network and the learning stages) |
| `breakout/` | `python -m src.export_web` | `docs/model/` (the final network and the learning stages) |

Training on Colab? The last cell of every project's notebook runs the export and downloads `docs_model.zip`:
extract it into the project's folder. (Hockey and Breakout already include networks trained with a NumPy
copy of their training code; running their notebooks replaces them with networks trained by the PyTorch code.)

### The "Played N times" counters
The seven games (doodle, emotion, car, runner, maze, hockey, breakout) show how many times they were played
by all visitors together: under each game's title, and on the home page. A game counts once each time someone
starts it: a doodle game, an emotion challenge, a car race, a runner race, a maze flight, a hockey match, a
Breakout race (watching the AI does not count).

GitHub Pages can't store a number, so the counters are kept by a small counter of our own on Cloudflare (a
Worker with a D1 database, free): see [`counter/README.md`](counter/README.md). The page side is one file,
`plays.js` at the top of the repository (every game loads it as `../plays.js`). The counters' names come from
the site's address (`galnov_github_io_ai`/`maze` and so on). Good to know:
* The internet is not always reliable, so `plays.js` doesn't give up: every request is tried again a few
  times, a play that could not be sent is kept in the visitor's browser and sent later (even on their next
  visit), and the last number seen is shown right away while the current one is on its way
* Nothing is counted while testing locally (`localhost`), so your tests don't inflate the numbers
* If the service is down, the games work as usual
* The Worker only accepts this site's namespace and the games in its `GAMES` list. **A new game must be
  added there** (Cloudflare dashboard → the Worker → Edit code → the `GAMES` line → Deploy). Until then its
  counter answers "Unknown counter" and its badge doesn't appear
* The counters are public: anyone could add to them. It's a fun number, not a statistic

### The leaderboards
Breakout, Runner (today's track), Maze (the maze of the day) and the car race (one per preset track) have
leaderboards, kept by the same Cloudflare Worker. These pages never send a score: they send the player's
moves, and the Worker replays the whole game with the same game code to compute the score, so a score can't
be made up with a direct request. Doodle and Emotion have leaderboards too, but their scores can't be
checked (they come from the camera and from the network in the browser), so they could be faked. How it works, setup and moderation: [`counter/README.md`](counter/README.md).

### Before every push
* After training a better network, export it again (the table above)
* Changed the rules of Breakout, Runner, Maze or the car race? Run `python counter/build_worker.py` and paste the
  new `counter/worker.js` into the Worker, or the leaderboards will reject the games
* The big datasets and virtual environments are in `.gitignore`, so they are never uploaded: `*/data/raw/`,
  `emotion/data/` (it also holds the face photos you collected yourself) and `traffic/data/` (the camera
  images can be made again at any time with `python -m src.make_camera_data`). The runner's recordings
  (`runner/data/recordings/`) are small, and they are uploaded: they are that project's training data
* Try a project's page locally: `python -m http.server 8000 --directory maze/docs`, then open http://localhost:8000
  (the play counter doesn't load this way, and nothing is counted - the game works as usual)
