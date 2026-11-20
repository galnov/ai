# Runner AI: an endless runner where you race neural networks

A final project in Software Engineering, Artificial Intelligence track.

A 3D endless runner: run along a railway track that gets faster and faster, jump over hurdles, slide under
overhead bars and change lanes to dodge trains. You race an AI "ghost" on exactly the same track, and whoever
gets farther wins.

There are three AI players, trained in three different ways, and **you are part of the training data**:

```
   you play the app ──► every run is recorded (track seed + key presses)
                                  │
            ┌─────────────────────┼──────────────────────────┐
            ▼                     ▼                          ▼
   1. Imitation           3. Combo (DQfD)              2. DQN
   learns to press what   starts from the recordings,  learns only from its own
   the humans pressed     then keeps practicing        trial and error, from scratch
   (supervised learning)  (both)                       (reinforcement learning)
            └─────────────────────┼──────────────────────────┘
                                  ▼
               race them in the app, compare them in the evaluation
```

In the app you can see what the AI "sees" (a map of the 48 m ahead) and what it is "thinking" (its score for
each of the 5 keys) at every moment.

---

## 1. Problem definition

**The task:** survive as long as possible. Every 50 ms the player chooses one of 5 actions:
*run* (do nothing), *left*, *right*, *jump* or *slide*. The speed rises from 36 km/h to 79 km/h over two
minutes, so the same decisions have to be made faster and faster.

**The research questions of the project:**
1. **Imitation:** can a network learn to play just by copying people? How much play does it need, and
   does the copy end up better or worse than the people it copied?
2. **Reinforcement learning:** can a network that never saw a human play learn on its own, and beat people?
3. **The combination:** does starting from human play save practice? The combo gets *half* the practice of
   plain DQN.

---

## 2. Project structure

```
runner/
├── src/
│   ├── config.py           all the constants: game rules, what the network sees, rewards, training settings
│   ├── game.py             the game (NumPy, many games in parallel) - the twin of app/static/game.js
│   ├── bot.py              a rule-based player: a baseline, and practice recordings
│   ├── recordings.py       saving runs, and replaying them into training data
│   ├── make_bot_demos.py   practice recordings from the bot (to test the pipeline before people play)
│   ├── nets.py             the neural network
│   ├── train_bc.py         1. imitation learning (behavioral cloning)
│   ├── dqn.py              2 + 3. DQN, and the DQfD additions for learning from recordings
│   ├── train_dqn.py        training DQN (and the combo with --combo)
│   ├── players.py          all the players behind one interface
│   ├── evaluate.py         the final comparison
│   ├── export_web.py       packs the trained networks for the web page (docs/model/)
│   └── import_recordings.py  adds runs downloaded from the web page to data/recordings/
├── app/
│   ├── server.py           Flask: the AI ghost, saving recordings, the leaderboard
│   ├── templates/          the page (HTML)
│   └── static/
│       ├── game.js         the game rules in JavaScript (identical to game.py)
│       ├── engine3d.js     a small 3D engine written for this project (WebGL2)
│       ├── app.js          the game loop, controls, drawing and panels
│       └── style.css
├── docs/                   the web version for GitHub Pages: the same game, with no server (section 11)
├── notebooks/train_colab.ipynb   training on Google Colab
├── data/recordings/        the recorded runs (JSON, one file per run)
├── models/                 the trained networks (created by training)
└── reports/                charts and reports (created automatically)
```

---

## 3. Installation and running

You need Python 3.10 to 3.13. Everything works offline: the 3D engine is part of the project.

```bash
python -m venv venv
source venv/bin/activate          # on Windows:  venv\Scripts\activate
pip install -r requirements.txt
```

| Step | Command | Approx. time on a laptop |
|---|---|---|
| 1. Play! Every run is recorded | `python -m app.server` → http://localhost:5000 | as long as you like |
| 2. Imitation learning | `python -m src.train_bc` | a few minutes |
| 3. DQN from scratch | `python -m src.train_dqn` | about 10-15 minutes |
| 4. The combo | `python -m src.train_dqn --combo` | 5-10 minutes |
| 5. Evaluation | `python -m src.evaluate` | a few minutes |
| 6. Race the networks | `python -m app.server` | |

* **Before there are networks**, the AI ghost in the app is the rule-based bot.
* **No recordings yet?** `python -m src.make_bot_demos` creates 40 practice runs from the bot with
  "human-like" mistakes, so you can test the whole pipeline. Training always prefers human recordings when
  there are any, and says loudly when it is using the bot's. For the real experiment, play the app with
  friends or classmates, then delete the `bot_*.json` files.
* **How much to play?** Aim for at least 20-30 minutes of play from several people: more players means more
  variety. The `--curve` experiment (section 6) shows how much the amount of play matters.
* **On Colab:** zip the project folder *with* your recordings and use `notebooks/train_colab.ipynb`.
  Then put the `.pt` files into `models/` on your computer.

---

## 4. The game

| | |
|---|---|
| **Lanes** | 3. Changing lanes takes 0.2 s; pressing twice queues a second change |
| **Hurdle** | jump over it. A jump lasts 0.65 s and is high enough for 0.5 s of it |
| **Overhead bar** | slide under it. A slide lasts 0.6 s |
| **Train** | 8 to 24 m long and too tall to jump: the only way past is another lane |
| **Speed** | 10 m/s at the start, +0.2 m/s every 2 seconds, top speed 22 m/s after 2 minutes |
| **Obstacles** | in rows of 1 to 3, every 0.9 to 1.7 seconds. Every row has at least one way through |

**Fair tracks:** the rule-based bot with perfect timing survives all 50 test tracks for the full 5 minutes,
so every track can be beaten.

### One game, two programs
The game exists twice: in Python (`src/game.py`, for training, many games in parallel with NumPy) and in
JavaScript (`app/static/game.js`, in the browser, so the controls respond instantly). Both use whole numbers
only (centimeters and ticks of 50 ms) and the same small random-number generator (xorshift32), so **the same
seed and the same key presses give exactly the same run in both**. That is why a recording is tiny: just the
seed and the list of `[tick, key]`. When the app saves a recording, the server replays it in Python and
checks that it ends at exactly the same distance. During development this was tested tick by tick on long
runs (over 60,000 ticks, including the observations).

---

## 5. What the network sees, and the network

```
 48 m ahead                       9 numbers about the runner:
   ▲  bin 31   . . .                 which lane (3), lane change still to go,
   │   ...     . T .   T = train     queued lane change, height, how far into the jump,
   │  bin 10   H . T   H = hurdle    slide time left, speed
   │   ...     . B .   B = overhead bar
   │  bin 0    . . .
      lanes:   L M R
```

The track ahead is 32 bins of 1.5 m × 3 lanes × 3 obstacle types = 288 numbers (0 or 1), plus the 9 numbers
about the runner: 297 inputs in total.

| Layer | Size | What it does |
|---|---|---|
| input | 297 | the track ahead + the runner |
| fully connected + ReLU | 256 | combinations such as "a train in my lane in 20 m, and the left lane is free" |
| fully connected + ReLU | 256 | |
| output | 5 | a score for each key |

About 143,000 parameters, so all training runs well on a laptop CPU.

**A design decision worth presenting: why not convolutions?** A hurdle 10 m ahead and a hurdle 30 m ahead
look the same, just shifted, which is exactly what convolutions are good at. We tried a 1D convolutional network
along the track. In tests during development it worked for imitation, but with DQN it learned much more
slowly than the fully connected network (about 150 m after 32,000 training steps, against about 800 m). The
input is small (32 × 3 cells), so the fully connected network is fast enough, and we kept the simpler choice.

The same network is used three times. Only the meaning of the 5 outputs changes:
* **imitation:** the probability that a human would press each key here
* **DQN / combo:** the Q value of each key: how much future reward pressing it is worth

---

## 6. Player 1: imitation learning (`train_bc.py`)

**Data collection:** people play the app. The browser records the seed and every key press, and the server
saves it as JSON. To learn from a recording it is replayed in the Python game, giving pairs of
(what the player saw, what the player pressed) for every tick.

**This is supervised learning**, like classifying images: input = the observation, label = one of 5 keys.

**Problems and solutions:**
| Problem | Solution |
|---|---|
| About 98% of ticks have no key press. A network that never presses anything is "98% accurate" and dies at the first obstacle | **class weights**: rare keys count more in the loss (weight ∝ 1/√frequency) |
| Not much data | **mirror images**: every run is also used flipped left ↔ right, which doubles the data |
| Ticks from the same run are very similar, so a random split would leak | **split by whole runs**: the validation runs are never seen in training |
| Accuracy is not what we care about | each epoch the network **plays** 16 validation tracks, and the epoch that gets farthest is kept |

**A known weakness, worth discussing in the project defense: compounding errors.** People rarely make the
same small mistake twice in a row, so the recordings almost never show how to recover from a bad position. When
the copy makes a small mistake, it ends up in a situation it never saw and has no idea what to do. This is
why imitation alone usually ends up *worse* than the players it copied. (An extension that solves it:
DAgger, in which people correct the network's own runs.)

**Experiment:** `python -m src.train_bc --curve` trains on 10%, 25%, 50% and 100% of the recordings and
plots distance against minutes of play.

---

## 7. Player 2: DQN, reinforcement learning from scratch (`dqn.py`, `train_dqn.py`)

| Concept | In the project |
|---|---|
| **Agent** | the runner |
| **Environment** | the game (32 games in parallel) |
| **State** | the observation (297 numbers) |
| **Actions** | 5: run, left, right, jump, slide |
| **Reward** | +0.1 for every tick alive, −2 for a crash, −0.01 for each key press (no fidgeting) |

**The Bellman equation:** `Q(s, a) ≈ r + γ · Q_target(s', argmax Q(s', ·))`, with γ = 0.99, so the network
looks about 5 seconds ahead. That matters: the reason to change lanes now is a train that only arrives in a
second.

**What makes it work (as in the car project, plus a few new things):**
| What | Why |
|---|---|
| replay buffer (300,000) + target network with soft updates + Double DQN + Huber loss | stable learning |
| **exploration that is mostly "run"** (60% run, 10% each key) | pressing random keys all the time just gets the runner killed, which teaches little |
| **a random starting speed** (as if 0 to 2 minutes had passed) | the network also practices at top speed, not only at the easy start |
| **cut at 2.5 minutes, not "done"** | when the time runs out the future still exists, so the Bellman equation keeps the next state's value |
| **the learning rate goes down to 10%** | steadier at the end of training |
| **test every 2,500 steps and keep the best** | DQN's quality goes up and down during training |

---

## 8. Player 3: the combo, imitation + reinforcement learning (DQfD)

Based on *Deep Q-learning from Demonstrations* (Hester et al., 2018):

1. **Pre-training on the recordings only** (4,000 updates). Two losses:
   * the normal Bellman (TD) loss on the recorded transitions
   * a **margin loss**: the key the human pressed must be worth at least 0.8 more than any other key:
     `max_a [Q(s,a) + 0.8·(a ≠ a_human)] − Q(s, a_human)`. With class weights, as in imitation.
2. **Practice** like DQN, but with **half** the steps, and a quarter of every batch still comes from the
   recordings (with the margin loss).

The idea: start already playing roughly like people, then let the Bellman equation fix their mistakes.

---

## 9. Evaluation (`evaluate.py`)

All players on the same **50 new tracks**, up to 5 minutes each:
* distance per run, and what each run crashed into (hurdle / bar / train)
* a survival curve: what share of the runs are still going after X meters
* learning curves: DQN vs. the combo during training, with imitation as a reference line
* the humans: the distances of the recorded runs, and **head to head**: in the app every human run is on
  the same track as the AI ghost, so the report counts who won

**What to expect.** These numbers come from checks during development. They used a NumPy copy of the same
algorithms and the same network, the bot's practice recordings instead of people, and the 16 validation
tracks. A 2-minute validation run can reach at most about 1,900 m. Your numbers will differ, especially with
real human recordings:

| Player | Distance on the validation tracks |
|---|---|
| Imitation | about 1,000 to 1,250 m, and it changes a lot from epoch to epoch |
| DQN from scratch | under 300 m for the first third of training, then up to about 1,250 m by the end, with ups and downs |
| Combo, after pre-training only | about 1,250 m before it practiced at all |
| Combo, after practice | about 1,800 m after only 5,000 to 10,000 of its 30,000 steps |

* **Imitation** ends up below the "players" it copied: compounding errors in action.
* **The combo** with half the practice of DQN got the farthest. Without the class weights in the margin
  loss, pre-training alone reached only about 80 m, because the network learned to never change lanes.
  This is a good story for the project defense.

**Is it a fair race?** Not completely, and that is worth saying in the defense: the AI sees the exact position of
every obstacle 48 m ahead and reacts within one tick (50 ms), while people need about 250 ms to react. An
extension: give the AI a reaction delay and compare again.

---

## 10. The app

* **Race the AI:** you and the ghost on the same track. Choose the opponent: combo, DQN, imitation or the
  rule-based bot. When you crash, the AI keeps running to the end and you see who won
* **Watch the AI:** the AI plays new tracks at ×1, ×2 or ×4
* **Inside the AI:** the map the network sees (48 m ahead, 3 lanes), its 5 scores at every moment, and the
  key it chose
* **Leaderboard** and the head-to-head score between people and the AI
* **Recording** is on by default (a checkbox turns it off). The name is used for the leaderboard
* **Controls:** ← → or A D change lanes, ↑ W or Space jump, ↓ or S slide. Swipes on touch screens

**The 3D engine** (`engine3d.js`, about 200 lines) was written for the project instead of using a library:
every object in the scene is a box with a model matrix (translate · rotate · scale), a camera
(`lookAt` + perspective projection), simple sun + sky lighting, fog that hides the end of the world, and
patterns (windows, sleepers, warning stripes) computed in the fragment shader. The runner is made of boxes,
with legs and arms animated by angle.

---

## 11. Publishing the app on GitHub Pages

GitHub Pages hosts static files for free. It can't run Python, so the `docs/` folder holds a version of the
app that needs no server. The game already ran in the browser (`game.js`); now the AI does too:

| On the server (Python) | On the web page (JavaScript) |
|---|---|
| the AI ghost plays in `src/game.py` | `docs/backend.js` plays it in its own copy of `game.js`, in chunks of 10 seconds, like the server did |
| `game.observe()`: the 297 numbers | `docs/backend.js`: the same numbers |
| the networks in PyTorch | `docs/mlp.js`: the same layers, in plain JavaScript |
| `src/bot.py` | `docs/backend.js`: the same rules |
| recordings saved to `data/recordings/` | kept in the visitor's browser, with a "Download my runs" button |

**1. Export the trained networks** (after training):
```bash
python -m src.export_web
```
This writes `docs/model/`: `weights.bin` (the networks that exist: combo, DQN, imitation; about 0.6 MB each)
and `model.json` (their layers, and whether the outputs are probabilities or Q values). Training on Colab?
The last cell of the notebook exports them too: extract `docs_model.zip` into the project folder. Without
`docs/model` the page still works, with the rule-based bot as the opponent.

**2. Try it locally:**
```bash
python -m http.server 8000 --directory docs
```
and open http://localhost:8000 (opening `docs/index.html` directly as a file does not load the model files).

**3. Publish:** this project lives in the shared repository of all the projects. Push the repository, and
its workflow publishes this folder's `docs/` at `https://<your-user>.github.io/<repository>/runner/` (the
README at the top of the repository explains the one-time setup).

**Recordings from the web page.** There is no server to save runs to, so they are kept in the visitor's
browser, and the leaderboard shows the runs played in that browser. "Download my runs" (under the
leaderboard) saves them to `my_runs.json`. Then:
```bash
python -m src.import_recordings my_runs.json
```
replays every run in the Python game (exactly like the server checks a recording), and saves it to
`data/recordings/`, where training finds it. Importing the same file twice is safe: runs that are already
there are skipped. So friends can play on the published page and send you their file.

**Checked during development:** the JavaScript observation is identical to Python's (all 297 numbers, at
every tick of a 1,500-tick run), the rule-based bot runs exactly the same distance in both, and a network
exported with this script ran the same distance in JavaScript and Python on 9 of 10 test tracks (on the
tenth, two scores were so close that float rounding picked a different key once: 113.3 m against 113.4 m).

---

## 12. Extensions

* **DAgger:** people correct the imitation network's own runs, which addresses compounding errors
* **A human-like AI:** a reaction delay and limited vision, for a fairer race
* **n-step returns and prioritized replay** for DQN: does it learn faster?
* **More game:** coins, power-ups, trains you can run on top of
* **Does the AI play like a person?** Compare when people and each network press keys (how early they jump,
  how often they change lanes)
