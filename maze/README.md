# Maze Pilot: a spaceship that teaches itself to fly through 3D mazes

A final project in Software Engineering, Artificial Intelligence track.

A spaceship flies through a narrow 3D maze of 4-meter corridors that turn left, right, up and down. You
watch from the cockpit: black walls covered in a glowing green grid, fading into the dark. Nobody taught
the ship how to fly. A neural network learned it by trial and error, with the reinforcement-learning
algorithm **PPO**, controlling three continuous values ten times per second: pitch, turn and throttle.

In the app:
* **Watch the AI:** the cockpit view, what the network "sees" (its depth camera), its three controls at
  every moment, and a rotating map of the whole maze with the ship's trail
* **Learning stages:** the same ship with its "brain" from different points in training, from crashing
  into the first wall to flying the whole maze
* **Fly it yourself:** the same ship and the same maze, with the keyboard, racing the AI's ghost. It is much
  harder than it looks, so you can lower your ship's top speed while you practice

---

## 1. Problem definition

**Goal:** fly from the entrance of a 3D maze to its exit as fast as possible, without touching a wall.

**Why it is hard:**
* **The corridors are narrow.** Every corridor is 4 m wide, and the ship is a ball with a 0.6 m radius
* **Inertia.** The velocity follows the nose gradually, like a spaceship drifting through a turn. At top
  speed (11 m/s) the turning radius is 5 m, wider than the corridor, so going into a corner too fast means
  hitting the outside wall. The pilot has to brake at the right moment and accelerate out of the turn
* **Real 3D.** Corridors go up and down too, including vertical shafts
* **Continuous control.** Three real numbers every 0.1 s, not a short list of buttons

| Concept | In the project |
|---|---|
| **Agent** | the pilot: a neural network |
| **Environment** | a simulation of the ship in a maze (`src/ship.py`) |
| **State** (93 numbers) | a 9 × 9 depth image, the direction to the next two cells on the way out, the ship's velocity, the previous action |
| **Action** (3 continuous numbers) | nose up/down, turn left/right (both around the ship's own axes), throttle, each between -1 and 1 |
| **Reward** | +0.5 per meter of progress toward the exit, +20 for the exit, -20 for a crash, a small cost for jerky steering |

---

## 2. Project structure

```
maze/
├── src/
│   ├── config.py      all the constants: maze, physics, sensors, reward, PPO settings
│   ├── maze.py        generating 3D mazes, and the distance / direction to the exit (BFS)
│   ├── ship.py        the environment: 64 ships in parallel (NumPy) - physics, depth camera, reward
│   ├── autopilot.py   a hand-written pilot, for comparison
│   ├── flights.py     flying fixed mazes with any pilot (used by training, evaluation and the app)
│   ├── ppo.py         the Actor and Critic networks and the PPO update
│   ├── train.py       training
│   ├── evaluate.py    comparison with the autopilot, and charts
│   └── export_web.py  packs the trained networks for the web page (docs/model/)
├── app/
│   ├── server.py      Flask: mazes and the AI's flights
│   ├── templates/     the page
│   └── static/
│       ├── neon.js    the WebGL renderer: the neon corridors and the maze map (written for this project)
│       ├── ship.js    the maze and the ship's physics in JavaScript (for flying yourself)
│       ├── app.js     modes, playback, controls, cockpit display
│       └── style.css
├── docs/              the web version for GitHub Pages: the same app, with no server (section 8)
├── notebooks/train_colab.ipynb
├── models/            the trained networks (created by training)
└── reports/           charts and reports (created automatically)
```

---

## 3. Installation and running

You need Python 3.10 to 3.13. Everything runs offline: the 3D renderer is part of the project.

```bash
python -m venv venv
source venv/bin/activate          # on Windows:  venv\Scripts\activate
pip install -r requirements.txt

python -m src.train               # training: 3 million steps, about 10-20 minutes on a laptop CPU
python -m src.evaluate            # comparison with the autopilot
python -m app.server              # http://localhost:5000
```

**No GPU needed.** The networks are small (128 neurons per layer), and most of the time goes into the
simulation. You can start the app before training too; then the ship is flown by the autopilot.

### What you see during training
Every 20 updates there is a line like this one (an example; your numbers will differ):
```
update   60/366 |  0.49M steps | return  112.4 | training: exit  80% crash  12% | validation: exit 100% crash   0% speed  6.2 m/s time  58.1 s  * saved
```
* **Training:** flights with exploration noise, half of them starting somewhere in the middle of a maze
* **Validation:** 32 mazes that are never used in training, flown without noise. The best version is
  saved (most exits, fewest crashes, and speed to break ties)

---

## 4. The environment

### The maze (`maze.py`)
A maze with n × n × n "rooms" lives on a grid of (2n+1)³ cubic cells, 4 m each. Rooms are at the odd
coordinates; a cell between two rooms is either rock or a passage.

* **Generation:** depth-first search ("recursive backtracker"). From the current room, pick a random
  unvisited neighbor in one of the **6 directions**, carve the passage, and continue from there; at a dead
  end, go back. The result is a "perfect" maze: exactly one way between any two cells, with long winding
  corridors and dead ends in all three dimensions. A 4 × 4 × 4 maze has about 410 m of corridor from the
  entrance to the exit
* **The exit:** the room farthest from the entrance
* **BFS from the exit** gives every cell its distance to the exit and the direction to go. This is used
  for the beacon and for measuring progress

### The ship (`ship.py`)
**Spaceship controls**, as in the game Descent: in space there is no "up". The ship has its own three axes:
forward *f* (the nose), up *u* (the top of the screen) and right *r = f × u*, and it turns around them:

```
turn right by a:   f ← f·cos a + r·sin a                            (the nose moves toward the right of the screen)
nose up by b:      f ← f·cos b + u·sin b,   u ← u·cos b − f·sin b   (the nose moves toward the top of the screen)
```

So "right" always means the right side of the screen and "up" the top of the screen, also in vertical
shafts, and nothing turns unless the pilot turns it. After several turns, what was the floor may appear at
the side of the screen; in a square corridor with the same grid on every wall, that looks exactly the same.

| | |
|---|---|
| **Turn / nose up-down** | up to 2.2 rad/s (about 126° per second) |
| **Throttle** | sets a target speed between 2 and 11 m/s |
| **Inertia** | `v ← v + (f·target − v)·(1 − e^(−k·dt))`: the velocity catches up with the nose gradually, so the ship drifts in turns |
| **Collisions** | 26 points on the ship's surface are checked against the maze cells, twice per decision |
| **Roll** (`Q` / `E`) | for human players only, to straighten the view if they want. The network does not need it |

### The sensors
* **The depth camera:** 81 rays in a 9 × 9 grid over 100°. Each ray walks through the grid cell by cell
  until it hits rock (the DDA algorithm, as in classic 3D games), giving the distance to the wall, up to
  24 m. This is how the network knows where the walls are, how close the next corner is and whether
  it is drifting toward a wall
* **The navigation beacon:** the positions of the next two cells on the way out, relative to the ship,
  like a GPS route. It tells the network *where* to go, but not *how*: the walls, the speed and the corners
  are up to it

### The reward
| Component | Why |
|---|---|
| **Progress** `+0.5` per meter along the way out | a signal at every step. Progress is measured smoothly inside each cell (distance of the cell × 4 m, minus how far along the cell the ship is), and going into a dead end counts as negative progress |
| **Exit** `+20` | finishing is the goal |
| **Crash** `−20` and the end of the flight | crashing loses all the future progress too, so the network learns caution |
| **Jerky steering** `−0.02 · Δaction²` | smooth flight, without twitching |

Because the reward is per meter, a faster flight collects the same total sooner. With the discount factor
γ = 0.99, sooner is worth more, so the network also learns to be fast.

### Training data from simulation
* A pool of **1,500 random training mazes** (3 × 3 × 3 and 4 × 4 × 4 rooms), generated once at the start
* **Half of the flights start at a random cell** in the maze, facing the way out. That way the network
  sees every kind of corner early, not only the first ones
* Validation and test mazes come from different seeds and are never used in training

---

## 5. The algorithm: PPO (`ppo.py`, `train.py`)

The same algorithm as in the drone project.

| Network | Input | Output | Role |
|---|---|---|---|
| **Actor** (policy) | 93 | 3 means, and a learned standard deviation | a **normal distribution** for each control. In training we sample from it (exploration); in flight we take the mean |
| **Critic** (value) | 93 | 1 | estimates V(s): how much reward to expect from here on |

Both: 93 → 128 → 128 → output, with tanh.

**A training round** (366 rounds in total):
1. **Collect:** 64 ships × 128 steps = 8,192 samples with the current policy
2. **Advantage (GAE):** `δₜ = rₜ + γ·V(sₜ₊₁) − V(sₜ)`, `Aₜ = δₜ + γλ·Aₜ₊₁` (γ = 0.99, λ = 0.95)
3. **Update:** 10 passes over the data in minibatches of 2,048, with the clipped objective
   `L = −min(r·A, clip(r, 0.8, 1.2)·A) + 0.5·(V − R)²`, where `r = π_new(a|s) / π_old(a|s)`

| Detail | Why |
|---|---|
| observation normalization (running mean and variance) | inputs on a similar scale; saved with the network |
| reward normalization by the standard deviation of the return | the Critic learns values on a stable scale |
| time limit ≠ crash | when time runs out, add V of the last state, because the future still exists |
| orthogonal initialization, a tiny Actor output at first | at the start the ship roughly flies straight, and exploration comes from the noise |
| learning rate decreasing linearly, gradient clipping | stability |

---

## 6. Evaluation (`evaluate.py`)

The network against the **hand-written autopilot** (`autopilot.py`): it steers its nose toward the
beacon with proportional control, and slows down before turns. It does not use the depth camera, so it
does not notice when it drifts toward a wall.

Test sets, never used in training:
* **100 mazes of 4 × 4 × 4 rooms**, the size the network trained on
* **30 mazes of 6 × 6 × 6 rooms**, bigger than anything in training (about 3 times as long), to test
  whether it learned to fly in general or just memorized

Charts: exits / crashes / speed per pilot, speed along one flight (the dips are the corners), and the
learning stages.

**What to expect.** These numbers come from checks during development, using a NumPy copy of the same
PPO and network; your numbers will differ somewhat.

| | Autopilot | Network (NumPy check, 2M steps) |
|---|---|---|
| exits, 32 validation mazes | 72% | 100% |
| exits, 100 test mazes (4 × 4 × 4) | 87% | 100% |
| exits, 30 bigger test mazes (6 × 6 × 6) | 70% | 90% |
| average speed | 6.0 m/s | 7.6 m/s |
| average time to the exit, 4 × 4 × 4 | 64.7 s | 44.6 s |

In the check, nothing reached the exit for the first 80,000 steps, 59% of the validation flights did after
160,000, and all 32 after 410,000. After that the network kept every flight safe while getting faster and
faster (from 6.0 to 7.6 m/s): it learned to brake late, take corners tight and accelerate out of them.

### Two design decisions worth presenting
**1. The controls took three tries.** It is a good story for the project defense, because each version
taught something:
* **Airplane controls with a "flight computer"** that slowly rolled the ship back to level. Combining turns
  and climbs could leave the ship upside down, and then the computer rolled it 180° on its own: the whole
  corridor turned around the player without any key being pressed.
* **First-person-game controls:** the horizon is always level, and turning is around the world's vertical
  axis. But when the nose points straight down a shaft, that axis is the line of sight, so pressing "right"
  just spun the view around the center of the screen instead of turning right.
* **Spaceship controls (the final version):** turning is around the ship's own axes. "Right" always turns
  toward the right of the screen, and nothing ever moves on its own. The price: there is no fixed "up", so
  the floor can end up at the side of the screen, which in a uniform square corridor is invisible. Humans
  get `Q` / `E` to roll if they want to. For the network this version is the simplest: everything is
  measured relative to the ship itself.

During development the scenario that broke the second version (right, then down a shaft, then right) was
tested: the nose ends up exactly where the right side of the screen was, and with no keys pressed for 200
steps the ship does not turn at all.

**2. The physics was tuned against the autopilot.** If the ship is slow, simple rules are enough and there
is nothing for the network to prove. The autopilot reached the exit in all the validation mazes up to a top
speed of 10.5 m/s, but only in 72% at 11 m/s. The project uses 11 m/s: hard enough that the rules make
mistakes, and the network, which sees the walls, does better.

---

## 7. The app

* **Pilot:** the trained network, the learning stages, or the autopilot
* **Maze size:** 3 to 6 rooms per side. Sizes 5 and 6 were never seen in training
* **Watch the AI:** playback at ×1, ×2 or ×4, and a new maze automatically after each flight
* **Fly it yourself:** a start screen shows the controls, and you start with `Space`.
  `↑` `↓` nose toward the top / bottom of the screen, `←` `→` (or `A` `D`) turn toward the left / right of the
  screen, `W` `S` throttle, `Q` `E` roll (optional). Nothing moves unless you press a key: to follow the
  maze, just steer toward the yellow beacon on the screen. You can choose your ship's top speed (5, 6, 7.5 or
  11 m/s - the AI always flies at up to 11). The AI's flight in the same maze is shown as a ghost (the cyan
  triangle) to race against, and at the end you see both results. `R` tries again, `N` gives a new maze
* **The cockpit display:** the reticle (where the nose points), the white ring (where the ship is
  really going: the drift), the yellow beacon (the next cell on the way out), speed and throttle
* **What the pilot sees:** the depth camera exactly as the network receives it, and its three controls

The AI's flights are computed by the server in the same Python simulation the network trained in, and
played back in the browser (on the web version, section 8, they are computed in JavaScript). When you fly yourself, the browser runs a JavaScript copy of the physics
(`ship.js`). During development the two were checked against each other: the same flight to within
a millionth of a millimeter.

### The graphics (`neon.js`)
Written for the project with WebGL2, without a 3D library:
* **The walls:** for every empty cell, every side that touches rock becomes a square (two triangles)
* **The neon grid** is computed in the fragment shader from the position on the wall:
  `distance to the nearest line = |fract(uv − 0.5) − 0.5| / fwidth(uv)`, where `fwidth` is the size of one
  pixel in wall units. So the lines stay sharp at any distance and angle. A soft glow is added around each line
* **Depth:** far walls fade into dark gray fog
* **The exit** glows magenta and pulses
* **The map:** the whole maze as thin lines, with the way out, the trail and the ship, slowly rotating

---

## 8. Publishing the app on GitHub Pages

GitHub Pages hosts static files for free. It can't run Python, so the `docs/` folder holds a version of the
app that needs no server: everything the server did runs in the visitor's browser, in JavaScript.

| On the server (Python) | On the web page (JavaScript) |
|---|---|
| `src/maze.py` makes the mazes | `docs/backend.js`: the same algorithm (DFS, the exit, BFS) |
| `src/ship.py` + `src/flights.py` fly the AI | `docs/backend.js` + `ship.js`: the same sensors (93 numbers) and physics |
| `src/autopilot.py` | `docs/backend.js`: the same rules |
| the network in PyTorch | `docs/mlp.js`: the same layers, in plain JavaScript |

**1. Export the trained networks** (after training):
```bash
python -m src.export_web
```
This writes `docs/model/`: `weights.bin` (the numbers of the network and of the learning stages, under
1 MB) and `model.json` (their layers, the observation normalization and the labels). Only the Actor is
exported: flying uses the mean of its distribution. Training on Colab? The last cell of the notebook exports
it too: extract `docs_model.zip` into the project folder. Without `docs/model` the page still works, flown
by the autopilot.

**2. Try it locally:**
```bash
python -m http.server 8000 --directory docs
```
and open http://localhost:8000 (opening `docs/index.html` directly as a file does not load the model files).

**3. Publish:** this project lives in the shared repository of all the projects. Push the repository, and
its workflow publishes this folder's `docs/` at `https://galnov.github.io/ai/maze/` (the
README at the top of the repository explains the one-time setup).

**On phones and tablets:** "Fly it yourself" gets touch controls: a joystick for the left thumb (it appears
where the thumb lands; push up / down for the nose, left / right to turn), a throttle bar for the right thumb
(up = faster) and two roll buttons. The view gets wider when the phone is upright, and fills the screen when
it is sideways (the better way to fly). A computer with a keyboard works as before.

**What is different on the web page:** the mazes come from a JavaScript random generator, so maze #123 on
the page is not the maze with seed 123 in Python, but it is built by exactly the same rules.

**Checked during development:** on the same 40 mazes, flights computed in JavaScript and in Python ended
the same way every time (the network: 40 exits in both, at 7.6 m/s; the autopilot: 88% in both). The 93
observation numbers agree to within 0.0000001, and the JavaScript network gives the same outputs as NumPy
to within 0.000001.

---

## 9. Extensions

* **Without the beacon:** the network only has the depth camera and must *explore* the maze. That needs
  memory (an LSTM, or a map it builds) and is a much harder research problem
* **Round tunnels:** describe the tunnels with a distance function (distance to the tunnel's axis), use
  sphere tracing for the depth camera and ray marching for the graphics
* **Speed vs. safety:** change the crash penalty (`R_CRASH` in `config.py`) and see how the network's
  speed and crash rate change
* **Moving obstacles:** rotating fans or doors that open and close
* **SAC vs. PPO:** the same environment with another continuous-control algorithm
