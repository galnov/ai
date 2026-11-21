# Self-Driving Car: A Car That Learns to Drive with Reinforcement Learning

Final project in software engineering with a specialization in artificial intelligence.

A car in a 2D racing game learns to drive on its own, **without anyone showing it how**. It sees the world only
through 9 distance sensors, and it gets "points" when it makes progress along the track and a "penalty" when it
hits a wall. A deep neural network (DQN) learns by trial and error which action pays off in every situation.

In the app you can:
* **Watch** the trained driver on preset or random tracks, and see the sensors and the speed trail
* **Draw a track** with the mouse, and watch the driver drive on a track it has never seen
* **Race** against the AI in a 3-lap race, using the arrow keys
* **Compare learning stages**: cars with the "brain" from different points in training, from a network that knows nothing to a trained driver
* **Peek into the brain**: a live visualization of the network, with the neuron activations and the Q value of each action

---

## 1. Problem definition

**The goal:** teach an agent to drive around a closed track as fast as possible without crashing, and to succeed on new tracks too.

**Why reinforcement learning and not supervised learning?** We don't have a dataset of "this situation and the
correct action for it". All we have is an outcome: did we make progress or did we crash. The agent has to discover
on its own that braking now pays off, because it prevents a crash a second later. This is a **temporal credit
assignment** problem, and that's exactly what reinforcement learning is for.

| Concept | In this project |
|---|---|
| **Agent** | the driver: a neural network |
| **Environment** | the simulation: track, car physics, sensors (`src/env.py`) |
| **State** | 10 numbers: 9 distances to the walls (0 to 1) and the speed (0 to 1) |
| **Actions** | 6: gas, gas + left, gas + right, left, right, brake |
| **Reward** | progress along the track, minus a small time penalty. A crash: -5 and the episode ends |
| **Episode** | starts at a random point and ends with a crash, a long stop, or after 75 seconds |

---

## 2. Project structure

```
car/
├── requirements.txt
├── src/
│   ├── config.py        all the constants: physics, sensors, rewards, training settings
│   ├── track.py         tracks: random generation, building from a drawing, smoothing and validity checks
│   ├── env.py           the learning environment (NumPy, vectorized: many cars in parallel)
│   ├── agent.py         the Q network, the experience memory and the DQN agent (PyTorch)
│   ├── train.py         training
│   ├── validation.py    running a driver on a track and measuring performance
│   ├── evaluate.py      evaluation on new tracks and plots
│   └── export_web.py    packs the trained drivers for the web page (docs/model/)
├── app/
│   ├── server.py        Flask server
│   ├── sim.py           the simulation that runs while the app is in use
│   ├── templates/       the user interface (HTML)
│   └── static/          styling (CSS) and drawing (JavaScript)
├── tracks/              5 preset tracks
├── docs/                the web version for GitHub Pages: the same app, with no server (section 8)
├── notebooks/train_colab.ipynb   training on Google Colab (optional)
├── models/              the trained networks (created by training)
└── reports/             plots and reports (created automatically)
```

---

## 3. Installation and running

You need Python 3.10 to 3.13.

```bash
python -m venv venv
source venv/bin/activate          # on Windows:  venv\Scripts\activate
pip install -r requirements.txt
```

### Step A: Training

```bash
python -m src.train
```

Training takes a few minutes on a regular computer, and you don't need a graphics card: the network is small and
the CPU is fast enough. During training two kinds of lines are printed:

```
step  5,000/40,000 | epsilon 0.88 | average reward    62.3 | crashes  41% | loss 0.0812 |    31 s
    validation: 2.95 laps in 30 seconds  * best network so far - saved
```

**Validation** = how many laps the driver completes in 30 seconds, on average over 10 fixed tracks it doesn't train on.
A good driver reaches about 3. The network with the highest score is saved to `models/driver.pt`.

> Want a quick check that everything works? `python -m src.train --steps 3000` (under a minute).
> The resulting driver will be weaker, so run the full training afterwards.

> **Training on Colab (optional):** the notebook `notebooks/train_colab.ipynb` runs training and evaluation in the cloud,
> so your computer stays free. Upload the project zip, run the cells, and download `models.zip` at the end.
> No GPU is needed: the network is small and most of the time goes to the simulation, which runs on the CPU.

### Step B: Evaluation

```bash
python -m src.evaluate
```

The driver is tested on 30 new random tracks, on 15 "hard" tracks (narrow, with sharp turns) and on the 5 preset
tracks. Each run lasts up to 75 seconds and starts from a standstill. Created in `reports/`:

| File | What you see |
|---|---|
| `eval_report.txt` | a table: how many tracks were completed without crashing, how many laps, lap time |
| `training_curves.png` | reward, validation score, loss, epsilon and crash rate during training |
| `trajectories.png` | the path the car drove, colored by speed. You can see where it brakes |
| `learning_stages.png` | how well the "brain" from each stage of training drives |
| `actions.png` | how often the driver uses each action |

### Step C: The app

```bash
python -m app.server
```
and open in the browser: **http://localhost:5000**

---

## 4. The environment

### Physics (`env.py`)
At every step (0.05 seconds) the car receives an action. Gas adds speed, braking reduces it, and there is friction.
The steering turns the car, but:
* You can't turn on the spot from a standstill (you need speed to turn)
* **At high speed steering is 30% weaker**, so you must slow down before sharp turns. The agent has to discover this on its own.

### Sensors
9 rays leave the car at angles between 90° left and 90° right, up to a distance of 260. We compute the intersection
of each ray with every wall segment (ray-segment intersection, using the cross product) and take the nearest one.
The computation is vectorized in NumPy, so 16 cars × 9 rays × about 700 wall segments are computed together.

### Random tracks (`track.py`)
If the agent trains on a single track, it will "memorize" it. So every 500 steps a new track is generated, of two kinds:
* **Fourier**: an ellipse whose radius changes by a sum of cosine waves. Gentle turns.
* **Polygon**: random points that get smoothed. Sharp turns that require braking.

Every track is smoothed until it has no turns that are too sharp for its width, and checked so it doesn't cross itself.
The same code builds the track the user draws.

### Position on the track and reward
We find the nearest point on the center line (with a local search around the previous position, so we don't "jump"
to another part of the track that passes nearby). Progress is the change in distance along the center line. The
reward is about 1 per step at top speed: this way the agent learns both to drive fast and not to crash.

---

## 5. The network and the algorithm: Deep Q-Network

### The network
| Layer | Size |
|---|---|
| Input | 10 (9 sensors and the speed) |
| Hidden layer 1 | 128, ReLU |
| Hidden layer 2 | 128, ReLU |
| Output | 6: a Q value for each action |

**Q(s, a)** = the network's estimate of "how much reward I'll get from now on if I choose action a in state s and keep driving well".
The agent simply picks the action with the highest Q.

### Learning: the Bellman equation
```
Q(s, a)  ←  r + γ · Q_target(s', argmax Q(s', ·))
```
"The value of what I did = the immediate reward + the discounted value of the state I reached".
γ = 0.99, meaning the agent "looks" about 100 steps (5 seconds) ahead. That's enough to understand that braking now
prevents a crash later.

### What makes it stable

| Component | Why |
|---|---|
| **16 cars in parallel** | the same network drives 16 cars, so 16 times more varied experiences are collected at every step |
| **Replay buffer** (a memory of 200,000 experiences) | learn from a random sample of past experiences. This breaks the dependency between consecutive steps and stops the network from "forgetting" old situations |
| **Epsilon-greedy** | at first 100% random actions (exploration), dropping to 3%. Without exploration the agent won't discover better actions |
| **Target network** with soft updates (τ = 0.005) | the right side of the equation is computed with a copy that updates slowly, so the "target" doesn't keep moving |
| **Double DQN** | the main network chooses the next action and the target network evaluates it. Prevents over-estimating Q |
| **Huber loss** and gradient clipping | one large error doesn't "blow up" the update |
| **Validation and keeping the best network** | in DQN performance jumps around during training, and the last network isn't necessarily the best. We test every 2,500 steps and keep the best, similar to early stopping |

### Terminated vs. truncated
When the car crashes, there is no future, so the Q of the next state is 0. When time runs out or the track changes,
the future still exists, so we don't mark an end. This distinction matters so the agent doesn't learn that
"time is dangerous".

---

## 6. Expected results

Results vary a little between runs (depending on `--seed`), but usually:
* **A validation score** of about 3 laps in 30 seconds
* **Most new random tracks** are completed for 75 seconds without a crash (about 7 laps)
* On the hard tracks, with sharp turns, there may be some crashes or stops
* Learning is fast: after a few hundred updates the driver is already completing laps. In "Learning stages" in the
  app you can see the transition from a car that zigzags and crashes to a smooth driver

If the result is worse, you can run again with another seed: `python -m src.train --seed 1`.

---

## 7. The app: how it works

```
Browser (JavaScript)                            Server (Python / Flask)
────────────────────                            ───────────────────────
draw a track with the mouse ── POST /api/track ──►  smoothing, validity checks, building walls
choose a mode       ── POST /api/start ──►      create cars (AI / human)
every frame:        ── POST /api/step  ──►      sensors ← network ← action ← physics
  (keys, how many steps)                         (human driver: action from the keys)
                    ◄── positions, sensors, laps, network activations
draw: track, trail, sensors, cars, timing board, "brain"
```

The simulation runs on the server with **exactly the same code** that was used for training, so what you see is what the network really does.

---

## 8. Publishing the app on GitHub Pages

GitHub Pages hosts static files for free. It can't run Python, so the `docs/` folder holds a version of the
app that needs no server. `docs/backend.js` answers the same addresses the Flask server answered
(`/api/track`, `/api/start`, `/api/step`), but inside the page: it is a JavaScript copy of building tracks
(`src/track.py`: from a drawing, and at random), the sensors and the physics (`src/env.py`) and the cars in
the app (`app/sim.py`). The networks run in `docs/mlp.js`. Watching, drawing tracks, racing and the
learning stages all work the same.

**1. Export the trained drivers** (after training):
```bash
python -m src.export_web
```
This writes `docs/model/`: `weights.bin` (the trained driver and the learning stages, about 75 KB each) and
`model.json` (their layers, and for each driver the 14 + 14 neurons drawn in "the brain" panel, picked
exactly like the server picks them). It also writes `docs/tracks.json` from the preset tracks in `tracks/`.
Training on Colab? The last cell of the notebook exports them too: extract `docs_model.zip` into the
project folder. Without `docs/model` the page still works: you can draw tracks and drive yourself.

**2. Try it locally:**
```bash
python -m http.server 8000 --directory docs
```
and open http://localhost:8000 (opening `docs/index.html` directly as a file does not load the model files).

**3. Publish:** this project lives in the shared repository of all the projects. Push the repository, and
its workflow publishes this folder's `docs/` at `https://<your-user>.github.io/<repository>/car/` (the
README at the top of the repository explains the one-time setup).

**On phones and tablets:** in "Race the AI", buttons appear next to the track: ◀ ▶ to steer (left thumb),
Gas and Brake (right thumb), and you can hold several at once. They sit under the track when the phone is
upright, and on both sides of it when the phone is sideways (then the whole track fits on the screen).
Drawing a track works with a finger. A computer with a keyboard works as before.

**Checked during development:** a hand-drawn loop becomes the same track in JavaScript and Python (the
same 284 points, to within 10⁻¹² of a unit), and a drawing with a turn that is too sharp gets the same error
message. Four cars driven by the same network for 600 steps on two preset tracks stayed on exactly the same
paths in both versions (to within 10⁻¹³), with the same number of laps.

---

## 9. Troubleshooting

| Problem | Solution |
|---|---|
| "There is no trained driver yet" | run `python -m src.train`, then restart the server |
| "The track has a turn that is too sharp" | draw wider turns or make the road narrower |
| "The track crosses itself" / "parts are too close" | a single loop, without crossings, with space between parts |
| The human car doesn't move | click "Start race" first and wait for the countdown |
| The driver gets stuck or crashes on a drawn track | drawn tracks can be harder than the training tracks. Make the road wider or run longer training: `--steps 60000` |

---

## 10. Ideas for extensions

* **Compare algorithms:** PPO or A2C vs. DQN
* **More senses:** a sensor for the direction of the next turn. Does the driver get faster?
* **Cars that collide with each other:** a real race between agents
* **Curriculum learning:** start with easy tracks and make them gradually harder
* **Sensitivity check:** what happens with γ = 0.9 or with fewer sensors? (change `config.py`)
