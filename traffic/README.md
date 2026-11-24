# Smart Intersection: a traffic light that sees through cameras and learns to manage traffic

A final project in Software Engineering, Artificial Intelligence track.

A complete system with two neural networks that work together:

```
4 cameras at the intersection ──► Network 1: CNN (supervised) ──► how many cars are waiting in each direction
                                                                     │
                                                                     ▼
                         Network 2: DQN (reinforcement learning) ──► keep the light or switch it
```

The app shows two intersections side by side with **exactly the same traffic**: on the left a regular
traffic light, and on the right a traffic light run by the networks. You can see how much waiting time
the AI saves, the four cameras with the network's count vs. the true number (day, dusk, night and rain),
and the agent's decision at every moment.

---

## 1. Problem definition

**Goal:** reduce the time cars spend waiting at the traffic light.

A regular traffic light works on fixed timing: 30 seconds for each direction, even if there isn't a single
car in the green direction while twenty are waiting at red. A smart traffic light needs two things:
1. **To see:** know how many cars are waiting in each direction. This is where **network 1** comes in: it
   counts cars in camera images.
2. **To decide:** when to switch. The decision isn't simple: every switch "costs" 4 seconds of yellow and
   all-red, during which nobody passes. Switch too early and you waste time; switch too late and the queue
   on the other side grows. A good decision now affects the queues a minute from now. That's why we use
   **network 2**, which learns with reinforcement learning.

---

## 2. Project structure

```
traffic/
├── src/
│   ├── config.py             all constants: intersection, cars, traffic light, cameras, training
│   ├── traffic.py            the intersection simulation (NumPy): cars, traffic light, changing demand
│   ├── camera.py             cameras: draw the intersection in perspective, in day/dusk/night/rain
│   ├── make_camera_data.py   data collection: 32,000 automatically labeled camera images
│   ├── counter.py            network 1: a CNN that produces a density map and counts cars
│   ├── train_counter.py      train network 1
│   ├── evaluate_counter.py   evaluate network 1 on intersections not seen in training
│   ├── vision.py             using the trained network 1
│   ├── agent.py              network 2: DQN
│   ├── train_agent.py        train network 2
│   ├── evaluate.py           evaluate the whole system: camera, network 1, network 2 and traffic light
│   └── export_web.py         packs both networks and the camera background for the web page
├── app/                      the web app (Flask, HTML, CSS, JavaScript)
├── docs/                     the web version for GitHub Pages: the same app, with no server (section 9)
├── notebooks/train_colab.ipynb   all the training on Google Colab with a GPU
├── data/                     the image dataset (generated)
├── models/                   the trained networks (generated)
└── reports/                  charts and reports (generated)
```

---

## 3. Installation and running

```bash
python -m venv venv
source venv/bin/activate          # on Windows:  venv\Scripts\activate
pip install -r requirements.txt
```

| Step | Command | Approx. time |
|---|---|---|
| 1. Collect camera data | `python -m src.make_camera_data` | a few minutes (depends on CPU cores) |
| 2. Train network 1 | `python -m src.train_counter` | slow on CPU. Colab with a GPU is recommended |
| 3. Evaluate network 1 | `python -m src.evaluate_counter` | under a minute |
| 4. Train network 2 | `python -m src.train_agent` | a few minutes, even without a GPU |
| 5. Evaluate the system | `python -m src.evaluate` | a few minutes (draws thousands of camera images) |
| 6. The app | `python -m app.server` → http://localhost:5000 | |

> **Training on Colab:** the notebook `notebooks/train_colab.ipynb` runs steps 1 to 5 on a free T4 GPU.
> At the end, download `counter_cnn.pt` and `traffic_dqn.pt` into the `models/` folder on your computer
> and run the app locally.

You can start the app before training too: the simulation and cameras work, and a simple rule runs
instead of the networks.

---

## 4. The simulation (`traffic.py`)

* **Intersection:** 4 directions, one incoming and one outgoing lane in each direction. Cars drive straight.
* **Cars:** every driver follows the **IDM** (Intelligent Driver Model), a standard model in traffic
  engineering. The driver accelerates up to 50 km/h and brakes according to the distance and speed of the
  vehicle ahead or of the traffic light. This creates real queues, a stop wave that travels backward, and
  start-up delay when the light turns green. There are cars, trucks and buses.
* **Traffic light:** two phases (north-south or east-west). A transition = 3 seconds of yellow and one
  second of all-red. A car that is too close to the line when the light turns yellow continues, as in
  real life.
* **Demand:** every 20-minute episode draws a scenario, and the load changes gradually. The busy direction
  can change midway.
* **Sanity check:** with a permanent green, a lane passes about 0.44 cars per second (about 1,600 per hour).
  That is the usual order of magnitude in traffic engineering.
* **Speed:** the simulation is vectorized with NumPy, so 16 intersections run in parallel during training.

---

## 5. Network 1: counting cars from the camera

### Data collection: the simulation produces labeled images
Each direction has a camera on a pole at the stop line, looking along the road toward the approaching cars.
The image is produced from the state of the simulation (`camera.py`):
1. Draw the intersection from above: roads, crosswalks, buildings, trees and cars
2. Warp it with perspective (a homography): nearby cars are large, distant ones small and squeezed together
3. Add shooting conditions: **day** (with shadows), **dusk**, **night** (only headlights, tail lights and
   street lights are visible) and **rain** (blur and drops), plus noise, JPEG compression and small camera
   movements

**The label comes for free:** the simulation knows exactly how many cars are in the incoming lane, up to
120 meters from the line. `make_camera_data.py` runs intersections with different traffic lights (some bad
on purpose, so there are long queues too) and shoots 8,000 states × 4 cameras, i.e. 32,000 images.

**What makes the task challenging:**
* distant cars are very small and packed together
* at night only lights are visible
* cars in the outgoing lane, and cars crossing the intersection, appear in the image but are not counted
* **the test set is shot at 10 intersections that did not appear in training**, with different buildings
  and trees. This checks that the network learned to count cars and not to remember backgrounds.

### The network (`counter.py`)
| Layer | Output | Role |
|---|---|---|
| Input | 1 × 112 × 64 | grayscale image |
| Block 1 | 32 × 56 × 32 | edges, stripes, spots of light |
| Block 2 | 64 × 28 × 16 | shapes of cars and headlights |
| Block 3 | 128 × 28 × 16 | two convolutions and two **dilated** convolutions (dilation 2 and 4) that see a large area without shrinking the map |
| Head | 1 × 28 × 16 | **density map**: how many cars are in each region. Its sum is the count |

**Why a density map and not a direct number?** In training the network only gets the final number, but it
is forced to learn *where* the cars are. In the app you see the map as a colored layer over the camera,
showing that the network "looks" at the incoming lane and not at the outgoing lane.

**Training:** Huber loss on the count, AdamW, OneCycle LR and data augmentation (brightness, contrast,
gamma and noise). No horizontal flip, because the incoming lane is always on the left. The success metric
is **MAE**: by how many cars the network is off on average.

---

## 6. Network 2: the agent that manages the traffic light (DQN)

| Concept | In the project |
|---|---|
| **Agent** | the traffic light |
| **Environment** | the intersection simulation |
| **State** (6 numbers) | how many cars in the two green lanes, how many in the two red lanes (from network 1), whether mid-transition, how long it has been green |
| **Action** (every 5 seconds) | keep or switch |
| **Reward** | minus the number of waiting cars (averaged over the 5 seconds) |

**A trick that makes learning easier:** the state is ordered *relative to the current phase*, i.e. "the
green ones" and "the red ones" rather than "north" and "east". This way the agent learns one rule that fits
both phases.

**The network:** 6 → 64 → 64 → 2, with ReLU. Learning follows the Bellman equation with a replay buffer, a
target network with soft updates, Double DQN and Huber loss. γ = 0.95, so the agent "looks" about a minute
and a half ahead. 16 intersections learn in parallel, and every 1,000 decisions the agent is tested on 6
fixed validation scenarios, and the best one is kept.

**A continuing task:** an intersection has no "end". When a 20-minute episode finishes, it is not marked as
terminal, so the agent doesn't learn that "everything ends soon, so the queue can be neglected".

---

## 7. Evaluating the system (`evaluate.py`)

On exactly the same 6 test scenarios (20 minutes each):

| Controller | What it sees |
|---|---|
| Fixed light | nothing. 30 seconds for each direction |
| "Longest queue" | the exact count. Switches when red has more cars (with at least 10 seconds of green) |
| DQN with a perfect sensor | the exact count |
| DQN with cameras | network 1's count, in day, dusk, night and rain |

**Expected results:** in our tests during development, the agent cut the average waiting time per vehicle
**roughly in half** compared to a fixed light, and by tens of percent compared to "longest queue". When the
count is noisy, performance drops but is still better. The report `reports/system_report.txt` will give
the exact numbers on your machine.

### A recommended experiment for the project portfolio: training with noise
```bash
python -m src.train_agent --noise auto
```
Trains the agent with counting noise the size of network 1's real error (taken from
`counter_metrics.json`). Does an agent trained "with noise" cope better with the cameras, especially at
night? Run `evaluate.py` before and after and compare.

---

## 8. The app

* **Controls:** traffic scenario (normal, heavy from the north, rush hour, quiet night, alternating waves,
  or manual with sliders), camera weather, what the AI counts with (cameras or a perfect sensor), which
  regular traffic light to compare against, and speed (×1, ×4, ×16)
* **The two intersections:** cars move in real time, with brake lights when they stop, the light's color
  at the stop line, and the number of waiting cars in each direction
* **Percentage saved** in waiting time
* **The cameras:** the four cameras of the AI intersection, with the density map and the network's count
  vs. the true number
* **A chart** of waiting cars over time at both intersections
* **The decision:** what the agent sees, the Q values of both actions, and what it chose

---

## 9. Publishing the app on GitHub Pages

GitHub Pages hosts static files for free. It can't run Python, so the `docs/` folder holds a version of the
app that needs no server. `docs/backend.js` answers the same addresses the Flask server answered
(`/api/step`, `/api/config`...), but inside the page:

| On the server (Python) | On the web page (JavaScript) |
|---|---|
| `src/traffic.py` + `app/world.py`: the cars, the lights, the two intersections | `docs/sim.js` |
| `src/camera.py`: drawing the camera images | `docs/camera.js`: the background is drawn once by Python (`docs/scene.png`), the cars are drawn pixel by pixel like PIL draws them, then the same perspective warp, conditions, noise and JPEG |
| network 1 (the camera CNN) in PyTorch | `docs/counter.js`, in four background threads (one per camera) |
| network 2 (the DQN agent) in PyTorch | `docs/mlp.js` |

**1. Export the trained networks** (after training):
```bash
python -m src.export_web
```
This writes `docs/model/`: `weights.bin` (both networks, about 2.3 MB) and `model.json` (their layers and
validation results). BatchNorm in the camera network is folded into the convolutions (in evaluation mode it
is just a fixed scale and shift), so the browser computes exactly the same network with fewer steps. It also
writes `docs/scene.png` and `docs/scene.json`, the intersection the cameras see. Training on Colab? The last
cell of the notebook exports them too: extract `docs_model.zip` into the project folder. Without
`docs/model` the page still works: the AI light uses the simple rule with a perfect sensor.

**2. Try it locally:**
```bash
python -m http.server 8000 --directory docs
```
and open http://localhost:8000. (Opening `docs/index.html` directly as a file does not work: browsers
don't allow the background threads and the model files from `file://`.)

**3. Publish:** this project lives in the shared repository of all the projects. Push the repository, and
its workflow publishes this folder's `docs/` at `https://galnov.github.io/ai/traffic/` (the
README at the top of the repository explains the one-time setup).

**What is different on the web page:** the camera network is the heaviest part, about 400 million
multiplications per image, 4 images every 5 seconds of simulation. On a laptop that takes a fraction of a
second, so ×1 and ×4 run smoothly; at ×16 the simulation waits for the cameras. "Perfect sensor" still
runs the cameras (to show them), like the server does.

**Checked during development:** 3,000 steps (25 minutes) of the same arriving cars in JavaScript and in
Python: every car in the same place to within 10⁻¹³ m, the same camera counts at every step, the same
number of cars served and the same total waiting time. The top-down drawing is identical pixel for pixel,
and the camera views after the perspective warp differ by 0.07 out of 255 on average. The JavaScript camera
network gives the same count as the network computed with PyTorch's rules, to within 0.00001 cars.

---

## 10. Limitations and next steps

* **Sim-to-real:** network 1 was trained on synthetic images. It works well on the simulation's cameras,
  but not on real footage of an intersection. Options going forward: fine-tuning on a few dozen
  hand-labeled real images, or starting from a pretrained detection network (like YOLO).
* **A simple intersection:** no turns, no pedestrians and no multiple lanes. Each of these is a possible
  extension.
* **A network of intersections:** several intersections that coordinate, e.g. a "green wave", with an agent
  per intersection.
