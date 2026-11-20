# A drone that learned to fly: reinforcement learning with continuous actions (PPO)

A final project in Software Engineering, Artificial Intelligence track.

A 2D drone with two propellers learns to fly on its own: reach any point in the sky and hover there steadily,
even in wind, in gusts and with a heavy package. Nobody taught it how. A neural network learned by trial and
error, with the **PPO** algorithm, to control the **continuous power** of each propeller, 25 times per second.

In the app:
* **Free flight:** click on the sky and the drone flies there. You can change the wind and gusts, load a
  package mid-flight, and add a **PID drone** (the classic controller) that flies next to it for comparison
* **Delivery route:** mark points between skyscrapers, and the drone flies the route again and again
* **Learning stages:** drones with the "brain" from different points in training, from falling immediately
  to flying perfectly
* **Swarm show:** 40 drones, each with a copy of the same network, form a heart, circle, star or wave
* **Value map:** what the Critic "thinks" of every point in the sky
* **Flight instruments:** artificial horizon, altitude, speed, and the two numbers the network outputs

---

## 1. Problem definition

**Goal:** reach a target point and hover there, from any starting state, in changing conditions.

**Why it's hard:**
* **The drone doesn't move sideways directly.** To fly right, it first has to tilt right (more thrust on
  the left propeller), and then correct immediately, otherwise it keeps rotating and flips over. Every
  action affects both the altitude and the angle
* **Delays:** the motors don't respond instantly, and the drone keeps moving with momentum
* **Disturbances:** wind, gusts and a weight the drone doesn't know about in advance

**Why continuous actions?** The action is two real numbers: the power of each propeller. There is no finite
list of actions as in DQN, so we use an algorithm that outputs a **distribution** over real numbers.

| Concept | In the project |
|---|---|
| **Agent** | the flight controller: a neural network |
| **Environment** | a physics simulation of the drone (`src/drone.py`) |
| **State** (11 numbers) | distance to the target (x, y), velocity, angle (sin, cos), angular velocity, error integral, previous action |
| **Action** (2 continuous numbers) | the power of the left and right propellers. 0 = hover thrust, 1 = maximum, -1 = off |
| **Reward** | progress toward the target, a bonus for precise hovering, a small "alive" bonus, and penalties for spinning and wasting energy |
| **Crash** | hitting the ground, flipping over (more than 80 degrees) or leaving the area: a penalty of -10 and the end of the episode |

---

## 2. Project structure

```
drone/
├── src/
│   ├── config.py      constants: physics, wind, reward, PPO settings
│   ├── drone.py       the environment: physics of 128 drones in parallel (NumPy)
│   ├── pid.py         a classic PID controller for comparison
│   ├── scenarios.py   fixed test scenarios: calm, wind, gusts, heavy payload, storm
│   ├── ppo.py         the Actor and Critic networks and the PPO update
│   ├── train.py       training
│   ├── evaluate.py    comparison with PID and charts
│   └── export_web.py  packs the trained networks for the web page (docs/model/)
├── app/               the web app (Flask and canvas)
├── docs/              the web version for GitHub Pages: the same app, with no server (section 7)
├── notebooks/         Colab notebook (optional)
├── models/            the trained networks (generated)
└── reports/           charts and reports (generated)
```

---

## 3. Installation and running

```bash
python -m venv venv
source venv/bin/activate          # on Windows:  venv\Scripts\activate
pip install -r requirements.txt

python -m src.train               # training: 3 million steps, a few minutes on a regular computer
python -m src.evaluate            # comparison with PID
python -m app.server              # http://localhost:5000
```

**No GPU needed.** The networks are small (64 neurons per layer), and for a project like this the CPU is
faster than a graphics card. You can start the app before training too; then the drone is flown by the PID
controller.

### What you see during training
```
update  160/366 |  1.31M steps | return  114.2 | crashes  2.0% | validation (success/crash %): calm 100/0 wind 100/0 heavy 97/3 storm 97/0  * saved
```
* **Return:** the average sum of rewards per episode. It should rise
* **Validation:** the success rate and crash rate in each condition. **Success** = at the end of the 10
  seconds the drone is less than 0.5 m from the target and its speed is below 0.5 m/s
* Usually after about a million and a half steps the drone already flies well in all conditions

---

## 4. The environment: drone physics (`drone.py`)

A rigid body in a plane, with two propellers 25 cm on either side of the center:

```
Total force:           F = T₁ + T₂               in the drone's "up" direction: (−sinθ, cosθ)
Torque:                τ = (T₂ − T₁) · L          right one stronger: the drone tilts left
Acceleration:          a = F·(−sinθ, cosθ)/m − g − drag·(v − wind)/m
Angular acceleration:  α = τ / I
```

* **Motors:** respond with a 0.05 s delay, like real motors
* **Wind:** a constant wind of up to 4 m/s, with random gusts on top (an Ornstein-Uhlenbeck process:
  "smooth" gusts rather than white noise)
* **Payload:** up to 0.5 kg, i.e. up to 1.5 times the drone's weight. The drone **doesn't know** how much
  it weighs
* **Physics at 100 Hz, decisions at 25 Hz:** 4 physics steps between decisions
* **Randomness in training (domain randomization):** every episode draws the starting position, velocity
  and angle, the wind, gusts, payload and target. Sometimes the target changes midway, like the points of a
  delivery route. This way the network learns to fly in general, not one particular route

### Reward design decisions
| Component | Why |
|---|---|
| **Progress** `2·(d_prev − d)` | a clear signal in the right direction even when the target is far. This is a "potential-based" reward, meaning it doesn't change the best policy, it only makes learning easier |
| **Hover bonus** `0.6·exp(−(d/0.5)²)` | encourages hovering **exactly** on the target and not just getting close |
| **Alive bonus** `0.1` | this way a crash loses all future rewards. Without it, the network might learn that "crashing quickly" is better than accumulating penalties |
| **Penalties** on ω² and on action strength | smooth, economical flight, without spinning and without motor "jitter" |

### Why is there an "error integral" in the observation?
In a constant wind or with a heavy payload, a network that sees only the current state doesn't know it is
being "pushed" all the time. The integral of the error (the same I as in PID) gives it this information. This
way the network and the classic controller receive exactly the same information, and the comparison is fair.

---

## 5. The algorithm: PPO (`ppo.py`, `train.py`)

### Two networks
| Network | Input | Output | Role |
|---|---|---|---|
| **Actor** (policy) | 11 | 2 means, and a learned standard deviation | defines a **normal distribution** for each propeller. In training we sample from it (exploration), and in use we take the mean |
| **Critic** (value) | 11 | 1 | estimates V(s): how much reward to expect from this state onward |

Both: 11 → 64 → 64 → output, with tanh activation.

### A training round (366 rounds in total)
1. **Collection:** 128 drones × 64 steps = 8,192 samples with the current policy
2. **Advantage (GAE):** for each action, compute how much better it was **than expected**:
   `δₜ = rₜ + γ·V(sₜ₊₁) − V(sₜ)`, and `Aₜ = δₜ + γλ·Aₜ₊₁`  (γ = 0.99, λ = 0.95)
3. **Update:** 10 passes over the data, in minibatches of 2,048:
   ```
   r(θ) = π_new(a|s) / π_old(a|s)
   L    = −min( r·A ,  clip(r, 0.8, 1.2)·A )   +   0.5·(V(s) − R)²
   ```
   **Clipping** is the heart of PPO: the new policy can't move more than 20% away from the old one at once.
   This prevents "jumps" that destroy what has already been learned.

### Details that matter for success
| What | Why |
|---|---|
| **128 environments in parallel** | lots of varied experience in each round, and the simulation is cheap |
| **Observation normalization** (running mean and variance) | every input has a similar scale. Saved with the network |
| **Reward normalization** by the standard deviation of the return | the Critic learns values at a stable scale |
| **Truncation vs. termination** | when time runs out (not a crash), add V of the last state, because the future still exists |
| **Orthogonal initialization**, and a very small Actor output at first | at first the network "roughly hovers", and exploration comes from the standard deviation |
| **Linearly decreasing learning rate**, and gradient clipping at 0.5 | stability |
| **Testing and keeping the best one** | every 40 rounds on 4 fixed conditions |

---

## 6. Comparison with a PID controller (`pid.py`, `evaluate.py`)

PID is the classic engineering solution, and it's what most drones in the world use. Our controller is
**cascaded**: a position loop computes a desired acceleration, from which the total thrust and the tilt angle
are computed, and an angle loop computes the torque. The gains were tuned by hand for normal conditions. The
PID doesn't know the payload, and its integral corrects for it over time.

**Expected results** (from tests during development; your numbers will be similar):

| Condition | PPO | PID |
|---|---|---|
| Calm, wind, gusts | ~100% success, **reaches the target within about 2 s** | ~95-100% success, about 3 s |
| Heavy payload | about 94% success, about 5% crashes | about 80% success, about 20% crashes |
| Storm (all together) | about 97% success, almost no crashes | about 85% success, about 6% crashes |
| **Hover accuracy in a constant wind** | about 33 cm from the target | **about 13 cm** |

**The interesting conclusion:** the network is **faster and more robust**: it flies straight and
confidently, and survives a heavy payload and a storm. The PID is **more precise in steady state**, because
its integral completely cancels the error in a constant wind. Each method is good at something different.
This is exactly the kind of discussion worth presenting in the project defense, and you can also propose a
combination (for example, the network + an integral correction).

---

## 7. Publishing the app on GitHub Pages

GitHub Pages hosts static files for free. It can't run Python, so the `docs/` folder holds a version of the
app that needs no server. `docs/backend.js` answers the same addresses the Flask server answered
(`/api/step`, `/api/click`...), but inside the page: it is a JavaScript copy of the physics
(`src/drone.py`), the PID controller (`src/pid.py`) and the four modes (`app/world.py`). The networks run
in `docs/mlp.js`. All four modes, the wind, the gusts, the package and the value map work the same.

**1. Export the trained networks** (after training):
```bash
python -m src.export_web
```
This writes `docs/model/`: `weights.bin` (about 120 KB: the Actor of the trained network and of each
learning stage, and the trained network's Critic for the value map) and `model.json` (their layers,
the observation normalization and the labels). Training on Colab? The last cell of the notebook exports
them too: extract `docs_model.zip` into the project folder. Without `docs/model` the page still works,
flown by the PID controller.

**2. Try it locally:**
```bash
python -m http.server 8000 --directory docs
```
and open http://localhost:8000 (opening `docs/index.html` directly as a file does not load the model files).

**3. Publish:** this project lives in the shared repository of all the projects. Push the repository, and
its workflow publishes this folder's `docs/` at `https://<your-user>.github.io/<repository>/drone/` (the
README at the top of the repository explains the one-time setup).

**Checked during development:** with gusts off, four drones flown for 8 seconds by the same network in
JavaScript and in Python ended up in the same place to within 0.01 mm (with wind and a package too), the
PID controller matched to within 10⁻¹⁴, and the value map matched to the 3 decimals it is sent with.
Gusts are random, so with gusts on the two versions fly differently, as two runs of the server would.

---

## 8. Ideas for extensions

* **Obstacles:** give the drone "distance sensors" (as in the car project) so it learns to avoid buildings
  on its own
* **Landing:** a task of soft landing on a moving platform (a ship)
* **A 3D drone:** 4 propellers, 4 continuous actions
* **Memory:** a network with an LSTM that identifies wind and weight over time on its own, without an integral
* **Improving the PID:** automatic tuning of the gains, and checking whether it closes the gap
