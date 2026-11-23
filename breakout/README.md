# Pixel Breakout AI: a network that learned to play from the screen's pixels

A final project in Software Engineering, Artificial Intelligence track.

In 2013, DeepMind showed that one neural network could learn Atari games, Breakout among them, from
nothing but the screen and the score. That result started modern deep reinforcement learning. This project
is a small version of the same experiment: a **convolutional neural network (CNN)**, trained with
**PPO**, learns Breakout from the raw pixels of a 48 × 48 screen. It is never told where the ball or the
paddle is.

On the page:
* **Race the AI:** you and the network play side by side, with the same bricks, the same serves and 3
  balls each. Whoever breaks more bricks wins
* **What the AI sees:** the two screens the network gets, exactly as it gets them, and a **Grad-CAM**
  heat map of where on the screen its decision comes from
* **Learning stages:** play or watch the network at earlier points of its training
* **Watch the AI:** the network plays alone

---

## 1. Problem definition

**Goal:** break as many bricks as possible with 3 balls.

**What makes it different from the other projects:** they all give the network numbers that describe the
world (distances, positions, speeds). Here the network gets **only pixels**, like a person looking at the
screen. It has to discover on its own which pixels are the ball and which are the paddle, and how the ball
moves.

| Concept | In the project |
|---|---|
| **Agent** | the player: a convolutional neural network |
| **Environment** | the Breakout game (`src/game.py`) |
| **State** | the last 2 screens, 48 × 48 gray pixels each (4,608 numbers from 0 to 255) |
| **Action** (discrete) | stay, left or right |
| **Reward** | +1 for every brick, -1 for a lost ball |

**Why 2 screens?** One screen shows where the ball is, but not where it is going. The difference between
two screens shows the direction and the speed. (DeepMind used 4.)

---

## 2. Project structure

```
breakout/
├── src/
│   ├── config.py      all the constants: the screen, bricks, speeds, rewards, PPO settings
│   ├── game.py        the game: many boards at once (NumPy) and the 48 x 48 screen
│   ├── ppo.py         the CNN (policy + value) and the PPO update
│   ├── train.py       training
│   ├── evaluate.py    every stage against a random player and a scripted player
│   └── export_web.py  packs the trained networks for the web page (docs/model/)
├── docs/              the game (static web page for GitHub Pages)
│   ├── game.js        the game in JavaScript (a copy of game.py)
│   ├── cnn.js         runs the CNN in the browser, and computes Grad-CAM
│   ├── app.js         the boards, the controls, the race, "what the AI sees"
│   ├── style.css
│   └── model/         the exported networks (model.json + weights.bin)
├── notebooks/train_colab.ipynb
├── models/            the trained networks (created by training)
└── reports/           training curves (created automatically)
```

---

## 3. Installation and running

You need Python 3.10 to 3.13.

```bash
python -m venv venv
source venv/bin/activate          # on Windows:  venv\Scripts\activate
pip install -r requirements.txt

python -m src.train               # 10 million decisions; with a GPU about an hour
python -m src.evaluate
python -m src.export_web          # writes docs/model/ for the web page
```

Or train on Google Colab with `notebooks/train_colab.ipynb` (choose a GPU runtime).

To play locally: `python -m http.server 8000 --directory docs` and open http://localhost:8000.
(The play counter only works on the published site.)

---

## 4. The game (`game.py`)

The screen is 48 × 48 pixels: 5 rows of 8 bricks at the top, a paddle 8 pixels wide at the bottom.
Every decision lasts 1/15 s and is split into 4 physics steps, so the ball can't jump over a brick:

1. the paddle moves (66 pixels per second, the same for you and the AI)
2. the ball moves and bounces off the side walls and the top
3. a brick under the ball breaks; the ball bounces up or down if it came from the row above or below,
   sideways if it came from the side. The ball gets 2% faster with every brick (up to 50 pixels per second)
4. **aiming:** where the ball lands on the paddle decides the angle it goes back up, from straight up in
   the middle to 60° at the edges. This is the only way a player controls the ball
5. a ball that falls below the paddle is lost; after 3 the game ends. A cleared wall is replaced by a new one

**The screen** is drawn by the same code that runs the game: black background, bricks in a different gray
in each row, the paddle and the ball in white. The JavaScript copy (`docs/game.js`) draws exactly the same
pixels, so the network sees the same thing in the browser as in training.

**Many boards at once:** all the state is in NumPy arrays with one row per board, so 64 games run together.

---

## 5. The network and PPO (`ppo.py`, `train.py`)

```
2 screens 48x48 ──► conv 6x6, stride 3, 16 filters + ReLU ──► 15x15x16
                ──► conv 3x3, stride 2, 32 filters + ReLU ──► 7x7x32
                ──► dense 256 + ReLU ──┬──► policy: 3 numbers -> probabilities of stay / left / right
                                       └──► value: how many points to expect from here
```

**Why convolutions?** A ball looks the same wherever it is on the screen. A convolutional filter slides over
the whole screen, so a filter that learned to find the ball in one corner finds it everywhere. A dense layer
on 4,608 pixels would have to learn every position separately, and would need far more weights.

**A training round:** 64 boards play 128 decisions each (8,192 samples), choosing actions by sampling from
the probabilities (exploration). The advantage of every action is computed with GAE (γ = 0.99, λ = 0.95).
A lost ball ends the "episode" for learning: the moment it should learn to avoid. Then 4 passes over the data
in minibatches of 1,024 with the clipped PPO objective (10%) and a small entropy bonus (0.01), which keeps it
trying different actions.

**Learning stages:** the network is saved after 0.2, 1 and 3 million decisions, and the best version in the
second half of training (32 games without randomness every 25 updates) is the final one.

---

## 6. Grad-CAM: where does it look?

To check that the network really "sees" the ball, the page shows a heat map over the screen, computed with
**Grad-CAM**: take the gradient of the chosen action's score with respect to the 32 maps of the last
convolutional layer (7 × 7 each), average it over each map to get the map's weight, and add the maps up with
these weights. Places that pushed the decision are bright. In a trained network the heat follows the ball
and the paddle; in an early stage it is spread over the bricks.

---

## 7. Evaluation (`evaluate.py`)

64 full games per player, without randomness, compared with:
* a **random** player
* a **scripted** player that reads the ball's position straight from the game's memory and moves toward it.
  The network has to find the ball in the pixels

### Results of one training run (5 million decisions)

64 games each. A game ends after 3 lost balls or after 200 seconds of play.

| Player | Bricks per game | Games that reached 200 s | Balls lost per game |
|---|---|---|---|
| random | 4.3 | 0% | 3.00 |
| network after 0.2M decisions | 18.0 | 0% | 3.00 |
| network after 1M | 113.2 | 45% | 2.34 |
| network after 3M | 156.7 | 98% | 0.53 |
| **final network** (4.9M) | **156.0** | 100% | 0.42 |
| scripted: reads the ball's x from memory | 135.1 | 100% | 0.08 |

What this shows:
* **It learned from pixels alone:** from 4 bricks (random) to 156, and almost never losing a ball
* **It beats the scripted player** that "cheats" by reading the ball's position: the scripted player almost
  never misses, but it only follows the ball, while the network learned to **aim**, hitting the ball with
  the edge of the paddle to send it where the bricks are. It takes a little more risk (0.42 lost balls
  against 0.08) and breaks 15% more bricks in the same time
* **Most of the learning happens early:** between 0.2M and 1M decisions it goes from 18 to 113 bricks. The
  last 2 million decisions mostly make it safer

---

## 8. The web page (`docs/`)

GitHub Pages only serves files, so everything runs in the visitor's browser:
* `game.js` is a line-by-line copy of `game.py` for one board. They were checked against each other: the same
  actions give the same screens, pixel for pixel, for hundreds of steps. After that, tiny differences between
  Python's and JavaScript's `sin` and `cos` (about 1e-12) can change which pixel the ball is drawn on
* `cnn.js` runs the exported CNN (no libraries, about 3 ms per decision) and computes Grad-CAM. Its output
  matches NumPy to within 1e-8
* the weights are stored as float16, which halves the download
* in a race both boards get the same serves (a seeded random generator), so the comparison is fair
* a play counter (`../plays.js`) counts races started by visitors

To update the networks after training: `python -m src.export_web`, then commit `docs/model/` and push.
