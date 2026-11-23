# Air Hockey AI: an opponent that learned by playing against itself

A final project in Software Engineering, Artificial Intelligence track.

A 3D air hockey table in the browser. You hold the red paddle; the blue paddle is a neural network. Nobody
told the network how to play air hockey. It learned with the reinforcement-learning algorithm **PPO**, by
playing millions of points: first against a simple rule-based bot, then more and more against earlier
versions of itself (**self-play**).

On the page:
* **Play:** first to 7 goals, with the mouse or a finger (phones and tablets work too)
* **Opponents:** the fully trained network, the network at earlier stages of its training, and the
  rule-based bot, so you can feel how it improved
* **Watch AI vs AI:** pick two players and watch them play each other

---

## 1. Problem definition

**Goal:** score more goals than the opponent. Each player defends one end of the table and may not cross
the center line.

**Why it is hard:**
* **The opponent fights back.** Unlike a maze or a race track, the "environment" includes another player.
  A strategy that beats one opponent can lose to another
* **Fast physics.** The puck bounces off walls and paddles at up to 6 m/s. A good shot depends on hitting the
  puck at the right angle, at the right moment
* **Continuous control.** Two real numbers (the paddle's desired velocity), 30 times a second
* **Fairness.** The network has the same paddle speed and acceleration limits as a human, and reacts
  67 ms late (it sees the puck and the opponent as they were two decisions ago)

| Concept | In the project |
|---|---|
| **Agent** | the player: a neural network |
| **Environment** | a simulation of the table, the puck and both paddles (`src/physics.py`) |
| **State** (14 numbers) | positions and velocities of its paddle, the opponent's paddle and the puck, seen from its own side, and the two 7-second clocks |
| **Action** (2 continuous numbers) | the velocity the paddle tries to reach, x and y, each between -1 and 1 |
| **Reward** | +10 for scoring, -10 for conceding, +0.6 per meter the puck moves toward the opponent's goal (and -0.6 back), +0.05 for hitting the puck, -3 for breaking the 7-second rule |

---

## 2. Project structure

```
hockey/
├── src/
│   ├── config.py      all the constants: table, physics, observation, rewards, PPO and self-play settings
│   ├── physics.py     the environment: many tables in parallel (NumPy) - paddles, puck, walls, goals, rewards
│   ├── bot.py         the hand-written opponent (rules, no learning)
│   ├── ppo.py         the Actor and Critic networks and the PPO update
│   ├── train.py       training with self-play
│   ├── evaluate.py    every stage against the bot, and the final network against its earlier stages
│   └── export_web.py  packs the trained networks for the web page (docs/model/)
├── docs/              the game (static web page for GitHub Pages)
│   ├── physics.js     the table and the bot in JavaScript (a copy of physics.py and bot.py)
│   ├── mlp.js         runs the trained networks in the browser
│   ├── app.js         the 3D renderer, the controls, the match
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

python -m src.train               # training: 12 million decisions, about 30-60 minutes on a laptop CPU
python -m src.evaluate            # how good is it?
python -m src.export_web          # writes docs/model/ for the web page
```

Or train on Google Colab with `notebooks/train_colab.ipynb` (no GPU needed).

To play locally: `python -m http.server 8000 --directory docs` and open http://localhost:8000.
(The play counter only works on the published site.)

### What you see during training
Every 25 updates there is a line like this one (an example; your numbers will differ):
```
update  300/732 |  4.92M | reward/point   1.35 | vs bot 212:131 (57 no-goal points) | vs first stage 398:2 | bot opponents 41% | std 0.52 |  610 s
```
* **vs bot:** 400 points against the full-speed bot, without exploration noise: goals scored : goals conceded
* **no-goal points:** a 7-second-rule foul, or a rally of 30 seconds with no goal (replayed - in training only)
* **vs first stage:** against the network as it was after the first update (almost random)
* **bot opponents:** the share of training opponents that are the bot (it goes down during training)

---

## 4. The environment (`physics.py`)

The table is 1 m wide and 2 m long. Player 0 defends y = -1 (the near end on the page), player 1 defends
y = +1. The goal mouths are 30 cm wide.

Every decision lasts 1/30 s and is split into 8 physics steps of 1/240 s, so a fast puck can't jump through
a paddle. In every physics step:
1. **Paddles:** each paddle accelerates toward the velocity its player asked for, with a maximum acceleration
   (30 m/s²) and a maximum speed (2.6 m/s). It stays inside its own half
2. **The puck glides** with a little friction (an air table: it keeps 78% of its speed after a second), and
   bounces off the side walls and the end walls (outside the goal mouth), losing 12% of its speed
3. **Paddle hits:** if the puck overlaps a paddle, it is pushed out, and its velocity relative to the paddle
   is reflected along the line between their centers (with 85% bounce). A moving paddle hits harder
4. **Goal:** the puck crossed an end line inside the goal mouth

**Many tables at once:** all the state is stored in NumPy arrays with one row per table, so 64 tables are
simulated together in one set of array operations.

**One network for both sides:** player 1 sees the table turned 180 degrees (every x and y multiplied by -1),
and its actions are turned back. So both players see "my goal is at the bottom", and one network can
play either side. This is what makes self-play possible.

**Reaction time:** the puck and the opponent are seen as they were 2 decisions ago (67 ms), for the bot
and the network alike. A player's own paddle is seen as it is now, just as you feel where your hand is
without looking. (A first version delayed the player's own paddle too. Then both the bot and the network
kept overshooting, because they were steering a paddle they saw where it had been. With the change, the
network's score against the bot rose from about 25% of the goals to about 50% in the same training time.)

**The 7-second rule (as in real air hockey):** a player who keeps the puck in their own half for 7 seconds
commits a foul: -3 for them, and the opponent gets the puck. Both players see the two clocks (inputs 12-13).

Why it is needed: an earlier version only had a "replay after 12 seconds with no goal" rule, with -3 for both
players. The network found an easy way out: when a slow puck was in its half, it stood still in front of
its goal and waited for the replay. Tests found it frozen for 1.5 seconds or more in 1,334 situations in
2 minutes of play on 200 tables, almost all with a slow puck in its own half. With the 7-second rule waiting
costs only the waiting player, and it happened 2 times in the same test. The network also became much
stronger: it now has to attack the puck instead of waiting.

In training a point with no goal and no foul is still replayed after 30 seconds, so every episode ends. The
web page has no replays: thanks to the 7-second rule the puck can't get stuck.

---

## 5. The bot (`bot.py`)

The first opponent, and a baseline. It uses the same observation and the same paddle limits:
* **Defend** when the puck is in the other half or coming fast: stand on the line between the puck and the
  middle of the goal, 18 cm in front of the goal
* **Attack** when the puck is in its half and slow: get behind the puck (on the line from the opponent's
  goal through the puck), then drive through it. If it is in front of the puck, it first steps aside and
  goes around, so it doesn't knock the puck into its own goal. A puck stuck at its end wall is knocked out
  sideways

Its top speed is 70% of the limit: in tests, a faster bot didn't score more (it overshoots its target more
often), and the network gets a rule-based opponent it can learn from.

---

## 6. Learning: PPO with self-play (`ppo.py`, `train.py`)

**The networks:** an Actor (14 → 128 → 128 → 2, tanh) gives the mean of a normal distribution for each of
the two actions; a learned standard deviation adds exploration. A Critic with the same shape estimates the
value of the state. Observations are normalized with a running mean and variance.

**A training round:** 64 tables play 256 decisions each (16,384 samples). The advantage of every action is
computed with GAE (γ = 0.99, λ = 0.95), then the networks are updated 8 times on this data, in minibatches
of 1,024, with the clipped PPO objective (20%). Every point is an episode. (Minibatches of 4,096 meant only
32 gradient steps per round, and learning was clearly slower.)

**The opponents (self-play):** the network always plays player 0. The opponent on each table is chosen
again after every point:

| Opponent | Share |
|---|---|
| the rule-based bot, at a random speed (60-100%) | 90% at the start, going down to 20% |
| the current network itself | 40% of the rest |
| a random earlier version from the "pool" (up to 12, one saved every 25 updates) | the remaining |

Why a pool and not only the current network? Against only itself, a network can chase its own tail: learn
a trick, then learn a defense against it, then forget the trick. Playing earlier versions too keeps old
lessons alive. The bot stays in the mix so the network keeps beating a different style of play.

**Learning stages:** the network is saved after 5, 40, 150 and 400 updates. On the page these are the
"Beginner", "Rookie", "Club player" and "Expert" opponents.

**Which network is the "Champion"?** Self-play goes up and down a little: a version that beats the last
one can be worse against a different style. So every 25 updates the network plays 400 points against the
bot, and in the second half of training the version with the best score is kept as `models/hockey_ppo.pt`.
The last version is saved too (`models/hockey_last.pt`).

---

## 7. Evaluation (`evaluate.py`)

1,000 points per match-up, without exploration noise:
* every stage against the bot at full speed and at 70% speed
* the final network against each earlier stage
* the bot against itself, as a sanity check (close to 50%)

The result is the share of goals each side scored. Training curves are saved in `reports/training_curves.png`.

### Results of one training run (12 million decisions)

| Player | Goals against the bot | Share | The Champion against it |
|---|---|---|---|
| Beginner (0.1M decisions) | 4 : 474 | 1% | 95% of the goals |
| Rookie (0.7M) | 29 : 216 | 12% | 99% |
| Club player (2.5M) | 91 : 82 | 53% | 85% |
| Expert (6.6M) | 308 : 84 | 79% | 60% |
| **Champion** (best against the bot, 8.2M) | 451 : 114 | **80%** | - |
| Last version (12M) | 452 : 150 | 75% | 46% (the last version wins head to head) |

What this shows:
* **It learned a lot:** from losing 4 : 474 to scoring 80% of the goals against a hand-tuned bot, without
  ever being shown how to play
* **Rules shape what is learned:** the same training with the 12-second replay rule reached only 43% against
  the bot, because waiting was a safe choice. With the 7-second rule it had to learn to attack
* **Self-play is not a straight line:** the last version beats the Champion head to head, but does worse
  against the bot. Each version learns to beat the opponents it meets most, which is why the pool of
  earlier versions and the bot stay in the training mix

---

## 8. The web page (`docs/`)

GitHub Pages only serves files, so everything runs in the visitor's browser:
* `physics.js` is a line-by-line copy of `physics.py` and `bot.py` for one table. They were checked against
  each other: the same starting position and actions give the same positions to within 1e-9 for hundreds of
  steps (after that, tiny rounding differences grow, as in any chaotic system with bounces)
* `mlp.js` runs the exported Actor networks (no randomness: the mean action)
* `app.js` draws the table in perspective on a 2D canvas, with no 3D library: every point (x, y, z) is
  projected through a pinhole camera behind your goal. Your pointer is projected back onto the table, and
  your paddle steers toward that point with the same speed and acceleration limits as the AI
* on a phone the camera looks more from above, so the long table fills the tall screen
* a play counter (`../plays.js`) counts matches started by visitors

To update the networks after training: `python -m src.export_web`, then commit `docs/model/` and push.
