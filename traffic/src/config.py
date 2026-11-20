"""
All the project's constants: the intersection, the cars, the traffic light, the camera and both networks.
Units: meters and seconds.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CAMERA_DATA = DATA_DIR / "camera_dataset.npz"
COUNTER_PATH = MODELS_DIR / "counter_cnn.pt"     # network 1: counting cars from the camera
AGENT_PATH = MODELS_DIR / "traffic_dqn.pt"       # network 2: managing the traffic light

# ---------------------------------------------------------------- the intersection
# 4 incoming lanes: from the north, south, east and west. Every car drives straight.
LANES = ["N", "S", "E", "W"]
LANE_NAMES = {"N": "North", "S": "South", "E": "East", "W": "West"}
N_LANES = 4
PHASE_LANES = [(0, 1), (2, 3)]          # phase 0: north-south green | phase 1: east-west green
ROAD_LENGTH = 250.0                     # cars enter 250 m before the stop line
EXIT_AT = -40.0                         # ...and leave the simulation 40 m after the intersection
VIEW_RANGE = 120.0                      # the camera sees up to 120 m from the stop line

# ---------------------------------------------------------------- cars (IDM model)
DT = 0.5                                # simulation step
MAX_CARS = 64                           # maximum cars in a lane
V0 = 14.0                               # desired speed (50 km/h)
T_HEADWAY = 1.3                         # desired time gap to the vehicle in front
A_MAX = 1.6                             # normal acceleration
B_COMF = 2.5                            # comfortable braking
S0 = 2.0                                # minimum gap when standing
# vehicle types: (name, length, probability, acceleration factor)
VEHICLES = [("car", 4.5, 0.85, 1.0), ("truck", 9.0, 0.10, 0.6), ("bus", 12.0, 0.05, 0.6)]
WAIT_SPEED = 1.0                        # below this speed a vehicle counts as "waiting"

# ---------------------------------------------------------------- traffic light
DECISION_EVERY = 5.0                    # every 5 seconds the agent decides: keep or switch
YELLOW = 3.0
ALL_RED = 1.0
FIXED_GREEN = 30.0                      # a regular traffic light: 30 seconds for each direction

# ---------------------------------------------------------------- demand (cars per second in each lane)
# every episode draws a scenario: constant load, load that moves from one direction to another, traffic waves...
EPISODE_SECONDS = 1200                  # 20 minutes
DEMAND_RANGE = (0.03, 0.28)

# ---------------------------------------------------------------- network 2: DQN
OBS_SIZE = 6            # count in 2 green lanes + 2 red lanes, mid-transition?, how long green (traffic.observe)
N_ACTIONS = 2           # 0 = keep, 1 = switch
ACTION_NAMES = ["Keep", "Switch"]
COUNT_SCALE = 20.0      # normalizing the count for the network
TRAIN = dict(
    n_envs=16,
    total_decisions=12_000,      # decisions of all intersections together x16
    buffer_size=100_000,
    batch_size=128,
    lr=5e-4,
    gamma=0.95,
    eps_start=1.0,
    eps_end=0.02,
    eps_decay_fraction=0.4,
    learning_starts=500,
    tau=0.01,
    hidden=64,
    eval_every=1_000,
    eval_episodes=6,
    seed=0,
)

# ---------------------------------------------------------------- network 1: camera
# each direction has a camera on a pole at the stop line, looking toward the approaching cars.
# one network counts the cars in the incoming lane, in each of the 4 cameras.
CAM_W, CAM_H = 64, 112                  # the image the network receives (grayscale)
VIEW_W, VIEW_H = 192, 336               # the color image shown in the app
CAMERA_FRAMES = 8_000                   # x4 cameras = 32,000 labeled images
CONDITIONS = ["day", "dusk", "night", "rain"]
CONDITION_NAMES = {"day": "Day", "dusk": "Dusk", "night": "Night", "rain": "Rain"}
