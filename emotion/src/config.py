"""
Settings shared by the whole project: paths, list of emotions, image size.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "emotion_cnn.pt"
QUICK_MODEL_PATH = MODELS_DIR / "emotion_cnn_quick.pt"  # --quick mode: not loaded by the app

# below this number of training images, the data probably wasn't loaded correctly
MIN_TRAIN_SAMPLES = 10000

# image size the network receives (as in FER-2013)
IMG_SIZE = 48

# class order of the network (alphabetical - like the folder names in the folder version of FER-2013)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# in the original fer2013.csv the numbers use a different order - this table converts between them
CSV_LABEL_TO_EMOTION = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "sad", 5: "surprise", 6: "neutral",
}

EMOTION_NAMES = {
    "angry": "Angry", "disgust": "Disgust", "fear": "Fear", "happy": "Happy",
    "neutral": "Neutral", "sad": "Sad", "surprise": "Surprise",
}

EMOTION_EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprise": "😲",
}

EMOTION_COLOR = {
    "angry": "#ff4d4f", "disgust": "#7cb342", "fear": "#9c6ade", "happy": "#ffc53d",
    "neutral": "#8c9bab", "sad": "#4096ff", "surprise": "#ff7a45",
}
