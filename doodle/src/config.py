"""
Shared settings: paths, the categories (dataset name, display name, emoji), amounts of data.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # the files downloaded from Quick Draw
CUSTOM_DIR = DATA_DIR / "custom"    # drawings saved from the app
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "doodle_cnn.pt"

QUICKDRAW_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

IMG_SIZE = 28

# how many drawings to take from each category (out of more than 100,000 in the dataset)
PER_CLASS = 10_000
VAL_PER_CLASS = 1_000
TEST_PER_CLASS = 1_000

# (name in the Quick Draw dataset, display name, emoji)
CATEGORIES = [
    ("airplane", "Airplane", "✈️"),
    ("apple", "Apple", "🍎"),
    ("banana", "Banana", "🍌"),
    ("bicycle", "Bicycle", "🚲"),
    ("bird", "Bird", "🐦"),
    ("book", "Book", "📖"),
    ("butterfly", "Butterfly", "🦋"),
    ("cake", "Cake", "🎂"),
    ("camera", "Camera", "📷"),
    ("car", "Car", "🚗"),
    ("cat", "Cat", "🐱"),
    ("clock", "Clock", "⏰"),
    ("cloud", "Cloud", "☁️"),
    ("cup", "Cup", "☕"),
    ("dog", "Dog", "🐶"),
    ("envelope", "Envelope", "✉️"),
    ("eye", "Eye", "👁️"),
    ("eyeglasses", "Eyeglasses", "👓"),
    ("fish", "Fish", "🐟"),
    ("flower", "Flower", "🌸"),
    ("guitar", "Guitar", "🎸"),
    ("hamburger", "Hamburger", "🍔"),
    ("hat", "Hat", "🎩"),
    ("house", "House", "🏠"),
    ("ice cream", "Ice cream", "🍦"),
    ("key", "Key", "🔑"),
    ("ladder", "Ladder", "🪜"),
    ("lightning", "Lightning", "⚡"),
    ("moon", "Moon", "🌙"),
    ("mountain", "Mountain", "⛰️"),
    ("mushroom", "Mushroom", "🍄"),
    ("pizza", "Pizza", "🍕"),
    ("rainbow", "Rainbow", "🌈"),
    ("scissors", "Scissors", "✂️"),
    ("smiley face", "Smiley face", "😊"),
    ("snake", "Snake", "🐍"),
    ("star", "Star", "⭐"),
    ("sun", "Sun", "☀️"),
    ("sword", "Sword", "🗡️"),
    ("t-shirt", "T-shirt", "👕"),
    ("tree", "Tree", "🌳"),
    ("umbrella", "Umbrella", "☂️"),
]
CLASS_NAMES = [c[0] for c in CATEGORIES]
NUM_CLASSES = len(CATEGORIES)


def file_name(category):
    """A safe file name: 'ice cream' -> 'ice_cream'."""
    return category.replace(" ", "_")
