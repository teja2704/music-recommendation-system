import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

IMAGE_SIZE = 48
EMOTION_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)
NUM_CLASSES = len(EMOTION_LABELS)


def _configured_path(variable_name, default):
    configured_value = os.getenv(variable_name)
    path = Path(configured_value).expanduser() if configured_value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DATASET_PATH = _configured_path("DATASET_PATH", PROJECT_ROOT / "dataset")
ARTIFACTS_PATH = _configured_path("ARTIFACTS_PATH", PROJECT_ROOT / "artifacts")
MODEL_PATH = ARTIFACTS_PATH / "emotion_recognition_model.keras"
