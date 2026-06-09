import argparse
import random
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from project_config import EMOTION_LABELS, IMAGE_SIZE, MODEL_PATH


MUSIC_RECOMMENDATIONS = {
    "angry": [
        "https://www.youtube.com/watch?v=2vjPBrBU-TM",
        "https://www.youtube.com/watch?v=YykjpeuMNEk",
    ],
    "disgust": ["https://www.youtube.com/watch?v=o3mP3mJDL2k"],
    "fear": ["https://www.youtube.com/watch?v=UfcAVejslrU"],
    "happy": [
        "https://www.youtube.com/watch?v=09R8_2nJtjg",
        "https://www.youtube.com/watch?v=kJQP7kiw5Fk",
    ],
    "neutral": ["https://www.youtube.com/watch?v=3HjG1Y4QpVA"],
    "sad": ["https://www.youtube.com/watch?v=J_ub7Etch2U"],
    "surprise": ["https://www.youtube.com/watch?v=GRz4FY0ZcwI"],
}


def predict_emotion(model, image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = image.astype(np.float32) / 255.0
    image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    prediction = model.predict(image, verbose=0)
    return EMOTION_LABELS[int(np.argmax(prediction))]


def recommend_music(emotion, open_browser=False):
    song_link = random.choice(MUSIC_RECOMMENDATIONS[emotion])
    print(f"Recommended song for {emotion}: {song_link}")
    if open_browser:
        webbrowser.open(song_link)
    return song_link


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict emotion from an image and recommend music."
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the recommendation in the default browser.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {args.model_path}. Run train.py first."
        )

    model = tf.keras.models.load_model(args.model_path)
    emotion = predict_emotion(model, args.image_path)
    print(f"The predicted emotion is: {emotion}")
    recommend_music(emotion, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
