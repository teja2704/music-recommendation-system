import argparse
from pathlib import Path

import numpy as np

from model import create_model
from project_config import ARTIFACTS_PATH, MODEL_PATH


def load_array(artifacts_path, filename):
    path = artifacts_path / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing preprocessing artifact: {path}. Run preprocess.py first."
        )
    return np.load(path)


def train_model(artifacts_path, model_path, epochs=30, batch_size=64):
    artifacts_path = Path(artifacts_path).resolve()
    model_path = Path(model_path).resolve()

    train_images = load_array(artifacts_path, "X_train.npy")
    train_labels = load_array(artifacts_path, "y_train.npy")
    validation_images = load_array(artifacts_path, "X_val.npy")
    validation_labels = load_array(artifacts_path, "y_val.npy")

    model = create_model()
    model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation_images, validation_labels),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model training complete. Saved model to: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the emotion recognition model.")
    parser.add_argument("--artifacts-path", type=Path, default=ARTIFACTS_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(
        artifacts_path=args.artifacts_path,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
