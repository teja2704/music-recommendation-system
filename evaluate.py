import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from project_config import ARTIFACTS_PATH, MODEL_PATH


def evaluate_model(artifacts_path, model_path):
    artifacts_path = Path(artifacts_path).resolve()
    model_path = Path(model_path).resolve()

    test_images_path = artifacts_path / "X_test.npy"
    test_labels_path = artifacts_path / "y_test.npy"
    for path in (test_images_path, test_labels_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation artifact not found: {path}")

    test_images = np.load(test_images_path)
    test_labels = np.load(test_labels_path)
    model = tf.keras.models.load_model(model_path)
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained emotion model.")
    parser.add_argument("--artifacts-path", type=Path, default=ARTIFACTS_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_model(args.artifacts_path, args.model_path)


if __name__ == "__main__":
    main()
