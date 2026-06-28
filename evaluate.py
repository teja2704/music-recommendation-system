import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from evaluation_metrics import (
    calculate_metrics,
    print_metric_summary,
    save_confusion_matrix_csv,
    save_metrics,
)
from project_config import ARTIFACTS_PATH, EMOTION_LABELS, MODEL_PATH


def evaluate_model(artifacts_path, model_path, report_path, confusion_matrix_path):
    artifacts_path = Path(artifacts_path).resolve()
    model_path = Path(model_path).resolve()
    report_path = Path(report_path).resolve()
    confusion_matrix_path = Path(confusion_matrix_path).resolve()

    test_images_path = artifacts_path / "X_test.npy"
    test_labels_path = artifacts_path / "y_test.npy"
    for path in (test_images_path, test_labels_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation artifact not found: {path}")

    test_images = np.load(test_images_path)
    test_labels = np.load(test_labels_path)
    model = tf.keras.models.load_model(model_path)
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    prediction_probabilities = model.predict(test_images, verbose=0)
    metrics = calculate_metrics(
        test_labels,
        prediction_probabilities,
        EMOTION_LABELS,
    )
    metrics["keras_loss"] = float(test_loss)
    metrics["keras_accuracy"] = float(test_accuracy)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    print_metric_summary(metrics)

    save_metrics(metrics, report_path)
    save_confusion_matrix_csv(metrics, confusion_matrix_path)
    print(f"Saved evaluation report to: {report_path}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained emotion model.")
    parser.add_argument("--artifacts-path", type=Path, default=ARTIFACTS_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ARTIFACTS_PATH / "evaluation" / "metrics.json",
    )
    parser.add_argument(
        "--confusion-matrix-path",
        type=Path,
        default=ARTIFACTS_PATH / "evaluation" / "confusion_matrix.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_model(
        args.artifacts_path,
        args.model_path,
        args.report_path,
        args.confusion_matrix_path,
    )


if __name__ == "__main__":
    main()
