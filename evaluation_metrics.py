import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


def labels_from_one_hot(one_hot_labels):
    return np.asarray(one_hot_labels).argmax(axis=1)


def get_prediction_labels(prediction_probabilities):
    return np.asarray(prediction_probabilities).argmax(axis=1)


def get_prediction_confidence(prediction_probabilities):
    return np.asarray(prediction_probabilities).max(axis=1)


def calculate_expected_calibration_error(
    true_labels,
    predicted_labels,
    prediction_confidence,
    bin_count=10,
):
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    prediction_confidence = np.asarray(prediction_confidence)

    bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
    expected_calibration_error = 0.0
    calibration_bins = []

    for index in range(bin_count):
        lower_bound = bin_edges[index]
        upper_bound = bin_edges[index + 1]
        if index == 0:
            in_bin = (
                (prediction_confidence >= lower_bound)
                & (prediction_confidence <= upper_bound)
            )
        else:
            in_bin = (
                (prediction_confidence > lower_bound)
                & (prediction_confidence <= upper_bound)
            )

        sample_count = int(np.count_nonzero(in_bin))
        if sample_count == 0:
            calibration_bins.append(
                {
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "sample_count": 0,
                    "accuracy": None,
                    "average_confidence": None,
                    "gap": None,
                }
            )
            continue

        bin_accuracy = float(np.mean(true_labels[in_bin] == predicted_labels[in_bin]))
        bin_confidence = float(np.mean(prediction_confidence[in_bin]))
        gap = abs(bin_accuracy - bin_confidence)
        expected_calibration_error += (sample_count / len(true_labels)) * gap
        calibration_bins.append(
            {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "sample_count": sample_count,
                "accuracy": bin_accuracy,
                "average_confidence": bin_confidence,
                "gap": float(gap),
            }
        )

    return float(expected_calibration_error), calibration_bins


def calculate_metrics(
    true_one_hot_labels,
    prediction_probabilities,
    emotion_labels,
    calibration_bin_count=10,
):
    true_labels = labels_from_one_hot(true_one_hot_labels)
    predicted_labels = get_prediction_labels(prediction_probabilities)
    prediction_confidence = get_prediction_confidence(prediction_probabilities)
    label_indices = list(range(len(emotion_labels)))

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=label_indices,
        target_names=emotion_labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=label_indices,
    )
    expected_calibration_error, calibration_bins = (
        calculate_expected_calibration_error(
            true_labels,
            predicted_labels,
            prediction_confidence,
            bin_count=calibration_bin_count,
        )
    )

    return {
        "sample_count": int(len(true_labels)),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                true_labels,
                predicted_labels,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=label_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=label_indices,
                average="weighted",
                zero_division=0,
            )
        ),
        "top_3_accuracy": float(
            top_k_accuracy_score(
                true_labels,
                prediction_probabilities,
                k=min(3, len(emotion_labels)),
                labels=label_indices,
            )
        ),
        "mean_confidence": float(np.mean(prediction_confidence)),
        "median_confidence": float(np.median(prediction_confidence)),
        "expected_calibration_error": expected_calibration_error,
        "classification_report": report,
        "confusion_matrix": {
            "labels": list(emotion_labels),
            "matrix": matrix.astype(int).tolist(),
        },
        "calibration_bins": calibration_bins,
    }


def save_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_confusion_matrix_csv(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    with output_path.open("w", encoding="utf-8") as file:
        file.write("," + ",".join(labels) + "\n")
        for label, row in zip(labels, matrix):
            file.write(label + "," + ",".join(str(value) for value in row) + "\n")


def print_metric_summary(metrics):
    print(f"Samples: {metrics['sample_count']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
    print(
        "Expected Calibration Error: "
        f"{metrics['expected_calibration_error']:.4f}"
    )
