import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from project_config import (
    ARTIFACTS_PATH,
    DATASET_PATH,
    EMOTION_LABELS,
    IMAGE_SIZE,
    NUM_CLASSES,
)


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_split_directory(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {directory}")

    available_classes = {
        path.name for path in directory.iterdir() if path.is_dir()
    }
    missing_classes = set(EMOTION_LABELS) - available_classes
    if missing_classes:
        missing = ", ".join(sorted(missing_classes))
        raise ValueError(
            f"{directory} is missing required emotion folders: {missing}"
        )


def load_data(directory):
    directory = Path(directory)
    validate_split_directory(directory)

    images = []
    labels = []

    print(f"Loading data from: {directory}")
    for label_index, emotion in enumerate(EMOTION_LABELS):
        emotion_path = directory / emotion
        image_paths = sorted(
            path
            for path in emotion_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )

        loaded_count = 0
        unreadable_count = 0
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                unreadable_count += 1
                continue

            images.append(cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)))
            labels.append(label_index)
            loaded_count += 1

        if loaded_count == 0:
            raise ValueError(f"No readable images found for '{emotion}' in {emotion_path}")

        message = f"Loaded {loaded_count} images for {emotion}"
        if unreadable_count:
            message += f" ({unreadable_count} unreadable files skipped)"
        print(message)

    image_array = np.asarray(images, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=-1) / 255.0
    label_array = np.asarray(labels, dtype=np.int64)

    print(f"Finished loading {len(image_array)} images from {directory}")
    return image_array, label_array


def one_hot_encode(labels):
    return np.eye(NUM_CLASSES, dtype=np.float32)[labels]


def get_class_counts(labels):
    return {
        emotion: int(np.count_nonzero(labels == index))
        for index, emotion in enumerate(EMOTION_LABELS)
    }


def preprocess_dataset(dataset_path, output_path, validation_size=0.2, seed=42):
    dataset_path = Path(dataset_path).resolve()
    output_path = Path(output_path).resolve()

    train_images, train_labels = load_data(dataset_path / "train")
    test_images, test_labels = load_data(dataset_path / "test")

    train_images, validation_images, train_labels, validation_labels = (
        train_test_split(
            train_images,
            train_labels,
            test_size=validation_size,
            random_state=seed,
            stratify=train_labels,
        )
    )

    output_path.mkdir(parents=True, exist_ok=True)
    arrays = {
        "X_train.npy": train_images,
        "y_train.npy": one_hot_encode(train_labels),
        "X_val.npy": validation_images,
        "y_val.npy": one_hot_encode(validation_labels),
        "X_test.npy": test_images,
        "y_test.npy": one_hot_encode(test_labels),
    }
    for filename, array in arrays.items():
        np.save(output_path / filename, array)

    metadata = {
        "dataset_path": str(dataset_path),
        "image_size": IMAGE_SIZE,
        "emotion_labels": list(EMOTION_LABELS),
        "validation_size": validation_size,
        "random_seed": seed,
        "splits": {
            "train": {
                "samples": int(len(train_images)),
                "class_counts": get_class_counts(train_labels),
            },
            "validation": {
                "samples": int(len(validation_images)),
                "class_counts": get_class_counts(validation_labels),
            },
            "test": {
                "samples": int(len(test_images)),
                "class_counts": get_class_counts(test_labels),
            },
        },
    }
    with (output_path / "dataset_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Training data shape: {train_images.shape}")
    print(f"Validation data shape: {validation_images.shape}")
    print(f"Testing data shape: {test_images.shape}")
    print(f"Saved preprocessed artifacts to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess the FER emotion dataset.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DATASET_PATH,
        help=f"Dataset directory (default: {DATASET_PATH})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ARTIFACTS_PATH,
        help=f"Output directory for NumPy arrays (default: {ARTIFACTS_PATH})",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of training images reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.validation_size < 1:
        raise ValueError("--validation-size must be between 0 and 1.")
    preprocess_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        validation_size=args.validation_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
