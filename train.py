import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import create_model
from project_config import ARTIFACTS_PATH, MODEL_PATH


def load_array(artifacts_path, filename):
    path = artifacts_path / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing preprocessing artifact: {path}. Run preprocess.py first."
        )
    return np.load(path)


def _json_safe_history(history):
    return {
        metric: [float(value) for value in values]
        for metric, values in history.history.items()
    }


def save_json(payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def plot_training_curves(history_payload, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history_payload.get("loss", [])) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history_payload.get("accuracy", []), label="Train Accuracy")
    axes[0].plot(
        epochs,
        history_payload.get("val_accuracy", []),
        label="Validation Accuracy",
    )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history_payload.get("loss", []), label="Train Loss")
    axes[1].plot(epochs, history_payload.get("val_loss", []), label="Validation Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_callbacks(
    checkpoint_path,
    use_callbacks=True,
    early_stopping_patience=8,
    reduce_lr_patience=3,
    min_lr=1e-6,
):
    if not use_callbacks:
        return []

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
    ]


def build_augmentation_layers(
    rotation_factor=0.04,
    zoom_factor=0.08,
    brightness_factor=0.10,
    contrast_factor=0.10,
    seed=42,
):
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(
                rotation_factor,
                fill_mode="nearest",
                seed=seed + 1,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-zoom_factor, zoom_factor),
                width_factor=(-zoom_factor, zoom_factor),
                fill_mode="nearest",
                seed=seed + 2,
            ),
            tf.keras.layers.RandomBrightness(
                brightness_factor,
                value_range=(0.0, 1.0),
                seed=seed + 3,
            ),
            tf.keras.layers.RandomContrast(contrast_factor, seed=seed + 4),
        ],
        name="training_augmentation",
    )


def build_training_dataset(
    train_images,
    train_labels,
    batch_size,
    seed=42,
    use_augmentation=True,
    rotation_factor=0.04,
    zoom_factor=0.08,
    brightness_factor=0.10,
    contrast_factor=0.10,
):
    dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    dataset = dataset.shuffle(
        buffer_size=len(train_images),
        seed=seed,
        reshuffle_each_iteration=True,
    )
    dataset = dataset.batch(batch_size)

    if use_augmentation:
        augmentation_layers = build_augmentation_layers(
            rotation_factor=rotation_factor,
            zoom_factor=zoom_factor,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            seed=seed,
        )

        def augment_batch(images, labels):
            augmented_images = augmentation_layers(images, training=True)
            augmented_images = tf.clip_by_value(augmented_images, 0.0, 1.0)
            return augmented_images, labels

        dataset = dataset.map(
            augment_batch,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.prefetch(tf.data.AUTOTUNE)


def train_model(
    artifacts_path,
    model_path,
    history_path,
    metadata_path,
    curves_path,
    checkpoint_path,
    epochs=30,
    batch_size=64,
    seed=42,
    use_callbacks=True,
    early_stopping_patience=8,
    reduce_lr_patience=3,
    min_lr=1e-6,
    use_augmentation=False,
    rotation_factor=0.04,
    zoom_factor=0.08,
    brightness_factor=0.10,
    contrast_factor=0.10,
):
    artifacts_path = Path(artifacts_path).resolve()
    model_path = Path(model_path).resolve()
    history_path = Path(history_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    curves_path = Path(curves_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    train_images = load_array(artifacts_path, "X_train.npy")
    train_labels = load_array(artifacts_path, "y_train.npy")
    validation_images = load_array(artifacts_path, "X_val.npy")
    validation_labels = load_array(artifacts_path, "y_val.npy")

    tf.keras.utils.set_random_seed(seed)
    model = create_model()
    training_data = build_training_dataset(
        train_images,
        train_labels,
        batch_size=batch_size,
        seed=seed,
        use_augmentation=use_augmentation,
        rotation_factor=rotation_factor,
        zoom_factor=zoom_factor,
        brightness_factor=brightness_factor,
        contrast_factor=contrast_factor,
    )
    callbacks = build_callbacks(
        checkpoint_path=checkpoint_path,
        use_callbacks=use_callbacks,
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
        min_lr=min_lr,
    )
    start_time = time.time()
    history = model.fit(
        training_data,
        epochs=epochs,
        validation_data=(validation_images, validation_labels),
        callbacks=callbacks,
        verbose=2,
    )
    duration_seconds = time.time() - start_time

    history_payload = _json_safe_history(history)
    save_json(history_payload, history_path)
    plot_training_curves(history_payload, curves_path)

    best_checkpoint_used = False
    if use_callbacks and checkpoint_path.is_file():
        model = tf.keras.models.load_model(checkpoint_path)
        best_checkpoint_used = True

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    best_validation_accuracy = max(history_payload.get("val_accuracy", [0.0]))
    best_validation_loss = min(history_payload.get("val_loss", [0.0]))
    best_validation_accuracy_epoch = (
        history_payload["val_accuracy"].index(best_validation_accuracy) + 1
    )
    best_validation_loss_epoch = history_payload["val_loss"].index(best_validation_loss) + 1
    metadata = {
        "model_path": str(model_path),
        "artifacts_path": str(artifacts_path),
        "history_path": str(history_path),
        "curves_path": str(curves_path),
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_used_for_final_model": best_checkpoint_used,
        "final_model_source": "best_validation_accuracy_checkpoint"
        if best_checkpoint_used
        else "last_epoch",
        "final_model_source_epoch": best_validation_accuracy_epoch
        if best_checkpoint_used
        else len(history_payload.get("loss", [])),
        "epochs_requested": epochs,
        "epochs_completed": len(history_payload.get("loss", [])),
        "batch_size": batch_size,
        "seed": seed,
        "callbacks_enabled": use_callbacks,
        "early_stopping_patience": early_stopping_patience if use_callbacks else None,
        "reduce_lr_patience": reduce_lr_patience if use_callbacks else None,
        "minimum_learning_rate": min_lr if use_callbacks else None,
        "augmentation_enabled": use_augmentation,
        "augmentation": {
            "horizontal_flip": use_augmentation,
            "rotation_factor": rotation_factor if use_augmentation else None,
            "zoom_factor": zoom_factor if use_augmentation else None,
            "brightness_factor": brightness_factor if use_augmentation else None,
            "contrast_factor": contrast_factor if use_augmentation else None,
        },
        "duration_seconds": float(duration_seconds),
        "final_training_accuracy": history_payload["accuracy"][-1],
        "final_training_loss": history_payload["loss"][-1],
        "final_validation_accuracy": history_payload["val_accuracy"][-1],
        "final_validation_loss": history_payload["val_loss"][-1],
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_accuracy_epoch": best_validation_accuracy_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_loss_epoch": best_validation_loss_epoch,
        "train_shape": list(train_images.shape),
        "validation_shape": list(validation_images.shape),
    }
    dataset_metadata_path = artifacts_path / "dataset_metadata.json"
    if dataset_metadata_path.is_file():
        metadata["dataset_metadata_path"] = str(dataset_metadata_path)
        metadata["dataset_metadata"] = json.loads(
            dataset_metadata_path.read_text(encoding="utf-8")
        )
    save_json(metadata, metadata_path)

    print(f"Model training complete. Saved model to: {model_path}")
    print(f"Saved training history to: {history_path}")
    print(f"Saved training metadata to: {metadata_path}")
    print(f"Saved training curves to: {curves_path}")
    if best_checkpoint_used:
        print(f"Final model was restored from best checkpoint: {checkpoint_path}")
    print(f"Best validation accuracy: {best_validation_accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the emotion recognition model.")
    parser.add_argument("--artifacts-path", type=Path, default=ARTIFACTS_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--history-path",
        type=Path,
        default=ARTIFACTS_PATH / "training_history.json",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=ARTIFACTS_PATH / "training_metadata.json",
    )
    parser.add_argument(
        "--curves-path",
        type=Path,
        default=ARTIFACTS_PATH / "training_curves.png",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=ARTIFACTS_PATH / "checkpoints" / "best_emotion_model.keras",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-callbacks",
        action="store_true",
        help="Disable checkpointing, early stopping, and learning-rate reduction.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--reduce-lr-patience", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        help="Apply training-time image augmentation to batches.",
    )
    parser.add_argument("--rotation-factor", type=float, default=0.04)
    parser.add_argument("--zoom-factor", type=float, default=0.08)
    parser.add_argument("--brightness-factor", type=float, default=0.10)
    parser.add_argument("--contrast-factor", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(
        artifacts_path=args.artifacts_path,
        model_path=args.model_path,
        history_path=args.history_path,
        metadata_path=args.metadata_path,
        curves_path=args.curves_path,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        use_callbacks=not args.no_callbacks,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        min_lr=args.min_lr,
        use_augmentation=args.use_augmentation,
        rotation_factor=args.rotation_factor,
        zoom_factor=args.zoom_factor,
        brightness_factor=args.brightness_factor,
        contrast_factor=args.contrast_factor,
    )


if __name__ == "__main__":
    main()
