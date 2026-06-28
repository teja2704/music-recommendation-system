# Step 5: Baseline Model Training

Date: 2026-06-28

## Scope

This step trains the corrected seven-class baseline model using the architecture
already defined in `model.py`.

It does not tune the model architecture, add class weighting, add augmentation, or
deploy the app.

## Before

The corrected preprocessing artifacts existed under `artifacts/`:

- `X_train.npy`
- `y_train.npy`
- `X_val.npy`
- `y_val.npy`
- `X_test.npy`
- `y_test.npy`
- `dataset_metadata.json`

The trained model did not exist at:

```text
artifacts/emotion_recognition_model.keras
```

Because the model was missing, the browser app correctly returned a missing-model
`503` from `/api/predict`.

`train.py` could train and save a model, but it did not save the training history or
run metadata needed to understand the baseline later.

## Training Setup

The training script now records:

- `artifacts/training_history.json`
- `artifacts/training_metadata.json`

The CNN architecture was not changed. The baseline uses:

- 30 requested epochs.
- Batch size 64.
- Seed 42.
- The corrected seven-class train and validation arrays.

## After

Training completed successfully.

Created:

- `artifacts/emotion_recognition_model.keras`
- `artifacts/training_history.json`
- `artifacts/training_metadata.json`

Run summary:

- Epochs completed: 30
- Batch size: 64
- Seed: 42
- Duration: about 44.9 minutes
- Final training accuracy: 0.6956
- Final validation accuracy: 0.5961
- Best validation accuracy: 0.5961
- Best validation loss: 1.1102

The browser app now reports `model_ready=true` from `/api/status`.

## Evaluation

Evaluation completed on the held-out test split with the Step 3 metrics report.

Created:

- `artifacts/evaluation/metrics.json`
- `artifacts/evaluation/confusion_matrix.csv`

Overall test metrics:

- Test loss: 1.1121
- Test accuracy: 0.6000
- Macro precision: 0.6121
- Macro recall: 0.5544
- Macro F1: 0.5685
- Weighted F1: 0.5884
- Top-3 accuracy: 0.8738
- Mean confidence: 0.6212
- Expected calibration error: 0.0279

Per-class F1:

- `angry`: 0.4951
- `disgust`: 0.5325
- `fear`: 0.3820
- `happy`: 0.7845
- `neutral`: 0.5418
- `sad`: 0.4945
- `surprise`: 0.7493

The weakest class is currently `fear`, mostly confused with `sad`, `neutral`, and
`surprise`. This gives the next improvement step a concrete target.

## Why This Changed

A baseline model creates the reference point for every later improvement. Saving
history and metadata makes it possible to compare future model changes against this
run instead of relying on memory or terminal output.

## Next Improvement Targets

The baseline is useful but not production-ready yet. The next model-focused work
should investigate:

- Class imbalance, especially the small `disgust` class.
- `fear` confusion with nearby emotions.
- Data augmentation.
- Class weights or balanced sampling.
- Better callbacks and checkpointing.
- A stronger architecture or transfer-learning experiment.
