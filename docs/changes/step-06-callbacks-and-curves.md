# Step 6: Training Callbacks and Curves

Date: 2026-06-28

## Scope

This step improves the training process without changing the CNN architecture or the
preprocessed dataset.

It adds:

- Model checkpointing.
- Early stopping.
- Learning-rate reduction on plateau.
- Training curves for accuracy and loss.

It does not add data augmentation, class weighting, Batch Normalization, residual
connections, or transfer learning.

## Before

The Step 5 baseline existed as:

- `artifacts/emotion_recognition_model.keras`
- `artifacts/training_history.json`
- `artifacts/training_metadata.json`
- `artifacts/evaluation/metrics.json`
- `artifacts/evaluation/confusion_matrix.csv`

Baseline results:

- Best validation accuracy: 0.5961
- Test accuracy: 0.6000
- Macro F1: 0.5685
- Weighted F1: 0.5884
- Top-3 accuracy: 0.8738

Training ran for the full 30 epochs with no checkpoint, early stopping, learning-rate
schedule, or plot.

## Preservation

Before retraining, Step 5 artifacts were copied to:

```text
artifacts/baselines/step-05-baseline/
```

This keeps the original baseline available for comparison even though the active
model path is reused by the app.

## After

Callback training completed successfully.

Created/updated active artifacts:

- `artifacts/emotion_recognition_model.keras`
- `artifacts/checkpoints/best_emotion_model.keras`
- `artifacts/training_history.json`
- `artifacts/training_metadata.json`
- `artifacts/training_curves.png`
- `artifacts/evaluation/metrics.json`
- `artifacts/evaluation/confusion_matrix.csv`

The Step 6 run was also copied to:

```text
artifacts/runs/step-06-callbacks/
```

Run summary:

- Epoch ceiling: 50
- Epochs completed: 50
- Batch size: 64
- Seed: 42
- Duration: about 83.8 minutes
- Best validation accuracy: 0.6109 at epoch 44
- Best validation loss: 1.1070 at epoch 30
- Final active model: best validation-accuracy checkpoint
- Training curves: `artifacts/training_curves.png`

`ReduceLROnPlateau` activated several times. `EarlyStopping` did not stop early
because validation accuracy still improved late in the run, most notably at epochs
40, 41, and 44.

## Evaluation

The callback-trained model was evaluated on the same held-out test split.

| Metric | Step 5 Baseline | Step 6 Callbacks | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.6000 | 0.6147 | +0.0146 |
| Macro F1 | 0.5685 | 0.5914 | +0.0228 |
| Weighted F1 | 0.5884 | 0.6101 | +0.0217 |
| Top-3 Accuracy | 0.8738 | 0.8792 | +0.0054 |

Per-class F1:

| Emotion | Step 5 | Step 6 | Delta |
| --- | ---: | ---: | ---: |
| angry | 0.4951 | 0.5139 | +0.0187 |
| disgust | 0.5325 | 0.5682 | +0.0356 |
| fear | 0.3820 | 0.4178 | +0.0358 |
| happy | 0.7845 | 0.8112 | +0.0266 |
| neutral | 0.5418 | 0.5690 | +0.0272 |
| sad | 0.4945 | 0.5020 | +0.0076 |
| surprise | 0.7493 | 0.7575 | +0.0082 |

All seven classes improved on F1. `fear` remains the weakest class, but it improved
from 0.3820 to 0.4178.

## Why This Changed

Callbacks make training safer and easier to compare:

- `ModelCheckpoint` keeps the best validation-accuracy model.
- `EarlyStopping` avoids continuing once validation accuracy stops improving.
- `ReduceLROnPlateau` lowers the learning rate when validation loss stalls.
- Training curves make overfitting and learning progress visible in one artifact.

The same architecture and data are used so the comparison isolates the training
process rather than mixing in multiple model changes.
