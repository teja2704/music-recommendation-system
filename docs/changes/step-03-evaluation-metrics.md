# Step 3: Evaluation Metrics

Date: 2026-06-28

## Scope

This step adds a real evaluation layer for the emotion model. It does not start
training, tune the model, change the dataset, or redesign the UI.

## Before

`evaluate.py` loaded `X_test.npy`, `y_test.npy`, and the trained model, then printed
only Keras loss and accuracy.

That was not enough for this project because the dataset is imbalanced. For example,
the preprocessed training split has 5,772 `happy` samples but only 349 `disgust`
samples. A model can look acceptable on accuracy while performing poorly on smaller
emotion classes.

There was also no saved evaluation report, no confusion matrix, and no confidence or
calibration signal.

## After

`evaluation_metrics.py` now calculates:

- Accuracy.
- Macro precision.
- Macro recall.
- Macro F1.
- Weighted F1.
- Top-3 accuracy.
- Per-class precision, recall, F1, and support.
- Confusion matrix.
- Mean and median prediction confidence.
- Expected calibration error with 10 confidence bins.

`evaluate.py` now:

1. Loads the trained model and test split.
2. Runs Keras evaluation for loss and accuracy.
3. Runs prediction over the test split.
4. Calculates the richer metrics.
5. Prints a compact summary.
6. Saves:
   - `artifacts/evaluation/metrics.json`
   - `artifacts/evaluation/confusion_matrix.csv`

The default outputs are under `artifacts/`, which is ignored by Git.

## Why This Changed

Production decisions should not rely on accuracy alone. Macro metrics show whether
the model treats each emotion fairly, while weighted metrics show overall behavior
under the dataset distribution. The confusion matrix shows which emotions are being
mixed up. Confidence and calibration metrics help decide whether low-confidence
predictions should be hidden, retried, or shown with caution.

## How To Use After Training

After a model exists at `artifacts/emotion_recognition_model.keras`, run:

```powershell
.\env\Scripts\python.exe evaluate.py
```

Optional custom paths:

```powershell
.\env\Scripts\python.exe evaluate.py `
  --model-path artifacts\emotion_recognition_model.keras `
  --report-path artifacts\evaluation\metrics.json `
  --confusion-matrix-path artifacts\evaluation\confusion_matrix.csv
```

## Verification

This step was verified without training by using synthetic labels and prediction
probabilities. The metric code produced:

- A seven-label confusion matrix.
- A JSON metrics report.
- A CSV confusion matrix.
- Valid top-3 accuracy and expected calibration error values.

The full real-data evaluation intentionally was not run because the model has not
been trained yet.
