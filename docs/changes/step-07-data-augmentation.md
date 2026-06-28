# Step 7: Data Augmentation

Date: 2026-06-28

## Scope

This step adds training-time image augmentation while keeping the same CNN
architecture, preprocessed dataset, callbacks, and evaluation flow from Step 6.

It does not add class weights, balanced sampling, Batch Normalization, deeper CNN
layers, residual connections, or transfer learning.

## Before

The Step 6 callback model was the active model:

- Best validation accuracy: 0.6109
- Test accuracy: 0.6147
- Macro F1: 0.5914
- Weighted F1: 0.6101
- Top-3 accuracy: 0.8792

The Step 6 artifacts were preserved under:

```text
artifacts/baselines/step-06-callbacks/
```

The Step 6 training curves showed that training accuracy continued rising while
validation accuracy plateaued around 61%, which suggests overfitting and makes data
augmentation the right next experiment.

## Augmentation Setup

`train.py` now supports optional training-time augmentation with:

- Horizontal flip.
- Small rotations.
- Slight zoom in/out.
- Mild brightness variation.
- Mild contrast variation.

The augmentation is applied only to training batches through a `tf.data` pipeline.
Validation, evaluation, and inference continue to use the original normalized images.

Default augmentation settings:

- Rotation factor: `0.04`
- Zoom factor: `0.08`
- Brightness factor: `0.10`
- Contrast factor: `0.10`

## After

The augmented training run completed successfully and was preserved under:

```text
artifacts/runs/step-07-augmentation/
```

Run summary:

- Epoch ceiling: 50
- Epochs completed: 50
- Batch size: 64
- Seed: 42
- Best validation accuracy: 0.6050 at epoch 50
- Best validation loss: 1.0615 at epoch 50
- Duration: about 83.2 minutes

The active app model was restored to the Step 6 callback model after comparison
because Step 7 did not improve the primary metrics.

## Evaluation

| Metric | Step 6 Callbacks | Step 7 Augmentation | Delta |
| --- | ---: | ---: | ---: |
| Accuracy | 0.6147 | 0.6041 | -0.0106 |
| Macro F1 | 0.5914 | 0.5481 | -0.0433 |
| Weighted F1 | 0.6101 | 0.5923 | -0.0178 |
| Top-3 Accuracy | 0.8792 | 0.8922 | +0.0130 |
| Expected Calibration Error | 0.0564 | 0.0292 | -0.0272 |

Per-class F1:

| Emotion | Step 6 | Step 7 | Delta |
| --- | ---: | ---: | ---: |
| angry | 0.5139 | 0.5301 | +0.0162 |
| disgust | 0.5682 | 0.3671 | -0.2011 |
| fear | 0.4178 | 0.3312 | -0.0866 |
| happy | 0.8112 | 0.8173 | +0.0062 |
| neutral | 0.5690 | 0.5675 | -0.0015 |
| sad | 0.5020 | 0.4750 | -0.0270 |
| surprise | 0.7575 | 0.7484 | -0.0091 |

The augmented model improved top-3 accuracy and calibration, but the main target
metrics got worse. The largest regressions were `disgust` and `fear`, so this
augmentation recipe is not kept as the active production candidate.

## Why This Changed

Augmentation helps the model see plausible variations of each face without adding new
manual data. It should make the model less sensitive to small changes in pose,
lighting, and framing.

The experiment intentionally changes only the training data presentation so results
can be compared cleanly against Step 6.

## Conclusion

The current augmentation recipe is too broad or too strong for this baseline CNN.
The next data-side experiment should be more targeted, such as class weighting or a
lighter augmentation policy, instead of keeping this full augmentation setup active.
