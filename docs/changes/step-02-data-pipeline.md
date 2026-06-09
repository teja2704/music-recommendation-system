# Step 2: Data Pipeline and Portable Paths

Date: 2026-06-10

## Scope

This step repairs the emotion-class mapping and makes preprocessing, training,
evaluation, and single-image prediction portable and reproducible.

It does not tune the model architecture, add advanced metrics, redesign the browser
application, or deploy the project.

## Before

### Emotion labels did not match the dataset

The code expected:

- `angry`
- `disgusted`
- `fearful`
- `happy`
- `neutral`
- `sad`
- `surprised`

The actual FER dataset folders are:

- `angry`
- `disgust`
- `fear`
- `happy`
- `neutral`
- `sad`
- `surprise`

The preprocessor silently skipped missing folders. As a result, `disgust`, `fear`,
and `surprise` were excluded from the previous preprocessing run even though the
output labels were still created with seven columns.

### Paths were tied to one computer

`preprocess.py` pointed to an old absolute OneDrive path and `predict.py` pointed to
a personal Downloads image. Moving the repository made those paths invalid.

Training and evaluation also assumed that generated arrays and the model were in the
current terminal directory.

### Pipeline behavior was difficult to reproduce

- Importing preprocessing or training modules immediately executed work.
- Missing emotion folders were logged and ignored instead of failing.
- File iteration order was not deterministic.
- The train/validation split was not stratified.
- Generated artifacts had no metadata describing labels, seed, or class counts.
- The local virtual environment still referenced the repository's previous location
  and a Python executable that is no longer available.

## After

### One shared project configuration

`project_config.py` now defines:

- Project, dataset, artifact, and model paths.
- The image size.
- The canonical seven emotion labels.
- Optional `DATASET_PATH` and `ARTIFACTS_PATH` environment overrides.

All model and prediction code imports the same label order, preventing training and
inference from drifting apart.

### Strict and deterministic preprocessing

`preprocess.py` now:

- Requires every expected emotion folder.
- Loads files in sorted order.
- Reports unreadable files.
- Uses `float32` normalized image arrays.
- Uses a seeded, stratified train/validation split.
- Saves all arrays under `artifacts/` by default.
- Writes `dataset_metadata.json` with labels, counts, image size, split ratio, and
  random seed.
- Runs only when invoked directly.

### Portable command-line workflow

The intended sequence is:

```powershell
python preprocess.py
python train.py
python evaluate.py
python predict.py path\to\image.jpg
```

Paths, epochs, batch size, and prediction browser behavior can be changed with
command-line options. Use `--help` on each script for details.

The model format is now Keras' native `.keras` format under:

```text
artifacts/emotion_recognition_model.keras
```

### Environment reproducibility

`requirements.txt` records the core runtime dependencies, including
`python-dotenv`, which loads the ignored local `.env` file.

The existing `env/` directory was damaged after the repository move:

- NumPy was missing its compiled extension.
- OpenCV was missing or unable to load its compiled extension.
- TensorFlow 2.18 was incomplete and missing `_pywrap_tf2`.
- `python-dotenv` was not installed.

NumPy 1.26.4 and OpenCV 4.11.0.86 were reinstalled, `python-dotenv` was installed,
and the broken local TensorFlow 2.18 package was isolated inside the ignored
environment. The environment now uses the verified system TensorFlow 2.16.1, which
is also the version recorded in `requirements.txt`. Its compatible `ml-dtypes`,
protobuf, and TensorBoard versions are pinned as well.

## Why This Changed

Class names and their order form a contract between preprocessing, model training,
evaluation, and inference. A silent mismatch produces a model whose outputs cannot be
trusted. Central configuration and strict validation make that contract explicit.

Project-relative paths and command-line overrides allow the same code to run on a
developer computer, CI runner, or production service without source edits.

Stratification preserves each emotion's representation in validation data, while
metadata makes future results traceable.

## Generated Artifacts

The corrected preprocessor completed successfully across all 35,887 images and
created:

- 22,967 training samples.
- 5,742 validation samples.
- 7,178 test samples.
- Seven-column one-hot labels for every split.
- `dataset_metadata.json` with per-class counts.

All generated files are under the ignored `artifacts/` directory. The obsolete
root-level `y_val.npy` and `y_test.npy` files from the defective run were removed.

## Verification

The following checks passed:

- Python syntax compilation for all application modules.
- Imports for OpenCV, NumPy, scikit-learn, TensorFlow, Spotipy, Flask, and dotenv.
- A clean `pip check` with no broken requirements.
- `.env` loading without displaying credential values.
- Presence of all seven dataset classes in train and test splits.
- Full preprocessing over 28,709 source training images and 7,178 test images.
- Artifact shapes, dtypes, label order, and stratified class counts.
- Model construction with input shape `(None, 48, 48, 1)` and output shape
  `(None, 7)`.

The model has not been trained in this step. Training is a long-running,
compute-intensive action and should be started as the next deliberate decision.
