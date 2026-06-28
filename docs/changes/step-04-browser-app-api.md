# Step 4: Browser Camera App and Prediction API

Date: 2026-06-28

## Scope

This step replaces the local desktop webcam loop with a browser-based app boundary.
It does not train the model, deploy the service, or add user accounts.

## Before

`real_time_emotion_detection.py`:

- Opened the server machine's webcam with `cv2.VideoCapture(0)`.
- Displayed a native OpenCV window with `cv2.imshow`.
- Opened a browser tab from server-side Python with `webbrowser.open`.
- Loaded the Keras model at import time, which prevented the Flask app from starting
  when the model file was missing.
- Initialized Spotify at import time, which made app startup depend on credentials.
- Returned mostly HTML/plain text responses with no prediction API contract.

That shape can work for a local demo, but it is not a production web-app shape. A
deployed server cannot access the user's camera through OpenCV. Camera capture must
happen in the browser with explicit user permission.

## After

`real_time_emotion_detection.py` is now a Flask app with:

- `GET /` for the browser camera interface.
- `GET /api/status` for model/Spotify readiness.
- `POST /api/predict` for one captured browser frame.
- `GET /api/recommendations/<emotion>` for Spotify recommendations.
- `GET /songs/<emotion>` for a server-rendered recommendation page.

The trained model and Spotify client are loaded lazily. The app can start without a
trained model and returns a clear `503` response from `/api/predict` until
`artifacts/emotion_recognition_model.keras` exists.

The browser UI now lives in:

- `templates/index.html`
- `templates/songs.html`
- `static/app.js`
- `static/styles.css`

The UI uses `navigator.mediaDevices.getUserMedia` to request camera access, captures
a still frame to a canvas, and posts that image to the backend. Recommendation cards
are rendered with DOM nodes and `textContent` rather than injecting API data as HTML.

## Why This Changed

Production camera access belongs to the browser. This gives users a native camera
permission prompt and avoids requiring server-side webcam hardware. Separating the
prediction API from the UI also makes the app easier to test, deploy, and later
replace with a mobile or React frontend.

Lazy loading improves startup behavior. Missing model or Spotify configuration now
shows up as an explicit readiness state instead of crashing the app during import.

## Current Limitation

The model has not been trained yet, so live `/api/predict` returns a `503` with a
missing-model message. This is intentional. The UI and API boundary are ready for the
trained model once the training step is approved.

## Verification

Verification was performed without training:

- Flask app imports successfully.
- `GET /` returns the camera page.
- `GET /api/status` returns JSON readiness data.
- `POST /api/predict` without an image returns a `400`.
- `POST /api/predict` with an image returns a clear missing-model `503`.
- `GET /api/recommendations/not-real` returns a `404`.
- Static JavaScript and CSS files are present and referenced by the page.

No server-side webcam, OpenCV GUI window, or automatic browser opening remains.
The older CLI prediction helper now prints the recommended URL instead of opening a
browser tab itself.
