import base64
import binascii
import random

import cv2
import numpy as np
import spotipy
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from spotipy.oauth2 import SpotifyClientCredentials

from project_config import EMOTION_LABELS, IMAGE_SIZE, MODEL_PATH
from spotify_config import get_spotify_client_credentials


EMOTION_GENRE_MAPPING = {
    "angry": "rock",
    "disgust": "alternative",
    "fear": "dark",
    "happy": "pop",
    "neutral": "chill",
    "sad": "acoustic",
    "surprise": "electronic",
}

app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(
    f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml"
)
emotion_model = None
emotion_model_error = None
spotify_client = None
spotify_client_error = None


def get_emotion_model():
    global emotion_model, emotion_model_error

    if emotion_model is not None or emotion_model_error is not None:
        return emotion_model, emotion_model_error

    if not MODEL_PATH.is_file():
        emotion_model_error = (
            f"Trained model not found at {MODEL_PATH}. Run train.py before "
            "using live emotion prediction."
        )
        return None, emotion_model_error

    try:
        emotion_model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as exc:
        emotion_model_error = f"Unable to load trained model: {exc}"

    return emotion_model, emotion_model_error


def get_spotify_client():
    global spotify_client, spotify_client_error

    if spotify_client is not None or spotify_client_error is not None:
        return spotify_client, spotify_client_error

    try:
        client_id, client_secret = get_spotify_client_credentials()
        spotify_client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret,
            )
        )
    except Exception as exc:
        spotify_client_error = f"Unable to configure Spotify client: {exc}"

    return spotify_client, spotify_client_error


def decode_image_data(image_data):
    if not image_data:
        raise ValueError("Missing image data.")

    encoded_data = image_data.split(",", 1)[-1]
    try:
        image_bytes = base64.b64decode(encoded_data, validate=True)
    except binascii.Error as exc:
        raise ValueError("Image data is not valid base64.") from exc

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Image data could not be decoded.")
    return frame


def get_largest_face(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: face[2] * face[3])


def preprocess_face(frame, face_box):
    x, y, width, height = face_box
    face = frame[y:y + height, x:x + width]
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (IMAGE_SIZE, IMAGE_SIZE))
    normalized_face = resized_face.astype(np.float32) / 255.0
    return normalized_face.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)


def predict_emotion_from_frame(frame):
    model, error = get_emotion_model()
    if error:
        return None, error

    face_box = get_largest_face(frame)
    if face_box is None:
        return None, "No face was detected in the captured image."

    prediction = model.predict(preprocess_face(frame, face_box), verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    probabilities = {
        emotion: float(probability)
        for emotion, probability in zip(EMOTION_LABELS, prediction)
    }
    x, y, width, height = [int(value) for value in face_box]

    return {
        "emotion": EMOTION_LABELS[predicted_index],
        "confidence": float(prediction[predicted_index]),
        "probabilities": probabilities,
        "face_box": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
    }, None


def get_recommended_songs(emotion, limit=5):
    if emotion not in EMOTION_GENRE_MAPPING:
        raise ValueError(f"Unsupported emotion: {emotion}")

    client, error = get_spotify_client()
    if error:
        raise RuntimeError(error)

    genre = EMOTION_GENRE_MAPPING[emotion]
    random_offset = random.randint(0, 100)
    results = client.search(
        q=f"genre:{genre}",
        type="track",
        limit=limit,
        offset=random_offset,
    )

    songs = []
    for track in results.get("tracks", {}).get("items", []):
        album_images = track.get("album", {}).get("images", [])
        songs.append(
            {
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "url": track["external_urls"]["spotify"],
                "album_name": track["album"]["name"],
                "release_date": track["album"]["release_date"],
                "album_art": album_images[0]["url"] if album_images else "",
            }
        )

    return songs


@app.route("/")
def index():
    return render_template(
        "index.html",
        emotion_labels=EMOTION_LABELS,
        model_ready=MODEL_PATH.is_file(),
    )


@app.route("/api/status")
def status():
    model, model_error = get_emotion_model()
    client, spotify_error = get_spotify_client()
    return jsonify(
        {
            "emotion_labels": list(EMOTION_LABELS),
            "model_ready": model is not None,
            "model_error": model_error,
            "spotify_ready": client is not None,
            "spotify_error": spotify_error,
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        frame = decode_image_data(payload.get("image"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    result, error = predict_emotion_from_frame(frame)
    if error and "Trained model not found" in error:
        return jsonify({"error": error}), 503
    if error:
        return jsonify({"error": error}), 422

    return jsonify(result)


@app.route("/api/recommendations/<emotion>")
def api_recommendations(emotion):
    try:
        songs = get_recommended_songs(emotion)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify({"emotion": emotion, "songs": songs})


@app.route("/songs/<emotion>")
def display_songs(emotion):
    try:
        songs = get_recommended_songs(emotion)
        error = None
    except Exception as exc:
        songs = []
        error = str(exc)
    return render_template(
        "songs.html",
        emotion=emotion,
        songs=songs,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
