import random
import cv2
import numpy as np
import tensorflow as tf
import spotipy
import webbrowser
import time
from spotipy.oauth2 import SpotifyClientCredentials
from flask import Flask, render_template
from threading import Thread

# Load the trained emotion detection model
model = tf.keras.models.load_model("emotion_recognition_model.h5")

# Define the list of emotions
EMOTION_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Map emotions to Spotify genres
EMOTION_GENRE_MAPPING = {
    "happy": "pop",
    "sad": "acoustic",
    "angry": "rock",
    "fearful": "dark",
    "neutral": "chill",
    "surprised": "electronic",
    "disgusted": "alternative"
}

# Spotify API credentials
SPOTIFY_CLIENT_ID = "a20569cb114a4cd380f6c240f0ad744c"
SPOTIFY_CLIENT_SECRET = "76b780a37b444824a6372dfec6ec41f3"

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
))

# Function to detect emotion from an image
def predict_emotion(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (48, 48))  # Resize to match the model's expected input
    img = img / 255.0  # Normalize the image
    img = img.reshape(1, 48, 48, 1)  # Add batch dimension
    prediction = model.predict(img)  # Get the model's prediction
    predicted_class = np.argmax(prediction)  # Get the emotion label
    return EMOTION_LABELS[predicted_class]

# Function to get recommended songs from Spotify
def get_recommended_songs(emotion):
    genre = EMOTION_GENRE_MAPPING.get(emotion, "pop")  # Default to pop
    try:
        return _extracted_from_get_recommended_songs_5(genre)
    except Exception as e:
        print(f"Error while fetching songs from Spotify: {e}")
        return None


# TODO Rename this here and in `get_recommended_songs`
def _extracted_from_get_recommended_songs_5(genre):
    # Add a random offset to fetch different songs each time
    random_offset = random.randint(0, 100)

    # Fetch new recommendations from Spotify
    results = sp.search(q=f"genre:{genre}", type="track", limit=5, offset=random_offset)
    if not results['tracks']['items']:
        return None  # Return None if no songs found

    songs = []
    for track in results["tracks"]["items"]:
        song_details = {
            "name": track["name"],
            "artist": track["artists"][0]["name"],
            "url": track["external_urls"]["spotify"],
            "album_name": track["album"]["name"],
            "release_date": track["album"]["release_date"],
            "album_art": track["album"]["images"][0]["url"]
        }
        songs.append(song_details)

    return songs

# Initialize Flask app for web display
app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml"
)

# Variable to store the first detected emotion
first_emotion_detected = None

@app.route('/')
def index():
    return "Emotion Detection - Please keep the webcam on for 5 seconds."

@app.route('/songs/<emotion>')
def display_songs(emotion):
    # Get recommended songs for the detected emotion
    songs = get_recommended_songs(emotion)
    if songs is None:
        return "Error: No songs found or failed to fetch data."
    return render_template('songs.html', emotion=emotion, songs=songs)

@app.route('/refresh_songs/<emotion>')
def refresh_songs(emotion):
    # Fetch completely new songs by making a new request to Spotify
    new_songs = get_recommended_songs(emotion)
    if new_songs is None:
        return "Error: Unable to refresh songs."
    
    return render_template('songs.html', emotion=emotion, songs=new_songs)

@app.route('/refresh')
def refresh():
    global first_emotion_detected
    first_emotion_detected = None
    return "Emotion detection restarted. Please keep the webcam on for 5 seconds."

# Function to process webcam feed and detect emotion
def process_webcam():
    global first_emotion_detected

    # Wait for 5 seconds before detecting the first emotion
    start_time = time.time()
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            break
        
        elapsed_time = time.time() - start_time

        # If 5 seconds have passed, start detecting emotions
        if elapsed_time > 5 and not first_emotion_detected:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loop through each detected face
            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                emotion = predict_emotion(face)

                if not first_emotion_detected:  # Detect emotion only once
                    first_emotion_detected = emotion
                    print(f"First Detected Emotion: {emotion}")
                    # Open the Flask server to show the songs
                    webbrowser.open(f'http://localhost:5000/songs/{emotion}')
                    
        # Display the resulting frame
        cv2.putText(frame, f"Emotion: {first_emotion_detected or 'Detecting...'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the webcam thread to process the webcam
def start_webcam_thread():
    thread = Thread(target=process_webcam)
    thread.start()

if __name__ == "__main__":
    # Start the webcam thread
    start_webcam_thread()

    # Start the Flask app
    app.run(debug=True, use_reloader=False)
