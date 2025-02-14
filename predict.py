import cv2
import numpy as np
import tensorflow as tf
import random
import webbrowser  # To open YouTube links

# Load the trained model
model = tf.keras.models.load_model("emotion_recognition_model.h5")

# Define emotion labels
EMOTION_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Define music recommendations for each emotion
MUSIC_RECOMMENDATIONS = {
    "angry": ["https://www.youtube.com/watch?v=2vjPBrBU-TM",  # Example: Rock song
              "https://www.youtube.com/watch?v=YykjpeuMNEk"], # Another rock song

    "disgusted": ["https://www.youtube.com/watch?v=o3mP3mJDL2k"],  # Dark classical

    "fearful": ["https://www.youtube.com/watch?v=UfcAVejslrU"],  # Ambient

    "happy": ["https://www.youtube.com/watch?v=09R8_2nJtjg",  
              "https://www.youtube.com/watch?v=kJQP7kiw5Fk"],  # Pop/Dance

    "neutral": ["https://www.youtube.com/watch?v=3HjG1Y4QpVA"],  # Lo-Fi

    "sad": ["https://www.youtube.com/watch?v=J_ub7Etch2U"],  # Acoustic

    "surprised": ["https://www.youtube.com/watch?v=GRz4FY0ZcwI"]  # EDM
}

# Function to predict emotion from an image
def predict_emotion(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0  # Normalize
    img = img.reshape(1, 48, 48, 1)

    # Predict emotion
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    emotion = EMOTION_LABELS[predicted_class]

    return emotion

# Function to recommend a song based on detected emotion
def recommend_music(emotion):
    if emotion in MUSIC_RECOMMENDATIONS:
        song_link = random.choice(MUSIC_RECOMMENDATIONS[emotion])
        print(f"🎵 Recommended song for {emotion}: {song_link}")
        webbrowser.open(song_link)  # Open song in browser
    else:
        print("No song available for this emotion.")

# Example usage
image_path = r"C:\Users\patti\Downloads\OIP.jpg"  # Replace with image path
predicted_emotion = predict_emotion(image_path)
print(f"The predicted emotion is: {predicted_emotion}")

# Recommend and play music
recommend_music(predicted_emotion)
