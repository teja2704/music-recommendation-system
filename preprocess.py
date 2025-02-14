import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split

# Define constants
IMG_SIZE = 48  # Image size (FER-2013 is 48x48)
NUM_CLASSES = 7  # Number of emotions
DATASET_PATH = r"C:\Users\patti\OneDrive\Desktop\Facial- emotion-detection\dataset"  # Use raw string to avoid escape sequence issues

# Define emotions (based on folder names)
EMOTION_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

# Function to load images and labels
def load_data(directory):
    images = []
    labels = []
    
    print(f"📂 Loading data from: {directory}...")  # Debugging
    
    for label, emotion in enumerate(EMOTION_LABELS):
        emotion_path = os.path.join(directory, emotion)
        
        if not os.path.exists(emotion_path):  
            print(f"❌ Folder not found: {emotion_path}")  # Debugging
            continue  # Skip missing folders

        num_images = 0  # Track loaded images

        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"❌ Couldn't read image: {img_path}")  # Debugging
                continue
            
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
            num_images += 1
        
        print(f"✅ Loaded {num_images} images for {emotion}")  # Debugging
    
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    images = images / 255.0  # Normalize pixel values (0-1)
    labels = to_categorical(labels, num_classes=NUM_CLASSES)  # One-hot encode labels
    
    print(f"✅ Finished loading {len(images)} images from {directory}\n")  # Debugging
    return images, labels

# Load training and testing data
X_train, y_train = load_data(os.path.join(DATASET_PATH, "train"))
X_test, y_test = load_data(os.path.join(DATASET_PATH, "test"))

# Split training data into train & validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Print dataset shapes
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Testing data shape: {X_test.shape}")

# Save preprocessed data (optional)
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("✅ Data preprocessing complete!")
