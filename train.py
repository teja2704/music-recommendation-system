import numpy as np
from model import create_model

# Load the preprocessed data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Create the model
model = create_model()

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))

# Save the trained model
model.save("emotion_recognition_model.h5")

print("✅ Model training complete!")
