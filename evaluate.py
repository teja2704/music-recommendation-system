import numpy as np
import tensorflow as tf

# Load the preprocessed test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load the trained model
model = tf.keras.models.load_model("emotion_recognition_model.h5")

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
