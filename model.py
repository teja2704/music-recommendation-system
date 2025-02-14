import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

# Define CNN model architecture
def create_model():
    model = models.Sequential()

    # First convolutional block
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Second convolutional block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Third convolutional block
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    # Flatten the output from convolutional layers to feed into dense layers
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output layer (7 emotions)
    model.add(layers.Dense(7, activation='softmax'))  # Softmax for multi-class classification

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create and return the model
model = create_model()

# Display the model architecture
model.summary()
