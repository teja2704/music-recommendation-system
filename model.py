import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from project_config import IMAGE_SIZE, NUM_CLASSES

# Define CNN model architecture
def create_model():
    model = models.Sequential()

    # First convolutional block
    model.add(
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    )
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
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

    # Output layer
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    create_model().summary()
