# Reconstruction of the LeNet-5 model from the video
#
# Michael Yarmoshik
# 2021-10-21

# Imports
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Model LeNet-5
# I pulled this information from the video
model = models.Sequential([

    layers.ZeroPadding2D(padding=2, input_shape=(28, 28, 1)),

    layers.Conv2D(6, kernel_size=(5, 5), activation="sigmoid"),
    layers.AveragePooling2D(pool_size=(2, 2), strides=2),

    layers.Conv2D(16, kernel_size=(5, 5), activation="sigmoid"),
    layers.AveragePooling2D(pool_size=(2, 2), strides=2),

    layers.Flatten(),

    layers.Dense(120, activation="sigmoid"),
    layers.Dense(84, activation="sigmoid"),
    layers.Dense(10, activation="sigmoid")

])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train/eval the model
model.fit(x_train, y_train, epochs=15)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

# Print accuracy & loss
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# Test loss: 0.039618462324142456
# Test accuracy: 0.9864000082015991