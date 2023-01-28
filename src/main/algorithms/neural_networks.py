from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

"""
    Program Objective: Look at clothing/apparel images and classify them appropriately
    Data: Fashion MNIST dataset from keras: https://www.tensorflow.org/datasets/catalog/fashion_mnist
"""

# Load data
data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scaling the pixel values down to make computations easier
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
    keras.layers.Dense(units=128, activation="relu"),  # Hidden layer
    keras.layers.Dense(units=10, activation="softmax"),  # Output layer (10 classes)
])

# Picking the optimizer, loss function and metrics to keep track of
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(train_images, train_labels, epochs=5)

# Testing model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy: ', test_acc)

# Making predictions
predictions = model.predict(test_images)

for i in range(5):
    plt.figure(num="Fashion Prediction", figsize=(5, 5))
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.show()
