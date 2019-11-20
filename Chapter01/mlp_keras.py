# Author: Roi Yehoshua
# October 2019

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

# Setting random seed
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)
# In TF 2.0: tf.random.set_seed(SEED)

# Load the Cifar10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('x_train data type:', x_train.dtype)

# Print the number of samples in each category
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Plot the first 50 images from the training set
fig, axes = plt.subplots(5, 10)
k = 0
for ax in axes.flat:
    ax.imshow(x_train[k])
    ax.axis('off')
    ax.set_title(class_names[y_train[k][0]], fontsize=16)
    k += 1
plt.show()

# Scale the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the network as an MLP with two hidden layers
model = keras.models.Sequential()
model.add(layers.Flatten(input_shape=[32, 32, 3]))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# The same as:
# model = keras.models.Sequential([
#     layers.Flatten(input_shape=[32, 32, 3]),
#     layers.Dense(200, activation='relu'),
#     layers.Dense(100, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

model.summary()

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=30, validation_split=0.1)

# Plot the learning curve
pd.DataFrame(history.history).plot(figsize=(8, 5), fontsize=18)
plt.grid(True)
plt.xlabel('Epoch', fontsize=20)
plt.legend(fontsize=18)
plt.title('MLP for CIFAR-10 Classification', fontsize=20)
plt.show()

# Evaluate the model on the test set
results = model.evaluate(x_test, y_test)
results = np.round(results, 4)
print(f'Test loss: {results[0]}, test accuracy: {results[1]}')

# Using the model to make predictions
x_new = x_test[:5]
y_prob = model.predict(x_new)
print(y_prob.round(3))

y_pred = model.predict_classes(x_new)
print(y_pred)

# Plot the first 5 images from the test set, their true labels and predicted labels
fig, axes = plt.subplots(1, 5)
k = 0
for ax in axes.flat:
    ax.imshow(x_test[k])
    ax.axis('off')
    title = f'{class_names[y_test[k][0]]} ({class_names[y_pred[k]]})'
    ax.set_title(title, fontsize=18)
    k += 1
plt.show()











