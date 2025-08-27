import numpy as np
from keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images
x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0

# One-hot encode labels
num_classes = 10
y_train_one_hot = np.zeros((y_train.size, num_classes))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

y_test_one_hot = np.zeros((y_test.size, num_classes))
y_test_one_hot[np.arange(y_test.size), y_test] = 1

# Save the data as .npy files
np.save("data/train_labels.npy", y_train_one_hot)
np.save("data/train_images.npy", x_train)
np.save("data/test_images.npy", x_test)
np.save("data/test_labels.npy", y_test_one_hot)

print("MNIST dataset has been saved as .npy files!")