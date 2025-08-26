import numpy as np
from keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Save the data as .npy files
np.save("data/train_labels.npy", y_train)
np.save("data/train_images.npy", x_train)
np.save("data/test_images.npy", x_test)
np.save("data/test_labels.npy", y_test)

print("MNIST dataset has been saved as .npy files!")