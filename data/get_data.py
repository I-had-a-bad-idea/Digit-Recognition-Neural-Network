import numpy as np
from keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Save the data as .npy files
np.save("data/y_train.npy", y_train)
np.save("data/x_train.npy", x_train)
np.save("data/x_test.npy", x_test)
np.save("data/y_test.npy", y_test)

print("MNIST dataset has been saved as .npy files!")