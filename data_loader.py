import numpy as np


def load_data():

    train_images = np.load("data/train_images.npy")
    train_labels = np.load("data/train_labels.npy")
    test_images = np.load("data/test_images.npy")
    test_lables = np.load("data/test_labels.npy")


    return train_images, train_labels, test_images, test_lables


def get_batch(images, labels, batch_size):

    indices = np.random.permutation(images.shape[0][:batch_size])
    return images[indices], labels[indices]
