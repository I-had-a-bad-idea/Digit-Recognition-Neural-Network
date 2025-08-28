import numpy as np


def load_data():

    train_images = np.load("data/train_images.npy")
    train_labels = np.load("data/train_labels.npy")
    test_images = np.load("data/test_images.npy")
    test_lables = np.load("data/test_labels.npy")


    return train_images, train_labels, test_images, test_lables

