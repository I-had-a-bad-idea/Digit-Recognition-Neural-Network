import numpy as np
import tensorflow_datasets as tfds
from PIL import Image, ImageEnhance, ImageChops
import random
import cv2
import os
os.makedirs("data", exist_ok=True)

from augment_external_digits import Augmented_digits_generator

# Load EMNIST digits
print("Loading EMNIST digits...")
ds_train = tfds.load("emnist/digits", split="train", as_supervised=True)
ds_test = tfds.load("emnist/digits", split="test", as_supervised=True)
print("Loaded EMNIST digits!")

augmented_digits_generator = Augmented_digits_generator()

print("Generating augmented digits...")
# Generate synthetic digits from PNGs
x_extra, y_extra = augmented_digits_generator.generate_augmented_data("E:/Digit-Recognition-Neural-Network/data/external_digits", samples_per_image=100)
print("Generated augmented pictures!")

# Convert dataset to NumPy
def to_numpy(dataset, preprocess_fn):
    images, labels = [], []
    for img, label in tfds.as_numpy(dataset):
        arr = preprocess_fn(img)
        images.append(arr)
        labels.append(label)
    return np.array(images), np.array(labels)


print("Preprocessing EMNIST digits...")
# Training set gets augmentation
x_train, y_train = to_numpy(ds_train, augmented_digits_generator.preprocess_digit)
x_test, y_test = to_numpy(ds_test, augmented_digits_generator.preprocess_digit)

# One-hot labels
print("Generating EMNIST labels...")
num_classes = 10
y_train_one_hot = np.zeros((y_train.size, num_classes))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

y_test_one_hot = np.zeros((y_test.size, num_classes))
y_test_one_hot[np.arange(y_test.size), y_test] = 1

print("Merging EMNIST and external digits...")
# Merge PNGs with EMNIST
x_train = np.concatenate([x_train, x_extra], axis=0)
y_train = np.concatenate([y_train, y_extra], axis=0)

# Rebuild one-hot labels
y_train_one_hot = np.zeros((y_train.size, num_classes))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

print("Saving...")

# Save
np.save("data/train_labels.npy", y_train_one_hot)
np.save("data/train_images.npy", x_train)
np.save("data/test_images.npy", x_test)
np.save("data/test_labels.npy", y_test_one_hot)

print("EMNIST/digit dataset has been saved as .npy files!")
