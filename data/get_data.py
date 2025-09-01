import numpy as np
import tensorflow_datasets as tfds
from PIL import Image, ImageEnhance, ImageChops
import random
import cv2

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

def augment_image(img):
    if img.ndim == 3 and img.shape[-1] == 1:   # (28,28,1) → (28,28)
        img = img.squeeze(-1)

    pil_img = Image.fromarray(img.astype(np.uint8), mode="L")  # force grayscale

    if random.random() > 0.5:
        kernel = np.ones((2,2), np.uint8)
        if random.random() < 0.5:
            # Thicken
            cv2.dilate(img, kernel, iterations=1)
        else:
            # Thin
            cv2.erode(img, kernel, iterations=1)

    # Random rotation ±15°
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        pil_img = pil_img.rotate(angle)

    # Random shifts with offset
    if random.random() < 0.5:
        dx, dy = random.randint(-2, 2), random.randint(-2, 2)
        from PIL import ImageChops
        pil_img = ImageChops.offset(pil_img, dx, dy)

    # Random contrast
    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)

    # Random brightness
    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)

    return np.array(pil_img)

# Convert dataset to NumPy with augmentation
def to_numpy(dataset, augment=False):
    images, labels = [], []
    for img, label in tfds.as_numpy(dataset):
        if augment:
            img = augment_image(img)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

print("Last modificatiosn to EMNIST digits!")
# Training set gets augmentation
x_train, y_train = to_numpy(ds_train, augment=False)
x_test, y_test = to_numpy(ds_test, augment=False)

# Flatten + normalize
x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0

# One-hot labels
print("Generating EMNIST labels!")
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
