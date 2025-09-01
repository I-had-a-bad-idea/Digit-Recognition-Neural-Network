import numpy as np
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork


def load_model():
    w1 = np.load("model/w1.npy")
    b1 = np.load("model/b1.npy")
    w2 = np.load("model/w2.npy")
    b2 = np.load("model/b2.npy")

    input_size = w1.shape[0]
    hidden_size = w1.shape[1]
    output_size = w2.shape[1]

    model = NeuralNetwork(input_size, hidden_size, output_size)

    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")

    # Convert to numpy and binarize
    arr = np.array(image)
    arr = (arr > 128).astype(np.uint8) * 255

    # Compute center of mass
    coords = np.argwhere(arr > 0)
    if coords.shape[0] == 0:
        raise ValueError("No digit found in image!")

    y_center, x_center = coords.mean(axis=0)

    # Create 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    # Scale digit to fit in 20x20 box
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = arr[y0:y1, x0:x1]

    h, w = cropped.shape
    scale = 20.0 / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    digit = Image.fromarray(cropped).resize((new_w, new_h), Image.Resampling.LANCZOS)
    digit = np.array(digit)

    # Compute new center of mass
    coords = np.argwhere(digit > 0)
    y_center, x_center = coords.mean(axis=0)

    # Paste into 28x28 with COM centering
    y_offset = int(round(14 - y_center))
    x_offset = int(round(14 - x_center))

    y_start = max(0, y_offset)
    x_start = max(0, x_offset)
    y_end = min(28, y_offset + new_h)
    x_end = min(28, x_offset + new_w)

    canvas[y_start:y_end, x_start:x_end] = digit[
        0 : (y_end - y_start), 0 : (x_end - x_start)
    ]

    # Normalize + flatten
    image_array = canvas.astype(np.float32) / 255.0
    flattened = image_array.reshape(1, 784)

    return flattened


def predict_digit(model, image_path):
    image_data = preprocess_image(image_path) 
    output, _ = model.forward_pass(image_data)
    probabilities = output[0]
    prediction = np.argmax(output, axis=1)

    return prediction, probabilities

def run():

    image_path = input("Please enter the path to the image: \t")
    if not os.path.isfile(image_path):
        print("Please enter valid path")
        return
    
    model = load_model()

    prediction, probabilities = predict_digit(model, image_path)

    print(f"Predicted digit: {prediction}")
    print("Probabilities:")
    for digit, prob in enumerate(probabilities):
        print(f"  Digit {digit}: {prob*100:.2f}%")


run()