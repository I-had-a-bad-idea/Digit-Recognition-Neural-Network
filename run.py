import numpy as np
import os
from PIL import Image, ImageOps

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
    image = image.resize((28, 28))

    # Invert colors: make digits white on black
    image = ImageOps.invert(image)

    # Normalize to 0–1
    image_array = np.array(image) / 255.0

    # Flatten to match model input
    flattened = image_array.reshape(1, 784).astype(np.float32)

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