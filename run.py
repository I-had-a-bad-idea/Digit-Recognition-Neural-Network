import numpy as np
import os
from PIL import Image

from neural_network import NeuralNetwork
from data.augment_external_digits import Augmented_digits_generator


def load_model():
    w1 = np.load("model/w1.npy")
    b1 = np.load("model/b1.npy")
    w2 = np.load("model/w2.npy")
    b2 = np.load("model/b2.npy")

    input_size = w1.shape[0]
    hidden_size = w1.shape[1]
    output_size = w2.shape[1]

    model = NeuralNetwork(input_size, hidden_size, output_size)
    model.w1, model.b1, model.w2, model.b2 = w1, b1, w2, b2  # load weights
    return model


def preprocess_image(image_path):
    generator = Augmented_digits_generator()
    img = Image.open(image_path).convert("L")
    return generator.preprocess_digit(img).reshape(1, -1)


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
