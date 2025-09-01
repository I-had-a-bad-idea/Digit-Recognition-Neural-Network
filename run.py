import numpy as np
import os
from PIL import Image

from neural_network import NeuralNetwork
from data.augment_external_digits import Augmented_digits_generator


def load_model():
    metadata_npz = np.load("model/metadata.npz", allow_pickle=True)
    metadata = {k: metadata_npz[k].tolist() for k in metadata_npz}
    input_size = metadata["input_size"]
    hidden_layers = metadata["hidden_layers"]
    output_size = metadata["output_size"]
    learning_rate = metadata["learning_rate"]

    model = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate)

    # Load all layers
    for i in range(len(model.weights)):
        model.weights[i] = np.load(f"model/w{i}.npy")
        model.biases[i] = np.load(f"model/b{i}.npy")

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
