import numpy as np
from utils import calculate_loss, calculate_accuracy

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate

        layer_sizes = [input_size] + hidden_layers + [output_size]

        # Initialize weights and biases for all layers
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_pass(self, x):
        activations = [x]
        zs = []

        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.relu(z)
            zs.append(z)
            activations.append(a)

        # Last layer = softmax
        z_final = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a_final = self.softmax(z_final)
        zs.append(z_final)
        activations.append(a_final)

        return a_final, {"x": x, "activations": activations, "zs": zs}

    def backward_pass(self, y, cache):
        batch_size = y.shape[0]
        activations = cache["activations"]
        zs = cache["zs"]

        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Output layer error
        delta = activations[-1] - y
        grads_w[-1] = (1 / batch_size) * np.dot(activations[-2].T, delta)
        grads_b[-1] = (1 / batch_size) * np.sum(delta, axis=0, keepdims=True)

        # Backprop through hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(delta, self.weights[i+1].T) * self.relu_derivative(zs[i])
            grads_w[i] = (1 / batch_size) * np.dot(activations[i].T, delta)
            grads_b[i] = (1 / batch_size) * np.sum(delta, axis=0, keepdims=True)

        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def train_step(self, x, y):
        output, cache = self.forward_pass(x)
        grads_w, grads_b = self.backward_pass(y, cache)
        self.update_parameters(grads_w, grads_b)

        loss = calculate_loss(output, y)
        accuracy = calculate_accuracy(np.argmax(output, axis=1), y)

        self.learning_rate *= 0.9999
        return loss, accuracy

    def predict(self, x):
        output, _ = self.forward_pass(x)
        return np.argmax(output, axis=1)
