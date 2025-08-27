import numpy as np
from utils import calculate_loss, calculate_accuracy

class NeuralNetwork:
        def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
            
            self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
            self.b1 = np.zeros((1, hidden_size))
            self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
            self.b2 = np.zeros((1, output_size))
            self.learning_rate = learning_rate


        def relu(self, x):
            return np.maximum(0, x)
        
        def relu_derivative(self, x):
            return np.where(x > 0, 1, 0)
        
        # Converts output into probability distribution 
        def softmax(self, x):
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

        def forward_pass(self, x):
              
            z_1 = np.dot(x, self.w1) + self.b1
            activation_1 = self.relu(z_1)
            z_2 = np.dot(activation_1, self.w2) + self.b2
            activation_2 = self.softmax(z_2)

            values = {"x": x, "z1": z_1, "a1": activation_1, "z2": z_2, "a2": activation_2}

            return activation_2, values
        
        def backward_pass(self, y, values):
            batch_size = y.shape[0]

            x = values["x"]
            activation_1 = values["a1"]
            activation_2 = values["a2"]
            z_1 = values["z1"]

            error_activation_2 = activation_2 - y
            gradient_w2 = (1 / batch_size) * np.dot(activation_1.T, error_activation_2)
            gradient_b2 = (1 / batch_size) * np.sum(error_activation_2, axis=0, keepdims=True)
            
            error_activation_1 = np.dot(error_activation_2, self.w2.T)
            error_z_1 = error_activation_1 * self.relu_derivative(z_1)
            gradient_w1 = (1 / batch_size) * np.dot(x.T, error_z_1)
            gradient_b1 = (1 / batch_size) * np.sum(error_z_1, axis=0, keepdims=True)

            gradients = {"w1": gradient_w1, "b1": gradient_b1, "w2": gradient_w2, "b2": gradient_b2}

            return gradients

        def update_parameters(self, gradients):

            self.w1 -= self.learning_rate * gradients["w1"]
            self.b1 -= self.learning_rate * gradients["b1"]
            self.w2 -= self.learning_rate * gradients["w2"]
            self.b2 -= self.learning_rate * gradients["b2"]     

        def train_step(self, x, y):
            
            output, values = self.forward_pass(x)
            gradients = self.backward_pass(y, values)

            self.update_parameters(gradients)
            loss = calculate_loss(output, y)
            accuracy = calculate_accuracy(np.argmax(output, axis=1), y)

            return loss, accuracy
        
        def predict(self, x):
            
            output, _ = self.forward_pass(x)
            return np.argmax(output, axis=1)
        
