import numpy as np
import time
import os

from data_loader import load_data
from neural_network import NeuralNetwork
from utils import evaluate_model

input_size = 784
hidden_layers = [128, 128]
output_size = 10
learning_rate = 0.5
batch_size = 128
cycles = 10

starting_seed = 95
number_of_seeds = 5

def train():
    start_time = time.time()

    print("Loading MNIST dataset...")
    train_images, train_labels, test_images, test_labels = load_data()
    print("Loaded MNIST dataset!")

    print("Initializing network")
    neural_network = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate)
    print("Network initialized")

    num_samples = train_images.shape[0]
    num_batches = num_samples // batch_size

    print(f"Starting training for {number_of_seeds} seeds with {cycles} cycles each...")

    for i in range(number_of_seeds):
        np.random.seed(starting_seed + i)
        for cycle in range(cycles):
            cycle_start_time = time.time()
            cycle_loss = 0
            cycle_accuracy = 0

            indices = np.random.permutation(num_samples)
            images = train_images[indices]
            labels = train_labels[indices]

            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = (batch + 1) * batch_size

                x_batch = images[batch_start:batch_end]
                y_batch = labels[batch_start:batch_end]

                loss, accuracy = neural_network.train_step(x_batch, y_batch)
                neural_network.learning_rate = learning_rate * (0.5 * (1 + np.cos(np.pi * cycle / cycles)))
                cycle_loss += loss
                cycle_accuracy += accuracy

                if (batch + 1) % 100 == 0:
                    print(f"seed {i + 1}/{number_of_seeds}, cycle {cycle + 1}/{cycles}, Batch {batch + 1}/{num_batches}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            test_accuracy, test_loss = evaluate_model(neural_network, test_images, test_labels)

            cycle_time = time.time() - cycle_start_time
            print(
                f"Seed {i + 1}/{number_of_seeds} "
                f"Cycle {cycle + 1}/{cycles} completed in {cycle_time:.2f}s "
                f"Avg. Loss: {cycle_loss/num_batches:.4f} "
                f"Avg. Accuracy: {cycle_accuracy/num_batches:.4f}% "
                f"Test Loss: {test_loss:.4f} "
                f"Test Accuracy: {test_accuracy:.4f}% "
                f"Learning rate: {neural_network.learning_rate} "
            )
        
    print("Training completed")

    final_accuracy, final_loss = evaluate_model(neural_network, test_images, test_labels)
    training_time = time.time() - start_time
    print(
        f"Final Test Accuracy: {final_accuracy:.4f}%\n"
        f"Final Test Loss: {final_loss:.4f}\n"
        f"Training took {training_time:.1f}s\n"
        )

    print("Saving network parameters...")
    os.makedirs("model", exist_ok=True)

    # Save metadata (layer sizes, learning rate, etc.)
    metadata = {
        "input_size": input_size,
        "hidden_layers": [w.shape[1] for w in neural_network.weights[:-1]],
        "output_size": output_size,
        "learning_rate": neural_network.learning_rate,
    }
    np.savez("model/metadata.npz", **metadata)

    # Save weights and biases
    for i, (w, b) in enumerate(zip(neural_network.weights, neural_network.biases)):
        np.save(f"model/w{i}.npy", w)
        np.save(f"model/b{i}.npy", b)

    print("Model saved")

if __name__ == "__main__":
    train()
