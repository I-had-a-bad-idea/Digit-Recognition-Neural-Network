import numpy as np
import time
import os

from data_loader import load_data, get_batch
from neural_network import NeuralNetwork
from utils import evaluate_model

input_size = 784
hidden_size = 32
output_size = 10
learning_rate = 0.01
batch_size = 64
epochs = 10

def main():

    print("Loading MNIST dataset...")
    train_images, train_labels, test_images, test_labels = load_data()
    print("Loaded MNIST dataset!")

    print("Initializing network")
    neural_network = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    print("Network initialized")

    num_samples = train_images.shape[0]
    num_batches = num_samples // batch_size

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        epoch_accuracy = 0

        indices = np.random.permutation(num_samples)
        images = train_images[indices]
        labels = train_labels[indices]

        for batch in range(num_batches):
            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size

            x_batch = images[batch_start:batch_end]
            y_batch = labels[batch_start:batch_end]

            loss, accuracy = neural_network.train_step(x_batch, y_batch)
            epoch_loss += loss
            epoch_accuracy += accuracy

            if (batch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch + 1}/{num_batches}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        test_accuracy, test_loss = evaluate_model(neural_network, test_images, test_labels)

        epoch_time = time.time() - start_time
        print(
            f"Expoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s"
            f"Avg. Loss: {epoch_loss/num_batches:.4f}"
            f"Avg. Accuracy: {epoch_accuracy/num_batches:.4f}%"
            f"Test Loss: {test_loss:.4f}"
            f"Test Accuracy: {test_accuracy:.4f}%"
        )
    
    print("Training completed")

    final_accuracy, final_loss = evaluate_model(neural_network, test_images, test_labels)
    print(f"Final Test Accuracy: {final_accuracy:.4f}%")
    print(f"Final Test Loss: {final_loss:.4f}")

    print("Saving network parameters...")
    os.makedirs("models", exist_ok=True)
    np.save("models/w1.npy", neural_network.w1)
    np.save("models/b1.npy", neural_network.b1)
    np.save("models/w2.npy", neural_network.w2)
    np.save("models/b2.npy", neural_network.b2)
    
    print("Model saved")


if __name__ == "__main__":
    main()
