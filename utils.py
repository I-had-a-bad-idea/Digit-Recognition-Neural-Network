import numpy as np

# Calculates the accuracy of the predictions 
def calculate_accuracy(predictions, labels):

    actual_labels = np.argmax(labels, axis=1) # Get actual value
    return np.mean(predictions == actual_labels) * 100 # Compare and return percentage


def calculate_loss(output, y):
    output = np.clip(output, 1e-10, 1.0) # Prevent log(0)
    return -np.mean(np.sum(y * np.log(output), axis=1)) # Calulate loss by how confident/ correct network is


def one_hot_encode(labels, num_classes = 10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def evaluate_model(model, test_images, test_labels, batch_size = 100):
    num_samples = test_images.shape[0]
    num_batches = (num_samples)

    total_loss = 0
    all_predictions = []
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, num_samples)

        batch_images = test_images[start_index:end_index]
        batch_labels = test_labels[start_index:end_index]

        output = [] # Replace with the model call
        prediction = np.argmax(output, axis=1)
        all_predictions.extend(prediction)

        loss = calculate_loss(output, batch_labels)
        total_loss += loss * (end_index - start_index)
    
    average_loss = total_loss / num_samples
    accuracy = calculate_accuracy(np.array(all_predictions), test_labels)

    return accuracy, average_loss



