import numpy as np

# Calculates the accuracy of the predictions 
def calculate_accuracy(predictions, labels):

    actual_labels = np.argmax(labels, axis=1) # Get actual value
    return np.mean(predictions == actual_labels) * 100 # Compare and return percentage

# Calulate loss by how unconfident/ wrong network is
def calculate_loss(output, y):
    output = np.clip(output, 1e-10, 1.0) # Prevent log(0)
    return -np.mean(np.sum(y * np.log(output), axis=1)) 

# Connvert the integer labels into vectors
def one_hot_encode(labels, num_classes = 10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

# Evaluate the model based on some test_images
def evaluate_model(model, test_images, test_labels, batch_size = 100):
    num_samples = test_images.shape[0] # Number of testt_images
    num_batches = (num_samples + batch_size - 1) // batch_size

    total_loss = 0 
    all_predictions = []

    # Loop through all batches
    for i in range(num_batches):
        start_index = i * batch_size # Start of batch
        end_index = min((i + 1) * batch_size, num_samples) # End of batch

        batch_images = test_images[start_index:end_index]
        batch_labels = test_labels[start_index:end_index]

        # Run it through the model
        output, _ = model.forward_pass(batch_images)
        prediction = np.argmax(output, axis=1) # Convert vector to integer
        all_predictions.extend(prediction)

        loss = calculate_loss(output, batch_labels) # Get the loss for the entire batch
        total_loss += loss * (end_index - start_index)
    
    average_loss = total_loss / num_samples
    accuracy = calculate_accuracy(np.array(all_predictions), test_labels)

    return accuracy, average_loss



