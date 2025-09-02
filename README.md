# Digit Recognition Neural Network

A neural network implementation for handwritten digit recognition using the EMNIST dataset, built from scratch in Python using NumPy. The network combines EMNIST training data with custom augmented samples for improved robustness.

---

## Features

- Pure NumPy implementation (no deep learning frameworks)
- Multi-layer neural network architecture
- Custom data augmentation pipeline
- Support for external digit images
- Real-time training metrics
- Model persistence and loading
- Batch processing for efficient training
- Automatic learning rate decay
- Custom digit preprocessing pipeline

---

## Architecture

### Network Structure
- Input layer: 784 neurons (28x28 pixel images)
- Hidden layers: 
  - Layer 1: 256 neurons with ReLU activation
  - Layer 2: 128 neurons with ReLU activation
- Output layer: 10 neurons with Softmax activation

### Components
- Loss function: Cross-entropy loss
- Activation functions:
  - Hidden layers: ReLU (Rectified Linear Unit)
  - Output layer: Softmax
- Weight initialization: He initialization
- Learning rate: 0.5 with Cosine Annealing

## Parameters

The default parameters are:
- Initial learning rate: 0.5
- Batch size: 128
- Training cycles: 20
- Hidden layer sizes: [128, 128]

These can be modified in `train.py` to experiment with different configurations.

---

## Performance

- Training accuracy: ~98.8%
- Test accuracy: ~98.7% on EMNIST test set
- Training time: ~3-7 minutes depending on CPU
- Memory usage: ~65MB RAM during training

---

## Installation

### Prerequisites

- Python 3.8+
- NumPy
- TensorFlow Datasets (for EMNIST data)
- Pillow (PIL) (for training data augmentation)
- SciPy (for training data augmentation)

> **Note:** If you only want to run the already trained model, you only need Python and NumPy

### Setup

1. Clone the repository:
```sh
git clone https://github.com/yourusername/Digit-Recognition-Neural-Network.git
cd Digit-Recognition-Neural-Network
```

2. Install dependencies:
```sh
pip install numpy tensorflow-datasets pillow scipy # If you dont want to train it, you only need numpy
```

---

## Usage

### Training

1. Download and prepare the data:
```sh
python data/get_data.py
```

2. Train the network:
```sh
python train.py
```

The model will:
- Load and preprocess the EMNIST dataset
- Augment training data
- Train for 20 cycles
- Save model parameters in `model/`
- Display real-time training metrics

### Prediction

To recognize digits:
```sh
python run.py
```

Then enter the path to your image when prompted.

---

## Project Structure

```
├── data/
│   ├── get_data.py           # Dataset download and preparation
│   ├── augment_external_digits.py  # Data augmentation utilities
│   └── external_digits/      # Custom digit images
├── model/                    # Saved model parameters
│   ├── metadata.npz         # Network configuration
│   ├── w*.npy              # Layer weights
│   └── b*.npy              # Layer biases
├── neural_network.py        # Neural network implementation
├── data_loader.py          # Data loading utilities
├── utils.py                # Helper functions
├── train.py               # Training script
└── run.py                 # Prediction script
```

---

## Implementation Details

### Data Preprocessing
- Image normalization (0-1 range)
- Center of mass centering
- Size standardization (28x28 pixels)
- Binarization with adaptive thresholding

### Data Augmentation
- Rotation (-20° to +20°)
- Elastic distortions
- Random shifts
- Contrast adjustment
- Gaussian noise
- Blur variations

### Training Process
- Batch-based training
- Automatic learning rate decay
- Random batch shuffling
- Real-time accuracy monitoring
- Model checkpointing

---

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
