# Digit Recognition Neural Network

A neural network implementation to recognize handwritten digits from the MNIST dataset, written from scratch in Python using NumPy.

## Description

This project implements a feedforward neural network with one hidden layer to classify handwritten digits (0-9) from the MNIST dataset. The network is built using only NumPy.

## Architecture

- Input layer: 784 neurons (28x28 pixel images)
- Hidden layer: 128 neurons with ReLU activation
- Output layer: 10 neurons with Softmax activation
- Loss function: Cross-entropy loss
- Learning rate: 0.063 with decay

## Hyperparameters

The default hyperparameters are:
- Learning rate: 0.063
- Batch size: 64
- Number of epochs: 20
- Hidden layer size: 128

You can modify these in the `train.py` file to experiment with different settings.

## Results

With the default settings, the model should achieve around 97% accuracy on the test set.


## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Keras (only for downloading MNIST data)

### Installation

1. Clone the repository:
```sh
git clone https://github.com/I-had-a-bad-idea/Digit-Recognition-Neural-Network.git
cd Digit-Recognition-Neural-Network
```

2. Download and prepare the MNIST dataset:
```sh
python data/get_data.py
```

### Usage

To train the network:
```sh
python train.py
```

The model will:
- Train for 20 cycles
- Use batch size of 64
- Save the trained model parameters in the `model/` directory

## Project Structure

```
├── data/                  # Dataset directory
│   ├── get_data.py       # Script to download MNIST data
│   └── *.npy             # MNIST data files
├── model/               # Saved model parameters
├── data_loader.py       # Data loading utilities
├── neural_network.py    # Neural network implementation
├── utils.py             # Helper functions
└── train.py             # Training script
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file