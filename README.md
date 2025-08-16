# Simple Neural Network for MNIST

A simple neural network implementation for digit recognition using the MNIST dataset.

## Overview

This project implements a basic neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset. The network has a simple architecture with:
- Input layer: 784 nodes (28x28 pixel images)
- Hidden layer: 10 nodes with Leaky ReLU activation
- Output layer: 10 nodes (digits 0-9) with Softmax activation

## Files

- `main.py` - Main training script for the neural network
- `visualize_network.py` - Visualization tool for the neural network
- `train.csv` - MNIST training dataset
- `test.csv` - MNIST test dataset
- `trained_weights.pkl` - Pre-trained model weights
- `requirements.txt` - Python dependencies

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python main.py
```

This will:
- Load and preprocess the MNIST training data
- Train the neural network
- Save the trained weights to `trained_weights.pkl`
- Display training progress and accuracy

### Visualizing the Network
```bash
python visualize_network.py
```

This will:
- Load the trained weights
- Display a visualization of the neural network
- Show activations for sample inputs

## Dependencies

- numpy==2.3.2
- pandas==2.3.1
- matplotlib (for visualization)
- pickle (for saving/loading weights)

## Project Structure

The neural network uses:
- **Forward Propagation**: Computes activations through the network
- **Backward Propagation**: Computes gradients for weight updates
- **Gradient Descent**: Updates weights to minimize loss
- **Leaky ReLU**: Activation function for hidden layer
- **Softmax**: Activation function for output layer

## Results

The model typically achieves around 85-90% accuracy on the MNIST test set after training. 