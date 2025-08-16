import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import pickle
import os

def load_data():
    """Load and prepare MNIST data"""
    data = pd.read_csv('train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)
    
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.
    
    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.
    
    return X_train, Y_train, X_dev, Y_dev

def leaky_ReLU(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)

def softmax_improved(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = leaky_ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax_improved(Z2)
    return Z1, A1, Z2, A2

def load_trained_weights():
    """Load trained weights from file"""
    if os.path.exists('trained_weights.pkl'):
        with open('trained_weights.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        print("No trained weights found. Please run main.py first to train the model.")
        return None

def visualize_simple_network(W1, b1, W2, b2, X_train, Y_train, sample_idx=0):
    """
    Simple visualization of neural network with 784 input, 10 hidden, 10 output nodes
    """
    # Get activations for the sample
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train[:, sample_idx:sample_idx+1])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))  # Square 600x600 equivalent
    ax.set_xlim(-1, 4)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Network structure
    input_size = 784
    hidden_size = 10
    output_size = 10
    
    # Draw input layer (784 nodes - simplified as a dense block)
    input_y_positions = np.linspace(-1.8, 1.8, 20)  # Show 20 representative nodes
    input_circles = []
    for i in range(20):
        # Scale input activation (simplified)
        input_activation = X_train[i*39, sample_idx] if i*39 < 784 else 0
        intensity = min(1.0, input_activation * 3)
        circle = Circle((0, input_y_positions[i]), 0.05, 
                      color='darkblue', alpha=0.3 + intensity * 0.7)
        ax.add_patch(circle)
        input_circles.append(circle)
    
    # Add input layer label
    ax.text(0, -1.9, 'Input Layer\n(784 nodes)', ha='center', va='center', 
           fontsize=10, weight='bold')
    
    # Draw hidden layer (10 nodes)
    hidden_y_positions = np.linspace(-1.5, 1.5, hidden_size)
    hidden_circles = []
    for i in range(hidden_size):
        activation = A1[i, 0]
        intensity = min(1.0, activation * 2)
        circle = Circle((2, hidden_y_positions[i]), 0.1, 
                      color='teal', alpha=0.3 + intensity * 0.7)
        ax.add_patch(circle)
        hidden_circles.append(circle)
        ax.text(2, hidden_y_positions[i], f'{activation:.2f}', 
               fontsize=8, ha='center', va='center', weight='bold')
    
    # Add hidden layer label
    ax.text(2, -1.9, 'Hidden Layer\n(10 nodes)', ha='center', va='center', 
           fontsize=10, weight='bold')
    
    # Draw output layer (10 nodes)
    output_y_positions = np.linspace(-1.5, 1.5, output_size)
    output_circles = []
    for i in range(output_size):
        activation = A2[i, 0]
        intensity = min(1.0, activation * 5)
        color = 'red' if i == np.argmax(A2) else 'lightblue'
        circle = Circle((4, output_y_positions[i]), 0.1, 
                      color=color, alpha=0.3 + intensity * 0.7)
        ax.add_patch(circle)
        output_circles.append(circle)
        ax.text(4, output_y_positions[i], f'{activation:.3f}', 
               fontsize=8, ha='center', va='center', weight='bold')
    
    # Add output layer label
    ax.text(4, -1.9, 'Output Layer\n(10 nodes)', ha='center', va='center', 
           fontsize=10, weight='bold')
    
    # Draw connections (simplified)
    # Input to hidden connections (show a few representative ones)
    for i in range(5):  # Show 5 input nodes
        for j in range(hidden_size):
            ax.plot([0.05, 1.9], [input_y_positions[i*4], hidden_y_positions[j]], 
                   'darkblue', alpha=0.2, linewidth=0.3)
    
    # Hidden to output connections
    for i in range(hidden_size):
        for j in range(output_size):
            ax.plot([2.1, 3.9], [hidden_y_positions[i], output_y_positions[j]], 
                   'teal', alpha=0.3, linewidth=0.5)
    
    # Sample info
    true_label = Y_train[sample_idx]
    predicted_label = np.argmax(A2)
    ax.text(2, 1.8, f'True: {true_label} | Predicted: {predicted_label}', 
           ha='center', va='center', fontsize=12, weight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.title('Simple Neural Network Visualization (784 → 10 → 10)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

def main():
    print("Loading trained model...")
    weights = load_trained_weights()
    if weights is None:
        return
    
    W1, b1, W2, b2 = weights
    X_train, Y_train, _, _ = load_data()
    
    print("Showing simple neural network visualization...")
    visualize_simple_network(W1, b1, W2, b2, X_train, Y_train, sample_idx=0)

if __name__ == "__main__":
    main() 