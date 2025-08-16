import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import pickle
import os

data = pd.read_csv('train.csv')

# print(data.head())

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
_, m_train = X_train.shape

def init_params():
    W1 = np.random.rand(10, 784) * 0.01
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10) * 0.01
    b2 = np.random.rand(10, 1)
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def leaky_ReLU(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

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

def ReLU_deriv(Z):
    return Z > 0

def leaky_ReLU_deriv(Z, alpha=0.01):
    grad = np.ones_like(Z)
    grad[Z < 0] = alpha
    return grad

def one_hot(Y):
    num_classes = 10  # MNIST has 10 classes (digits 0-9)
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * leaky_ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def update_visualization(W1, b1, W2, b2, X_train, Y_train, iteration, sample_idx=0):
    """
    Update the neural network visualization in real-time
    """
    # Get activations for the sample
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train[:, sample_idx:sample_idx+1])
    
    # Clear the previous plot
    plt.clf()
    
    # Create figure
    fig = plt.gcf()
    fig.set_size_inches(8, 8)  # Square 600x600 equivalent
    ax = plt.gca()
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
    ax.text(2, 1.8, f'Iteration: {iteration} | True: {true_label} | Predicted: {predicted_label}', 
           ha='center', va='center', fontsize=12, weight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.title(f'Neural Network Training Progress (784 → 10 → 10) - Iteration {iteration}', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.pause(0.1)  # Brief pause to show the update

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    
    # Initialize the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))  # Square 600x600 equivalent
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 5 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(accuracy)
            
            # Update visualization every 5 iterations with different samples
            sample_idx = (i // 5) % 100  # Cycle through first 100 samples
            update_visualization(W1, b1, W2, b2, X_train, Y_train, i, sample_idx)
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the final plot open
    return W1, b1, W2, b2

#-------------------------------- SGD --------------------------------
#def stochastic_gradient_descent(X, Y, alpha, epochs, batch_size=64):
#    W1, b1, W2, b2 = init_params()
#    m = X.shape[1]
#    for epoch in range(epochs):
#        indices = np.random.permutation(m)
#        for start in range(0, m, batch_size):
#            end = min(start + batch_size, m)
#            batch_idx = indices[start:end]
#            x_batch = X[:, batch_idx]
#            y_batch = Y[batch_idx]
#            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, x_batch)
#            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, x_batch, y_batch)
#            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
#        if epoch % 10 == 0:
#            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
#            predictions = get_predictions(A2)
#            acc = get_accuracy(predictions, Y)
#            print(f"Epoch: {epoch}, Accuracy: {acc}")
#    return W1, b1, W2, b2
#
#W1, b1, W2, b2 = stochastic_gradient_descent(X_train, Y_train, 0.1, 500)


# Train the neural network
print("Training Neural Network...")
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 500)

# Save trained weights for visualization scripts
import pickle
with open('trained_weights.pkl', 'wb') as f:
    pickle.dump((W1, b1, W2, b2), f)

print("\nTraining Complete!")
print("Trained weights saved to 'trained_weights.pkl'")
print("\nTo visualize the network, run:")
print("- python visualize_network.py")