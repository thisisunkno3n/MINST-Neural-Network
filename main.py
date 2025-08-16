import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 500)


# def stochastic_gradient_descent(X, Y, alpha, epochs, batch_size=64):
#     W1, b1, W2, b2 = init_params()
#     m = X.shape[1]
#     for epoch in range(epochs):
#         indices = np.random.permutation(m)
#         for start in range(0, m, batch_size):
#             end = min(start + batch_size, m)
#             batch_idx = indices[start:end]
#             x_batch = X[:, batch_idx]
#             y_batch = Y[batch_idx]
#             Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, x_batch)
#             dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, x_batch, y_batch)
#             W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
#         if epoch % 10 == 0:
#             Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
#             predictions = get_predictions(A2)
#             acc = get_accuracy(predictions, Y)
#             print(f"Epoch: {epoch}, Accuracy: {acc}")
#     return W1, b1, W2, b2

# W1, b1, W2, b2 = stochastic_gradient_descent(X_train, Y_train, 0.1, 500)