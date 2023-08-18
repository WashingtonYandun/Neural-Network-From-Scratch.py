# Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Activation functions
def sigmoid(x):
    """sgmoid dependent value from value, using numpy"""
    return 1 / (1 + np.exp(-x))


def binary_step(x):
    """0 if value is negative, 1 if it is positive or 0, using numpy"""
    return np.heaviside(x, 1)


def tanh(x):
    """hyperbolic tangent dependent value from value, using numpy"""
    return np.tanh(x)


def relu(x):
    """0 if the value is negative, return x if its not, using numpy"""
    return np.maximum(0, x)


def relu_classic(x):
    """0 if the value is negative, return x if its not, no libraries"""
    xA = []
    for i in x:
        if i < 0:
            xA.append(0)
        else:
            xA.append(i)
    return xA


def sigmoid_derivative(x):
    """sigmoid derivative dependent value from value, using numpy"""
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_derivative(x):
    """hyperbolic tangent derivative dependent value from value, using numpy"""
    return 1 - tanh(x) * tanh(x)


def mse(yt, yf):
    """mean square error cost function"""
    return np.square(yf - yt).mean() 


def mse_derivative(yt, yf):
    """mean square error derivative cost function"""
    return (yf - yt) / yt.size

# Neural Layer class
class NeuralLayer:
    def __init__(self, num_inputs, num_neurons, activation_function):
        self.activation_function = activation_function
        self.bias = np.random.rand(1, num_neurons) * 2 - 1
        self.weights = np.random.rand(num_inputs, num_neurons) * 2 - 1


# Create the dataset
num_samples = 1000
num_features = 2 # 2D x and y

X, y = make_circles(n_samples=num_samples, factor=0.6, noise=0.07)

plt.scatter(X[y == 0, 0], X[y == 0, 1], c="#1d3557")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="#e63946") 
plt.axis("equal")
plt.show()

# Create the model based on the topology
topology = [num_features, 4, 8, 1]

def create_nn(topology, activation_function):
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(NeuralLayer(topology[l], topology[l+1], activation_function))
    return nn


# Train the model
def train(nn, X, y, epochs, learning_rate, cost_function, cost_function_derivative):
    mses = []
    for i in range(epochs):
        # Forward pass
        nn[0].output = nn[0].activation_function(np.dot(X, nn[0].weights) + nn[0].bias)
        for l, layer in enumerate(nn[1:]):
            layer.output = layer.activation_function(np.dot(nn[l].output, layer.weights) + layer.bias)

        # Backward pass
        deltas = []
        for l in reversed(range(1, len(nn))):
            if l == len(nn) - 1:
                # Calculate delta last layer
                deltas.insert(0, cost_function_derivative(y, nn[l].output) * nn[l].activation_function(nn[l].output))
            else:
                # Calculate delta hidden layers
                deltas.insert(0, np.dot(deltas[0], nn[l+1].weights.T) * nn[l].activation_function(nn[l].output))
                
        # Gradient descent
        for l in range(len(nn)):
            layer = nn[l]
            layer.weights += -learning_rate * np.dot(nn[l-1].output.T, deltas[l])
            layer.bias += -learning_rate * deltas[l].sum(axis=0, keepdims=True)
        # Calculate mean square error
        mse = cost_function(y, nn[-1].output)
        mses.append(mse)
        if i % 1000 == 0:
            print(f"Epoch: {i}, MSE: {mse}")
    return mses