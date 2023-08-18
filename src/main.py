import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Activation functions
def sigmoid(x):
    """Compute the sigmoid function for the given input x using numpy."""
    return 1 / (1 + np.exp(-x))

def binary_step(x):
    """Apply binary step activation: 0 if x is negative, 1 otherwise, using numpy."""
    return np.heaviside(x, 1)

def tanh(x):
    """Compute the hyperbolic tangent function for the given input x using numpy."""
    return np.tanh(x)

def relu(x):
    """Apply ReLU activation: return x if x is non-negative, else return 0, using numpy."""
    return np.maximum(0, x)

def relu_classic(x):
    """Apply classic ReLU activation: return x if x is non-negative, else return 0, without using libraries."""
    return [max(0, val) for val in x]

def sigmoid_derivative(x):
    """Compute the derivative of the sigmoid function for the given input x using numpy."""
    return sigmoid(x) * (1 - sigmoid(x))

def tanh_derivative(x):
    """Compute the derivative of the hyperbolic tangent function for the given input x using numpy."""
    return 1 - np.tanh(x) ** 2

def mse(y_true, y_pred):
    """Compute the Mean Squared Error between the true values y_true and predicted values y_pred."""
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_true, y_pred):
    """Compute the derivative of Mean Squared Error loss function."""
    return 2 * (y_pred - y_true) / len(y_true)

# Neural Layer class
class NeuralLayer:
    def __init__(self, num_inputs, num_neurons, activation_function):
        """Initialize a neural layer with given number of inputs, neurons, and activation function."""
        self.activation_function = activation_function
        self.bias = np.random.rand(1, num_neurons) * 2 - 1
        self.weights = np.random.rand(num_inputs, num_neurons) * 2 - 1

# Create the dataset
num_samples = 1000
num_features = 2  # 2D x and y
X, y = make_circles(n_samples=num_samples, factor=0.6, noise=0.07)

# Visualize the dataset
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="#1d3557", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="#e63946", label="Class 1")
plt.axis("equal")
plt.legend()
plt.show()

# Create the model based on the topology
topology = [num_features, 4, 8, 1]

def create_nn(topology, activation_function):
    """Create a neural network based on the given topology and activation function."""
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(NeuralLayer(topology[l], topology[l + 1], activation_function))
    return nn

# Train the model
def train(nn, X, y, epochs, learning_rate, cost_function, cost_function_derivative):
    """Train the neural network using backpropagation and gradient descent."""
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
                deltas.insert(0, cost_function_derivative(y, nn[l].output) * nn[l].activation_function(nn[l].output))
            else:
                deltas.insert(0, np.dot(deltas[0], nn[l + 1].weights.T) * nn[l].activation_function(nn[l].output))
        
        # Gradient descent
        for l in range(len(nn)):
            layer = nn[l]
            layer.weights += -learning_rate * np.dot(nn[l - 1].output.T, deltas[l])
            layer.bias += -learning_rate * deltas[l].sum(axis=0, keepdims=True)
        
        # Calculate mean square error
        mse_value = cost_function(y, nn[-1].output)
        mses.append(mse_value)
        if i % 1000 == 0:
            print(f"Epoch: {i}, MSE: {mse_value}")
    return mses
