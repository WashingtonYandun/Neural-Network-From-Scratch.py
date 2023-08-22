import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


# Activation functions
def sigmoid(x, derivative=False):
    """Compute the sigmoid function for the given input x using numpy."""
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


def binary_step(x, derivative=False):
    """Apply binary step activation: 0 if x is negative, 1 otherwise, using numpy."""
    if derivative:
        return 0
    return np.heaviside(x, 1)


def tanh(x, derivative=False):
    """Compute the hyperbolic tangent function for the given input x using numpy."""
    if derivative:
        return 1 - np.tanh(x) ** 2
    return np.tanh(x)


def relu(x, derivative=False):
    """Apply ReLU activation: return x if x is non-negative, else return 0, using numpy."""
    if derivative:
        return np.heaviside(x, 1)
    return np.maximum(0, x)


def mse(y_true, y_pred, derivative=False):
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


# Create the neural network
def create_nn(topology, activation_function):
    """Create a neural network based on the given topology and activation function."""
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(NeuralLayer(
            topology[l], topology[l + 1], activation_function))
    return nn


# Train the model
def train(nn, X, Y, cost_function, learning_rate, train=True):
    # Forward pass
    out = [(None, X)]
    for l, layer in enumerate(nn):
        z = out[-1][1] @ nn[l].weights + nn[l].bias
        a = nn[l].activation_function(z)
        out.append((z, a))

    if train:
        # Backward pass
        deltas = []
        for l in reversed(range(0, len(nn))):
            z = out[l + 1][0]
            a = out[l + 1][1]

            if l == len(nn) - 1:
                delta = cost_function(y_true=Y, y_pred=a) * \
                    nn[l].activation_function(z, derivative=True)
            else:
                delta = deltas[0] @ _w.T * \
                    nn[l].activation_function(z, derivative=True)

            deltas.insert(0, delta)

            _w = nn[l].weights

            # Gradient descent
            nn[l].bias -= np.mean(deltas[0], axis=0,
                                  keepdims=True) * learning_rate
            nn[l].weights -= out[l][1].T @ deltas[0] * learning_rate

    return out[-1][1]


# Create the dataset
num_samples = 1000
num_features = 2  # 2D x and y
X, y = make_circles(n_samples=num_samples, factor=0.6, noise=0.07)


# Visualize the dataset
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="#1d3557", label="Class A")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="#e63946", label="Class B")
plt.axis("equal")
plt.legend()
plt.show()


# Create the model based on the topology
topology = [num_features, 4, 4, 1]


# Create the neural network
nn = create_nn(topology, sigmoid)

# Train the model
epochs = 1000
learning_rate = 0.01
loss = []
for i in range(epochs):
    y_pred = train(nn, X, y, mse, learning_rate, train=True)
    loss.append(mse(y, y_pred))
    if i % 50 == 0:
        print(f"Epoch: {i}, Loss: {mse(y, y_pred)}")


# Visualize the loss
plt.plot(range(epochs), loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
