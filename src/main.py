import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

class NeuralLayer:
    def __init__(self, num_inputs, num_neurons, activation_function):
        self.activation_function = activation_function
        self.bias = np.random.rand(1, num_neurons) * 2 - 1
        self.weights = np.random.rand(num_inputs, num_neurons) * 2 - 1

# Create the dataset
num_samples = 1000
num_features = 2

X, y = make_circles(n_samples=num_samples, factor=0.6, noise=0.07)

plt.scatter(X[y == 0, 0], X[y == 0, 1], c="#1d3557")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="#e63946") 
plt.axis("equal")
plt.show()