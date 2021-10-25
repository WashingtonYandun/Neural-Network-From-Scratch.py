import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from sklearn.datasets import make_circles
from function import *
from layer import *

# make datasets
samples = 500
features = 2  # bi dimensional

# plot the problem
X, Y = make_circles(n_samples=samples, factor=0.5, noise=0.06)
print(Y)
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c='#6a040f')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c='#0a9396')
plt.axis("equal")
plt.show()


# def a topology
networkStructure = [features, 2, 4, 8, 4, 2, 1]


def create_neural_network(structure, neuralNetworkFunction):
    neuralNetwork = []
    for l, layer in enumerate(structure[:-1]):
        neuralNetwork.append(Layer(structure[l], structure[l + 1], neuralNetworkFunction))
    return neuralNetwork



