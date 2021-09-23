import numpy as np


# act functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binaryStep(x):
    return np.heaviside(x, 1)


def tanh(x):
    return np.tanh(x)


def relu(x):
    xA = []
    for i in x:
        if i < 0:
            xA.append(0)
        else:
            xA.append(i)
    return xA


# derivative implementation
def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanhDerivative(x):
    return 1 - tanh(x) ** 2
