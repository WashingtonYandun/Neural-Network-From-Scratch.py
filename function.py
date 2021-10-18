import numpy as np


# act functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary_step(x):
    return np.heaviside(x, 1)


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


# def relu_classic(x):
#     xA = []
#     for i in x:
#         if i < 0:
#             xA.append(0)
#         else:
#             xA.append(i)
#     return xA


# derivative implementation
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_derivative(x):
    return 1 - tanh(x) * tanh(x)
