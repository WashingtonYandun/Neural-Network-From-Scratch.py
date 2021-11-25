import numpy as np


# act functions
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


# derivative implementation of act functions
def sigmoid_derivative(x):
    """sigmoid derivative dependent value from value, using numpy"""
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_derivative(x):
    """hyperbolic tangent derivative dependent value from value, using numpy"""
    return 1 - tanh(x) * tanh(x)


# cost function
def mse(yt,yf):
    """mean square error cost function"""
    return np.square(yf - yt).mean() 
