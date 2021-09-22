class Layer():
    def __init__(self, connections, neurons, activationFunction):
        self.activationFunction = activationFunction
        self.w = np.random.rand(connections, neurons) * 2 - 1
        self.b = np.random.rand(1, neurons) * 2 - 1