'''
neural network could be a class but i wanted to use an array instead
anyways i want to separate this from main
'''

def create_neural_network(structure, neuralNetworkFunction):
    neuralNetwork = []
    for l, layer in enumerate(structure[:-1]):
        neuralNetwork.append(Layer(structure[l], structure[l + 1], neuralNetworkFunction))
    return neuralNetwork


def train():
    pass