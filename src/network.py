import numpy as np
from layer import Layer

class Network:

    # pass a list like [2,8,8,1] for a 2-8-8-1 structure
    def __init__(self, structure, activation):
        self.layers = []
        self.activation = activation
        for i in range(len(structure)-1):
            n_in = structure[i]
            n_out = structure[i+1]
            self.layers.append(Layer(n_in,n_out,self.activation))

    def forward_pass(self, x):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(x)
            else:
                self.layers[i].forward(self.layers[i-1].a)
        return self.layers[-1].a

    # y is desired output
    def calculate_cost(self, x, y):
        self.cost = 0
        output = self.forward_pass(x)
        for i in range(len(output)):
            self.cost += 0.5 * pow(output[i]- y[i], 2)
        