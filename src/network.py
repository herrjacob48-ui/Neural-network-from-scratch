import numpy as np
from layer import Layer

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

class Network:

    # pass a list like [2,8,8,1] for a 2-8-8-1 structure
    def __init__(self, structure, activation, rate):
        self.layers = []
        self.activation = activation
        self.learning_rate = rate
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

    def compute_deltas(self, y):
        self.deltas = [None] * len(self.layers)
        for i in range(len(self.layers)-1,-1,-1):
            if i == len(self.layers)-1:
                delta = (self.layers[i].a - y) * sigmoid_prime(self.layers[i].z)
            else:
                 delta = (np.transpose(self.layers[i+1].w) @ self.deltas[i+1]) * sigmoid_prime(self.layers[i].z)  
            self.deltas[i] = delta

    def backpropagation(self, x):
        self.gradients = [None] * len(self.layers)
        for i in range(len(self.layers)-1,-1,-1):
            if i == 0:
                self.gradients[i] = self.deltas[i] @ np.transpose(x)
            else:
                self.gradients[i] = self.deltas[i] @ np.transpose(self.layers[i-1].a)

    def gradient_descent(self):
        for i in range(len(self.layers)):
            self.layers[i].w -= self.gradients[i] * self.learning_rate
            self.layers[i].b -= self.deltas[i] * self.learning_rate

    def train(self,x,y):
        self.forward_pass(x)
        self.calculate_cost(x,y)
        self.compute_deltas(y)
        self.backpropagation(x)
        self.gradient_descent()
        return self.cost