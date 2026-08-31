import numpy as np
class Layer:
    # activation is a function object ex: reLu or sigmoid.
    def __init__(self, n_in,n_out, activation=None):
        self.w = np.random.randn(n_out, n_in)
        self.b = np.random.randn(n_out, 1)
        self.activation = activation

    # x = a(L-1) vector
    def forward(self, x):
        self.z = self.w @ x + self.b
        self.a = self.activation(self.z)

        
            