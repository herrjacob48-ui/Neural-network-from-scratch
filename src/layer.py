import numpy as np
class Layer:
    def __init__(self, size):
        self.a = np.array([])
        self.w = []
        self.gradients = []
        
            