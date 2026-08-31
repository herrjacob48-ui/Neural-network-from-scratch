# **Neural network engine**

The core focus of this project is the hand built neural network framework contained in `network.py` and `layer.py`, written from scratch in python using only NumPy. This implements forward propagation, back propagation, gradient descent, and customizable network structure all without any external ML libraries.

# 1. **Connected layers**

Each layer stores:
   - a weight matrix, W, shape (n_out, n_in)
   - a bias vector, b, shape (n_out, 1)
   - an activation function, passed at construction
   - forward pass values:
       - z - preactivation
       - a - activation
         
Weights and biases are initialized with noise upon construction.

The layer’s forward function computes:
  - `z = W @ x + b`
  - `a = activation(z)`
    
These values are cached for use during backpropagation.

# 2. Network construction
A network is created with:
`Network([structure], activation, learning_rate)`

Where:

  - structure is a list describing the number of neurons per layer
  - activation is a function object such as `sigmoid`
  - learning rate is a float controlling the gradient descent size

ex:

```python
structure = [2,16,16,3]
activation = sigmoid
rate = 0.01
net = Network(structure, activation, rate)
```
The network internally connects the layers with matrix and vector sizes.
  
