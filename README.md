# **Neural network engine**

The core focus of this project is the hand built neural network framework contained in `network.py` and `layer.py`, written from scratch in python using only NumPy. This implements forward propagation, back propagation, gradient descent, and customizable network structure all without any external ML libraries.

# **1. Connected layers**

Each layer stores:
   - a weight matrix, w, shape (n_out, n_in)
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
The network internally connects the layers with matching matrix and vector sizes.

# **3. Forward propagation**
The network calculates predictions with the `forward_pass(x)` function, where x = the input vector. Iterating through each layer, executing their `forward(x)` functions with the activation in a previous layer being the input to the next.

The final layer's activation is returned.

`calculate_cost(y,x)` uses `0.5 * (output - target)^2` as the calculation where x is the prediction and y is the desired result. 

# **4. Back propagation**
The network has two functions to calculate the gradient vectors, `compute_deltas(y)` and `backpropagation(self, x)`.

`compute_deltas(y)` iterates backwards through the network's list of layers and computes the delta vector for each one, using their stored `a` and `z` values. Deltas represent the error signal for each layer.

`delta = (np.transpose(self.layers[i+1].w) @ self.deltas[i+1]) * sigmoid_prime(self.layers[i].z)`

`backpropagation(x)` then iterates backwards once again, calculating the gradient vectors for each layer.

`self.gradients[i] = self.deltas[i] @ np.transpose(self.layers[i-1].a)`

# **5. Gradient descent**
Weights and biases are updated in `gradient_descent()` using:
```python
self.layers[i].w -= self.gradients[i] * self.learning_rate
self.layers[i].b -= self.deltas[i] * self.learning_rate
```

# **6. Training loop**
The entire training loop is contained inside the network's `train(x,y)` function:
```python
      self.forward_pass(x)
      self.calculate_cost(x,y)
      self.compute_deltas(y)
      self.backpropagation(x)
      self.gradient_descent()
   return self.cost
```
Where x is the input vector and y is the desired result.

Together these components make a simple but effective neural network capable of learning from supervised training pairs, `(x,y)` using only NumPy.

# **Limitations and future work**

Because this network only learns from supervised pairs, its limited to strictly imitation learning, which has several limitations:
- no other forms of learning such as reinforcement learning are supported
- only applicable to situations in which the ideal result is already known

Fixed activation and loss functions:
- the network assumes sigmoid activation and mean squared error, which is not ideal for all training scenarios.

No batching or optimization:
- training uses plain gradient descent with an update after each training example, for more complex and nuanced tasks, this would prove very inefficient. 

Future work:
- add reinforcement learning support
- support additional activation functions (ReLU, tanh, softmax)
- enable batch training
- add weight saving/loading
