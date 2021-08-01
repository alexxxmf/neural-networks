import numpy as np


class Optimizer_AdaGrad:

  def __init__(self, learning_rate=0.1, lr_decay=0., iterations=0, epsilon=1e-7):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.lr_decay = lr_decay
    self.iterations = iterations
    self.epsilon = epsilon

  def pre_update_params(self):
    if self.lr_decay:
      self.current_learning_rate = self.learning_rate / \
        (1 + self.lr_decay * self.iterations)

  def update_params(self, *, layer):
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.bias_cache = np.zeros_like(layer.biases)

    layer.weight_cache += layer.dweights ** 2
    layer.bias_cache += layer.biases ** 2  
    
    weight_updates = - layer.dweights * (self.current_learning_rate / \
      (np.sqrt(layer.weight_cache) + self.epsilon))

    bias_updates = - layer.dbiases * (self.current_learning_rate / \
      (np.sqrt(layer.bias_cache) + self.epsilon))

    layer.weights += weight_updates
    layer.biases += bias_updates

  def post_update_params(self):
    self.iterations += 1