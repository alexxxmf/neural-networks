import numpy as np

class Optimizer_SGD:

  def __init__(self, learning_rate=0.1, lr_decay=0., iterations=0, momentum=0.):
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.lr_decay = lr_decay
    self.iterations = iterations
    self.momentum = momentum

  def pre_update_params(self):
    if self.lr_decay:
      self.current_learning_rate = self.learning_rate / \
        (1 + self.lr_decay * self.iterations)

  def update_params(self, *, layer):
    if self.momentum:
      if not hasattr(layer, 'weight_momentums'):
        layer.weight_momentums = np.zeros_like(layer.weights)
        layer.bias_momentums = np.zeros_like(layer.biases)
      
      weight_updates = \
        self.momentum * layer.weight_momentums - \
        self.current_learning_rate * layer.dweights
      layer.weight_momentums += weight_updates

      bias_updates = \
        self.momentum * layer.bias_momentums - \
        self.current_learning_rate * layer.dbiases
      layer.bias_momentums += bias_updates

    else:
      weights_update = - self.current_learning_rate * layer.dweights
      bias_update = - self.current_learning_rate * layer.dbiases

    layer.weights += weights_update
    layer.biases += bias_update

  def post_update_params(self):
    self.iterations += 1