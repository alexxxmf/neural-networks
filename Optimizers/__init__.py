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
      pass
    else:
      weights_update = - self.current_learning_rate * layer.dweights
      biases_update = - self.current_learning_rate * layer.dbiases

    layer.weights += weights_update
    layer.biases += biases_update

  def post_update_params(self):
    pass