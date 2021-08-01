import numpy as np
from math import exp

class Activation_ReLU():

  def forward(self, inputs, training=False):
    self.inputs = inputs
    self.output = np.maximum(0, inputs)

  def backward(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0

  def predictions(self, outputs):
    return outputs