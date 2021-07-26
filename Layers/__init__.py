import numpy as np

"""
Naming convention: Object type underscore object subtype
"""

class Layer_Dense():

  def __init__(self, input_dimension, n_neurons):
    self.weights = np.random.rand(input_dimension, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  def forward(self, inputs):
    # We need to have a record of the inputs so it can be used for backprop
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases

  def backward(self, dvalues):
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = dvalues.copy()
    self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Dropout():

  def __init__(self):
    pass