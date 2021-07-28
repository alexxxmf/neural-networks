import numpy as np
from math import sqrt


class Layer_Dense():

  def __init__(
    self, *, n_neurons, inputs, weight_init_type=None, bias_init_type=None,
    weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0,
    bias_regularizer_l2=0
  ):

    input_dimension = None
    if len(inputs.shape) == 1:
      # it's a vector
      input_dimension = inputs.shape[0]
    else:
      input_dimension = inputs.shape[1]
    
    params = self.initialize_params(
      weight_init_type=weight_init_type,
      bias_init_type=bias_init_type,
      input_dimension=input_dimension,
      n_neurons=n_neurons
    )
    # We need to have a record of the inputs so it can be used for backprop
    self.inputs = inputs

    self.weights = params['weights']
    self.biases = params['biases']

    self.weight_regularizer_l1 = weight_regularizer_l1
    self.bias_regularizer_l1 = bias_regularizer_l1
    self.weight_regularizer_l2 = weight_regularizer_l2
    self.bias_regularizer_l2 = bias_regularizer_l2

  def forward(self):
    self.output = np.dot(self.inputs, self.weights) + self.biases

  def backward(self, dvalues):
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = dvalues.copy()

    if self.weight_regularizer_l1 > 0:
      dL1 = np.ones_like(self.weights)
      dL1[self.weights < 0] = -1
      self.dweights += self.weight_regularizer_l1 * dL1

    if self.weight_regularizer_l2 > 0:
      self.dweights += 2 * self.weight_regularizer_l2 * \
                        self.weights

    if self.bias_regularizer_l1 > 0:
      dL1 = np.ones_like(self.biases)
      dL1[self.biases < 0] = -1
      self.dbiases += self.bias_regularizer_l1 * dL1

    if self.bias_regularizer_l2 > 0:
      self.dbiases += 2 * self.bias_regularizer_l2 * \
                      self.biases

    self.dinputs = np.dot(dvalues, self.weights.T)

  def initialize_weights(self, *, input_dimension, n_neurons, weight_init_type):
    if weight_init_type == 'he':
      std = sqrt(2.0 / input_dimension)
      return np.random.randn(input_dimension, n_neurons) * std
    elif weight_init_type == 'rand':
      return np.random.rand(input_dimension, n_neurons)
    elif weight_init_type == 'glorot':
      lower, upper = -(1.0 / sqrt(input_dimension)), (1.0 / sqrt(input_dimension))
      return lower + np.random.randn(input_dimension, n_neurons) * (upper - lower)
    elif weight_init_type == 'normalized-glorot':
      lower, upper = -(sqrt(6.0) / sqrt(input_dimension + n_neurons)), (sqrt(6.0) / sqrt(input_dimension + n_neurons))
      return lower + np.random.randn(input_dimension, n_neurons) * (upper - lower)
    else:
      return np.random.rand(input_dimension, n_neurons)

  def initialize_bias(self, *, n_neurons, bias_init_type):
    if bias_init_type == 'zeros':
      return np.zeros((1, n_neurons))
    else: 
      return np.zeros((1, n_neurons))

  def initialize_params(self, *, input_dimension, n_neurons, bias_init_type, weight_init_type):
    return {
      'weights': self.initialize_weights(
        input_dimension=input_dimension,
        n_neurons=n_neurons,
        weight_init_type=weight_init_type
      ),
      'biases': self.initialize_bias(
        n_neurons=n_neurons,
        bias_init_type=bias_init_type
      )
    }


class Layer_Input:

  def forward(self, inputs, training):
    self.output = inputs


class Layer_Dropout():

  def __init__(self):
    pass