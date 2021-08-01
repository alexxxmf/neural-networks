import numpy as np
from math import exp


class Activation_Softmax():

  def forward(self, inputs, training=False):
    # exp_values = np.exp(inputs)
    # self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    # This expression above it could be the regular way to solve the softmax but we need to be
    # mindful about the np.exp. If it s something like 5000 it could produce an overflow so
    # what we end up doing is multipliying the softmax function for (e ^ max(X)) / (e ^ max(X))
    # (e ^ max(X)) / (e ^ max(X)) = 1 so we are not changing the output but we are avoiding the overflow
    # because e^Xi / e ^ max(X) = e^(Xi - max(X)) 
    # (Xi - max(X) at max it could be 0 and in any other case would be a negative number
    self.inputs = inputs
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

  def backward(self, dvalues):
    self.dinputs = np.empty_like(dvalues)

    for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
        single_output = single_output.reshape(-1, 1)

        jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

        self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

  def predictions(self, outputs):
    return np.argmax(outputs, axis=1)

