import numpy as np
from math import exp

class Activation_ReLU():

  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.maximum(0, inputs)

  def backward(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax():

  def forward(self, inputs):
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


class Activation_Linear():

  def forward(self, inputs):
    # No need for now to record within the object the received inputs but there might
    # be something to do with it in the future
    self.inputs = inputs
    self.output = inputs

  def backward(self, dvalues):
   self.dinputs = dvalues.copy()


class Activation_Sigmoid():

  def normalized_sigmoid(self, inputs):
    # this could potentially cause an overflow because of the exp explosion
    # normally this is the formula for the sigmoid 1 / (1 + np.exp(-inputs))
    # for an exp big enough we could run into an overflow
    # this is a trick to apply a function element-wise for a np array
    # this is a normalized sigmoid to prevent problems derive from huge exponentials
    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    return np.vectorize(
      lambda x: 1 / (1 + exp(-x)) if x<0 else exp(x) / (1 + exp(x))
    )(inputs)

  def forward(self, inputs):
    self.inputs = inputs
    self.output = self.normalized_sigmoid(inputs)

  def backward(self, dvalues):
    self.dvalues = dvalues
    self.dinputs = self.normalized_sigmoid(dvalues) * (1 - self.normalized_sigmoid(dvalues))


class Activation_Tanh():

  def normalized_sigmoid(self, inputs):
    # this could potentially cause an overflow because of the exp explosion
    # normally this is the formula for the sigmoid 1 / (1 + np.exp(-inputs))
    # for an exp big enough we could run into an overflow
    # this is a trick to apply a function element-wise for a np array
    # this is a normalized sigmoid to prevent problems derive from huge exponentials
    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    return np.vectorize(
      lambda x: 1 / (1 + exp(-x)) if x<0 else exp(x) / (1 + exp(x))
    )(inputs)

  def forward(self, inputs):
    self.inputs = inputs
    self.output = 2 * self.normalized_sigmoid(2 * inputs) - 1

  def backward(self, dvalues):
    self.dvalues = dvalues
    self.dinputs = 1 - (2 * self.normalized_sigmoid(2 * dvalues) - 1) ** 2