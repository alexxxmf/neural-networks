import numpy as np
from math import exp

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