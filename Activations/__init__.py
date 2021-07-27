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
    pass

  def backward(self, dvalues):
    pass


class Activation_Linear():

  def forward(self, inputs):
    # No need for now to record within the object the received inputs but there might
    # be something to do with it in the future
    self.inputs = inputs
    self.output = inputs

  def backward(self, dvalues):
   self.dinputs = dvalues.copy()


class Activation_Sigmoid():

  def forward(self, inputs):
    # this could potentially cause an overflow because of the exp explosion
    # normally this is the formula for the sigmoid 1 / (1 + np.exp(-inputs))
    # for an exp big enough we could run into an overflow
    self.inputs = inputs

    # this is a trick to apply a function element-wise for a np array
    # this is a normalized sigmoid to prevent problems derive from huge exponentials
    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    self.output = np.vectorize(
      lambda x: 1 / (1 + exp(-x)) if x<0 else exp(x) / (1 + exp(x))
    )(inputs)

  def backward(self, dvalues):
    self.dvalues = dvalues
    self.dinputs = 1 / (1 + np.exp(-dvalues)) * (1 - 1 / (1 + np.exp(-dvalues)))