import numpy as np
from math import exp


class Activation_Linear():

  def forward(self, inputs):
    # No need for now to record within the object the received inputs but there might
    # be something to do with it in the future
    self.inputs = inputs
    self.output = inputs

  def backward(self, dvalues):
   self.dinputs = dvalues.copy()
