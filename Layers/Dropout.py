import numpy as np


class Layer_Dropout:

  def __init__(self, rate):
    self.rate = 1 - rate

  def forward(self, inputs, training):
    self.inputs = inputs

    if not training:
      self.output = inputs.copy()
      return

    # When we deactivate some neurons, to prevent the weights from growing larger to adapt
    # to some neurons being off, we divide the binary mask by the rate so instead of having
    # a matrix of 0s and 1s we'll have 0s and 1/self.rates
    # a dropout layer should have the effect of a sampling of neuronal subnetworks
    self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
    self.output = inputs * self.binary_mask

  def backward(self, dvalues):
    self.dinputs = dvalues * self.binary_mask