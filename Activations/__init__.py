import numpy as np

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

  def __init__(self):
    pass

  def forward(self):
    pass

  def backward(self):
    pass