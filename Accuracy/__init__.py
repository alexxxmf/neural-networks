import numpy as np


class Accuracy():

  def calculate(self, predictions, y):
    pass


class Accuracy_Categorical(Accuracy):

  def __init__(self, *, binary=False):
    self.binary = binary
  

  def compare(self, predictions, y):
    if not self.binary and len(y.shape) == 2:
      y = np.argmax(y, axis=1)
    return predictions == y


class Accuracy_Regression(Accuracy):
  
  def compare(self):
    pass