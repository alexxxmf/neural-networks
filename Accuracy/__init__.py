import numpy as np


class Accuracy():

  def calculate(self, predictions, y):
    # comparisons is an array where every position corresponds with one sample
    # and has as value a boolean saying if the prediction is right or not
    comparisons = self.compare(predictions, y)

    self.accumulated_sum += np.sum(comparisons)
    self.accumulated_count += len(comparisons)

    # np.mean, np.sum, etc do some coercion making True = 1 False = 0 so when doing this
    # some stuff will be done for us
    return np.mean(comparisons)
  
  def calculate_accumulated(self):
    accuracy = self.accumulated_sum / self.accumulated_count

    return accuracy

  def new_pass(self):
    self.accumulated_sum = 0
    self.accumulated_count = 0


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