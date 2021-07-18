import pandas
import numpy as np


class DenseLayer():
  def __init__(self, n_inputs, n_neurons):
    # Some note on weight and bias initialisation
    # https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
    # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases


class Activation_Softmax():

  def forward(self, inputs):
    self.inputs = inputs
    # np.exp(5000) causes a Runtime error, to prevent this for any set of inputs, we substract the max(i) to
    # from the inputs in a set so we make sure the higher we can get is 0, exp(0) = 1
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
  
  def backward():
    pass


class Activation_ReLU():

  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.maximum(0, inputs)

  def backward(self, dValues):
    self.dInputs = dValues.copy()
    self.dInputs[self.dInputs <= 0] = 0


class Loss():

  def calculate(self, output, y):
    sample_losses = self.forward(output, y)
    return np.mean(sample_losses)


class Loss_CategoricalCrossEntropy(Loss):

  def forward(self, y_pred, y_true):
    samples = len(y_pred)
    # Some notes regarding clipping values
    # https://stackoverflow.com/questions/65131391/what-exactly-is-kerass-categoricalcrossentropy-doing
    # essentially we want to prevent a Runtime Issue when doing a np.log on 0
    y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
    
    if len(y_true.shape) == 1:
      correct_confidences = y_pred_clipped[range(samples), y_true]

    elif len(y_true.shape) == 2:
      correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

    return -np.log(correct_confidences)