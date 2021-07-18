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
  def __init__():
    pass
  def forward():
    pass


class Loss_CrossEntropy():
  def __init__():
    pass
  def forward():
    pass