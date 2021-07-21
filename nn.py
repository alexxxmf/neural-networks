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
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases
  
  def backward(self, dvalues):
    # if we have a batch with 4 samples, each sample having 3 dimensions that are fed into 5 neurons
    # input matrix will be a [4 x 3], weights one [3 x 5], being the resulting one, [4 x 5]
    # given dvalues(gradient) has the same dimension as output, dvalues is [4 x 5]
    # if we want to get dweights (having same dim as weights, 3 x 5 ), we need to multiply
    # inputs Transpose by dvalues df/di = w | df/dw = i
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_Softmax():

  def forward(self, inputs):
    self.inputs = inputs
    # np.exp(5000) causes a Runtime error, to prevent this for any set of inputs, we substract the max(i) to
    # from the inputs in a set so we make sure the higher we can get is 0, exp(0) = 1
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
  
  def backward(self, dvalues):
    self.dinputs = np.empty_like(dvalues)

    for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
        single_output = single_output.reshape(-1, 1)

        jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

        self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Activation_ReLU():

  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.maximum(0, inputs)

  def backward(self, dvalues):
    self.dinputs = dvalues.copy()
    self.dinputs[self.inputs <= 0] = 0


class Loss:

  def calculate(self, output, y):
    sample_losses = self.forward(output, y)
    return np.mean(sample_losses)


class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():

  def __init__(self):
      self.activation = Activation_Softmax()
      self.loss = Loss_CategoricalCrossentropy()

  def forward(self, inputs, y_true):
      self.activation.forward(inputs)
      self.output = self.activation.output

      return self.loss.calculate(self.output, y_true)

  def backward(self, dvalues, y_true):

      samples = len(dvalues)

      if len(y_true.shape) == 2:
          y_true = np.argmax(y_true, axis=1)

      self.dinputs = dvalues.copy()
      self.dinputs[range(samples), y_true] -= 1
      self.dinputs = self.dinputs / samples


class Optimizer_SGD : 

  def __init__(self, learning_rate=1., decay=0., momentum=0.): 
    self.learning_rate = learning_rate
    self.current_learning_rate = learning_rate
    self.decay = decay
    self.iterations = 0
    self.momentum = momentum

  def pre_update_params(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * \
        (1. / (1. + self.decay * self.iterations))

  def update_params(self, layer):
    if self.momentum:
      if not hasattr(layer, 'weight_momentums'):
        layer.weight_momentums = np.zeros_like(layer.weights)
        layer.bias_momentums = np.zeros_like(layer.biases)
      
      weight_updates = \
        self.momentum * layer.weight_momentums - \
        self.current_learning_rate * layer.dweights
      layer.weight_momentums = weight_updates

      bias_updates = \
        self.momentum * layer.bias_momentums - \
        self.current_learning_rate * layer.dbiases
      layer.bias_momentums = bias_updates

    else:
      weight_updates = -self.current_learning_rate * \
                        layer.dweights
      bias_updates = -self.current_learning_rate * \
                      layer.dbiases

    layer.weights += weight_updates
    layer.biases += bias_updates
  
  def post_update_params(self):
    self.iterations += 1
