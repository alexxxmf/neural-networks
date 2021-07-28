import numpy as np


class Loss:

  def calculate(self, output, y):
    sample_losses = self.forward(output, y)

    data_loss = np.mean(sample_losses)

    return data_loss


class Loss_CategoricalCrossEntropy(Loss):

  def forward(self, y_pred, y_true):
    # Preventing log(0) = -inf from happening by clipping values
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    if len(y_true.shape) == 1:
      correct_confidences = y_pred_clipped[
        range(len(y_pred_clipped)), y_true]
    elif len(y_true.shape) == 2:
      correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

    return -np.log(correct_confidences)
  
  def backward(self, dvalues, y_true):
    samples = len(dvalues)
    labels = len(dvalues[0])

    if len(y_true.shape) == 1:
        y_true = np.eye(labels)[y_true]

    self.dinputs = -y_true / dvalues
    self.dinputs = self.dinputs / samples


class Loss_BinaryCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        outputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs

        self.dinputs = self.dinputs / samples


class Loss_MeanSquaredError(Loss):

  def forward(self, y_pred, y_true):
    sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
    return sample_losses

  def backward(self, dvalues, y_true):
    samples = len(dvalues)
    outputs = len(dvalues[0])

    self.dinputs = -2 * (y_true - dvalues) / outputs
    self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError(Loss):

  def forward(self, y_pred, y_true):
    return np.mean(np.abs(y_true - y_pred), axis=-1)

  def backward(self, dvalues, y_true):
    samples = len(dvalues)
    outputs = len(dvalues[0])

    self.dinputs = np.sign(y_true - dvalues) / outputs
    self.dinputs = self.dinputs / samples