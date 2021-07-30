import numpy as np
from Loss import Loss


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