import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

from nn import *

nnfs.init()

if __name__ == "__main__":
  X, y = spiral_data(samples=100, classes=2)
  X_test, y_test = spiral_data(samples=100, classes=2)

  y = y.reshape(-1, 1)
  y_test = y_test.reshape(-1, 1)

  model = Model()

  model.add(DenseLayer(2, 64, weight_regularizer_l2=5e-4,
                              bias_regularizer_l2=5e-4))
  model.add(Activation_ReLU())
  model.add(DenseLayer(64, 1))
  model.add(Activation_Sigmoid())

  model.set(
    loss=Loss_BinaryCrossentropy(),
    optimizer=Optimizer_Adam(decay=5e-7),
    accuracy=Accuracy_Categorical()
  )

  model.finalize()

  model.train(X, y, validation_data=(X_test, y_test),epochs=10000, print_every=100)