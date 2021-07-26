import numpy as np
import nnfs
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt

from nn import *

nnfs.init()

if __name__ == "__main__":
  X, y = sine_data()

  model = Model()

  model.add(DenseLayer(1, 64))
  model.add(Activation_ReLU())
  model.add(DenseLayer(64, 64))
  model.add(Activation_ReLU())
  model.add(DenseLayer(64, 1))
  model.add(Activation_Linear())

  model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
    accuracy=Accuracy_Regression()
  )

  model.finalize()

  model.train(X, y, epochs=10000, print_every=100)
