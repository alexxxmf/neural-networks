from Loss.BinaryCrossEntropy import Loss_BinaryCrossEntropy
from nn import Accuracy_Categorical, Activation_Sigmoid, Activation_Softmax
import Loss
import Model
from Layers.Dense import Layer_Dense
from Activations.ReLU import (
  Activation_ReLU
)
from Activations.Linear import Activation_Linear
from Loss.CategoricalCrossEntropy import (
  Loss_CategoricalCrossEntropy,
)
from Loss.MeanSquaredError import (
  Loss_MeanSquaredError,
)
from Optimizers.Adam import Optimizer_Adam
from Optimizers.AdaGrad import Optimizer_AdaGrad
from Accuracy.Regression import Accuracy_Regression
import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data, sine_data
from Model.Model import Model
from Layers.Dropout import Layer_Dropout


nnfs.init()

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Model()

model.add(Layer_Dense(
  input_dimension=2,
  n_neurons=512,
  weight_regularizer_l2=5e-4,
  bias_regularizer_l2=5e-4
))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))

model.add(Activation_Softmax())

model.set(
  loss=Loss_CategoricalCrossEntropy(),
  optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
  accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), 
            epochs=10000, print_every=100)

