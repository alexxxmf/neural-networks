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
import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data, sine_data
from Model.Model import Model


nnfs.init()

X, y = sine_data()

inputs = np.array([[0.2, 0.1, 0.3], [0.1, 0.3, 0.1]])

# dense_layer1 = Layer_Dense(inputs=X, n_neurons=64, weight_init_type='glorot', bias_init_type='zeros')
# dense_layer1.forward()
# relu1 = Activation_ReLU()
# relu1.forward(inputs=dense_layer1.output)

# dense_layer2 = Layer_Dense(inputs=relu1.output, n_neurons=3, weight_init_type='glorot', bias_init_type='zeros')
# dense_layer2.forward()
# softmax1 = Activation_Softmax()
# softmax1.forward(dense_layer2.output)

# loss_function = Loss_CategoricalCrossEntropy()
# result = loss_function.calculate(softmax1.output, y)

# conf = np.argmax(softmax1.output, axis=1)

# m = conf == y

# print(np.sum(m) / len(y))


# X, y = vertical_data(samples=100, classes=2)

# print(X)

model = Model()

model.add(Layer_Dense(
  input_dimension=1,
  n_neurons=64,
  weight_init_type='glorot',
  bias_init_type='zeros'
))
model.add(Activation_ReLU())
model.add(Layer_Dense(
  input_dimension=64,
  n_neurons=64,
  weight_init_type='glorot',
  bias_init_type='zeros'
))
model.add(Activation_ReLU())
model.add(Layer_Dense(
  input_dimension=64,
  n_neurons=1,
  weight_init_type='glorot',
  bias_init_type='zeros'
))
model.add(Activation_Linear())

model.set(
  loss=Loss_MeanSquaredError(),
  optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3)
)

model.finalize()

model.train(X, y, epochs=10000, print_every=300)

print(model.layers)
