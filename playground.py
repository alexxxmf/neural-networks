from Layers import Layer_Dense
from Activations import (
  Activation_Sigmoid,
  Activation_Softmax,
  Activation_Tanh,
  Activation_ReLU
)
from Loss import (
  Loss_CategoricalCrossEntropy
)
import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data


nnfs.init()

X, y = spiral_data(samples=100, classes=3)

inputs = np.array([[0.2, 0.1, 0.3], [0.1, 0.3, 0.1]])

dense_layer1 = Layer_Dense(inputs=X, n_neurons=64, weight_init_type='glorot', bias_init_type='zeros')
dense_layer1.forward()
relu1 = Activation_ReLU()
relu1.forward(inputs=dense_layer1.output)

dense_layer2 = Layer_Dense(inputs=relu1.output, n_neurons=3, weight_init_type='glorot', bias_init_type='zeros')
dense_layer2.forward()
softmax1 = Activation_Softmax()
softmax1.forward(dense_layer2.output)

loss_function = Loss_CategoricalCrossEntropy()
result = loss_function.calculate(softmax1.output, y)

conf = np.argmax(softmax1.output, axis=1)

m = conf == y

print(np.sum(m) / len(y))


# X, y = vertical_data(samples=100, classes=2)

# print(X)