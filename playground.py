from Layers import Layer_Dense
import numpy as np

dense_layer1 = Layer_Dense(3, 12)

print(dense_layer1)

input = np.array([[0.2, 0.1, 0.3], [0.1, 0.3, 0.1]])

dense_layer1.forward(input)

print(dense_layer1.output)

dense_layer1.backward(dense_layer1.output)

print(dense_layer1.dinputs)