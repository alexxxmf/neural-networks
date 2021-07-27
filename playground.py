from Layers import Layer_Dense
import numpy as np

inputs = np.array([[0.2, 0.1, 0.3], [0.1, 0.3, 0.1]])

dense_layer1 = Layer_Dense(inputs=inputs, n_neurons=3, weight_init_type='he', bias_init_type='zeros')

dense_layer1.forward()

print(dense_layer1.output.shape)

print('=============================')

dense_layer2 = Layer_Dense(inputs=dense_layer1.output, n_neurons=2, weight_init_type='glorot', bias_init_type='zeros')

dense_layer2.forward()

print(dense_layer2.output.shape)


dense_layerA = Layer_Dense(
  inputs=inputs, n_neurons=3, weight_init_type='he', bias_init_type='zeros'
)

dense_layerB = Layer_Dense(
  inputs=inputs, n_neurons=3, weight_init_type='glorot', bias_init_type='zeros'
)

dense_layerC = Layer_Dense(
  inputs=inputs, n_neurons=3, weight_init_type='normalized-glorot', bias_init_type='zeros'
)

print(dense_layerA.weights)
print(dense_layerB.weights)
print(dense_layerC.weights)