import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from nn import *

nnfs.init()

if __name__ == "__main__":
  X, y = spiral_data(samples=100, classes=3)

  dense1 = DenseLayer(2, 64)

  activation1 = Activation_ReLU()

  dense2 = DenseLayer(64, 3)

  loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

  # optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)
  # optimizer = Optimizer_AdaGrad(decay=1e-3, epsilon=1e-6)
  # optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)
  optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)

  for epoch in range(10001):
    # ======= FORWARD PASS =======
    dense1.forward(X)

    activation1.forward(dense1.output)

    dense2.forward(activation1.output)

    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
      y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
      print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f}, ' +
            f'lr: {optimizer.current_learning_rate}')
    
    # ======= BACKWARD PASS =======

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()