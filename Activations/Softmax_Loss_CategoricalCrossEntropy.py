import numpy as np

class Activation_Softmax_Loss_CategoricalCrossentropy():
    # This is just used for the backpropagation because calculating the backward step for
    # softmax and cat cross entropy is computationally way more expensive if done separately
    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples