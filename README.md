# Neural Networks

### Description

This is the repo where I implement a Neural Network from scratch to gain a deeper understaindg of it.
For tricky derivatives/gradients like the softmax one, I rely on given python implementations rather than deriving them myself.

### TO REVIEW AGAIN FOR DEEPER UNDERSTANDING:

- Softmax activation derivative (alone)
- Common Categorical Cross-Entropy loss and Softmax activation derivative
- Performance test. Calculating gradient for cross entropy and softmax separately and then all in one go. (HINT: joint should be around 6 to 8 times faster)

### Interesting posts/questions

- [Why we need bias in neural networks](https://towardsdatascience.com/why-we-need-bias-in-neural-networks-db8f7e07cb98)

- [Can any one explain why dot product is used in neural network and what is the intitutive thought of dot product](https://stats.stackexchange.com/questions/291680/can-any-one-explain-why-dot-product-is-used-in-neural-network-and-what-is-the-in)

- [Why should the initialization of weights and bias be chosen around 0?](https://datascience.stackexchange.com/questions/22093/why-should-the-initialization-of-weights-and-bias-be-chosen-around-0)

- [How should the bias be initialized and regularized?](https://datascience.stackexchange.com/questions/17987/how-should-the-bias-be-initialized-and-regularized)

- [Why must a nonlinear activation function be used in a backpropagation neural network?](https://stackoverflow.com/questions/9782071/why-must-a-nonlinear-activation-function-be-used-in-a-backpropagation-neural-net)

- [Why is step function not used in activation functions in machine learning?](https://stats.stackexchange.com/questions/271701/why-is-step-function-not-used-in-activation-functions-in-machine-learning)

- [ReLU vs GeLU](https://www.reddit.com/r/MachineLearning/comments/eh80jp/d_gelu_better_than_relu/)

- [What are good initial weights in a neural network?](https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network)

- [Regarding the understanding of the mathematics behind AdaGrad and AdaDelta](https://datascience.stackexchange.com/questions/27676/understanding-the-mathematics-of-adagrad-and-adadelta)

- [AdaDelta: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701)

- [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)

- [A question regarding the differences between decay and rho being the former a generic one applied to the LR and the latter a specific one that affects the squared gradients](https://stats.stackexchange.com/questions/351409/difference-between-rho-and-decay-arguments-in-keras-rmsprop)

- [Why can GPU do matrix multiplication faster than CPU?](https://stackoverflow.com/questions/51344018/why-can-gpu-do-matrix-multiplication-faster-than-cpu)

- [Understanding the Efficiency of GPU Algorithms for Matrix-Matrix Multiplication](https://graphics.stanford.edu/papers/gpumatrixmult/gpumatrixmult.pdf)

- [Some interesting notes on optimization](http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf)

- [Full collection of lectures on NN by UofT and prof. Hinton](https://www.cs.toronto.edu/~hinton/coursera_lectures.html)

- [Great notes by J. Brownlee on RMSProp compared to SGD and AdaGrad](https://machinelearningmastery.com/gradient-descent-with-rmsprop-from-scratch/)

- [Introduction to Adam Optimizer by J. Brownlee](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/#:~:text=Adam%20is%20a%20replacement%20optimization,sparse%20gradients%20on%20noisy%20problems.)

- [Paper for Adam Optimizer: "Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980)

- [How to Develop a CNN From Scratch for CIFAR-10 Photo Classification](https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/)

- [How Much Training Data is Required for Machine Learning?](https://machinelearningmastery.com/much-training-data-required-machine-learning/)

### Random notes

Softmax. How to prevent overflow when dealing with big numbershttps://nolanbconaway.github.io/blog/2017/softmax-numpy.html
(Substracting max number from all instances)
Video explanation with graphics
https://www.youtube.com/watch?v=ytbYRIN0N4g

Categorical cross-entropy, one-hot vector target
Summing really small numbers to prevent ending up doing np.exp(0) which is minus infinity

Why ….. softmax? Really good explanation on approximations
https://www.quora.com/Why-is-frac-e-u_i-sum_-j-e-u_j-called-softmax
Softmax gives us the differentiableapproximation of a non-differentiable
function max. Why is that important? For optimizing models, including machine learning models, it is required that functions describing the model be differentiable. So if we want to optimize a model which uses the max function then we can do that by replacing the max with softmax.

what is vanishing gradient problem?

Why ReLU over sigmoid?
https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks

Solve gaggle with NN from scratch
https://www.kaggle.com/manzoormahmood/mnist-neural-network-from-scratch

Covid neural network project
https://www.youtube.com/watch?v=nHQDDAAzIsI

Understanding the impact of learning rate decay:
https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/

https://cs231n.github.io/neural-networks-case-study/
https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/

On learning rate decay
https://opennmt.net/OpenNMT/training/decay/
https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
https://arxiv.org/pdf/1905.00094.pdf

### Tips

We ​should **NOT**​ check different hyperparameters using the test dataset; if we do that, we’re going to be manually optimizing the model to the test dataset, biasing it towards overfitting these data, and these data are supposed to be used only to perform the last check if the model trains and generalizes well. **Hyperparameter tuning using the test dataset is a mistake**. Hyperparameter tuning can be performed using yet another dataset called ​validation data.

The first is to temporarily split the training data into a smaller training dataset and validation dataset for hyperparameter tuning. Afterward, with the final hyperparameter set, train the model on all the training data. We allow ourselves to do that as we tune the model to the part of training data that we put aside as validation data. Keep in mind that we still have a test dataset to check the model’s performance after training.

Neural networks usually perform best on data consisting of **numbers in a range of 0 to 1 or -1 to 1, with the latter being preferable**. Centering data on the value of 0 can help with model training as it attenuates weight biasing in some direction. Models can work fine with data in the range of 0 to 1 in most cases, but sometimes we’re going to need to rescale them to a range of -1 to 1 to get training to behave or achieve better results.

There are many terms related to data ​preprocessing​: standardization, scaling, variance scaling, mean removal (as mentioned above), non-linear transformations, scaling to outliers, etc., but they are out of the scope of this book. We’re only going to scale data to a range by simply dividing all of the numbers by the maximum of their absolute values. For the example of an image that consists of numbers in the range between ​0​ and ​255​, we divide the whole dataset by ​255​ and return data in the range from ​0​ to ​1.​ We can also subtract ​127.5​ (to get a range from ​-127.5​ to 127.5)​ and divide by 127.5, returning data in the range from -1 to 1.
It is usually fine to scale datasets that consist of larger numbers than the training data using a scaler prepared on the training data. **If the resulting numbers are slightly outside of the ​-1​ to ​1​ range, it does not affect validation or testing negatively**, since we do not train on these data.
