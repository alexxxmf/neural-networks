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
