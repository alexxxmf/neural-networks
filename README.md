# Neural Networks

### Description

This is the repo where I implement a Neural Network from scratch to gain a deeper understaindg of it.
For tricky derivatives/gradients like the softmax one, I rely on given python implementations rather than deriving them myself.

### TO REVIEW AGAIN FOR DEEPER UNDERSTANDING:

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

- [A closer look at memorization in Deep Networks](https://arxiv.org/pdf/1706.05394.pdf)

- [On the geometry of generalization and memorization in deep neural networks](https://openreview.net/pdf?id=V8jrrnwGbuc)

- [On dropout for regularizing NN](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

- [Original paper that first discussed application of dropout layers: Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/pdf/1207.0580.pdf)

- [Much more recent paper on the same topic regarding dropout: Analysis on the Dropout Effect in Convolutional Neural Networks](http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf)

- [Some notes on cross entropy loss both categorical and binary](https://gombru.github.io/2018/05/23/cross_entropy_loss/)

- [More notes on sigmoid and softmax and why/when use one or the other](https://stats.stackexchange.com/questions/233658/softmax-vs-sigmoid-function-in-logistic-classifier/#answer-410112)

- [Regarding Glorot initialization](https://visualstudiomagazine.com/articles/2019/09/05/neural-network-glorot.aspx)

- [Glorot vs He initialization](https://datascience.stackexchange.com/questions/13061/when-to-use-he-or-glorot-normal-initialization-over-uniform-init-and-what-are)

- [Post with a simple plain numpy glorot init example](https://stackoverflow.com/questions/62249084/what-is-the-numpy-equivalent-of-tensorflow-xavier-initializer-for-cnn)

- [Post with some explanation around He weight init method](https://pylessons.com/Deep-neural-networks-part6/)

- [Interesting question regarding ReLU and linearity/non-linearity](https://datascience.stackexchange.com/questions/26475/why-is-relu-used-as-an-activation-function)

- [Interesting explanation regarding initialization and breaking simmetry](https://stats.stackexchange.com/questions/45087/why-doesnt-backpropagation-work-when-you-initialize-the-weights-the-same-value)

- [Original paper with the Glorot init method: Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

- [Post on why Tanh is preferably than sigmoid](https://stats.stackexchange.com/questions/330559/why-is-tanh-almost-always-better-than-sigmoid-as-an-activation-function)

- [Optimizers Obverview with referenced paper included](https://ruder.io/optimizing-gradient-descent/index.html#stochasticgradientdescent)

- [Biologically inspired activation functions](https://arxiv.org/pdf/1804.11237.pdf)

- [Some notebooks with experiments on BRU activation (bionodal root unit)](https://github.com/marek-kan/Bionodal-root-units)

- [Regarding extending pytorch activation functions for state of the art stuff](https://www.kaggle.com/aleksandradeis/extending-pytorch-with-custom-activation-functions)

- [Interesting plain explanation why Stochastic Gradient Descent](https://datascience.stackexchange.com/questions/36450/what-is-the-difference-between-gradient-descent-and-stochastic-gradient-descent)

- [Example of a 2D convolution](http://www.songho.ca/dsp/convolution/convolution2d_example.html)

- [Pytorch autograd system explained](https://www.youtube.com/watch?v=MswxJw-8PvE)

- [Numpy-only CNN implementation](https://becominghuman.ai/only-numpy-implementing-convolutional-neural-network-using-numpy-deriving-forward-feed-and-back-458a5250d6e4)

- [Sparsemax: from paper to code implementation](https://medium.com/deeplearningmadeeasy/sparsemax-from-paper-to-code-351e9b26647b)

- [Interesting paper, beginner friendly. CNN](https://www.reddit.com/r/learnmachinelearning/comments/llstam/i_found_a_paper_on_neural_style_transfer_and_i/)

- [Nice post with a really good recommendation on implementing a list of 26 papers](https://www.reddit.com/r/MachineLearning/comments/8vmuet/d_what_deep_learning_papers_should_i_implement_to/)

- [Video on how to implement dropblock paper](https://www.youtube.com/watch?v=GcvGxXePI2g&list=PLTl9hO2Oobd8UboKp8CyxomVqXyOfOjXe&index=2)

- [Short and concise mention to H&W cat experiment and how they discovered how the brain works, tighly related with how CNN work](https://www.historyofinformation.com/detail.php?id=4261)

- [More on Hubel and Wiesel experiment and how it relates to CNN studies](https://www.esantus.com/blog/2019/1/31/convolutional-neural-networks-a-quick-guide-for-newbies)

- [Guide to receptive field arithmetic for Convolutional Neural Network](https://blog.mlreview.com/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)

- [Making Pixel Art with GAN arhcitectures](https://inikolaeva.medium.com/make-pixel-art-in-seconds-with-machine-learning-e1b1974ba572)

- [Neuroevolution and how to train digital squids how to swim](https://jobtalle.com/neuroevolution_in_squids.html?utm_campaign=Dynamically%20Typed&utm_medium=email&utm_source=Revue%20newsletter)

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

Large weights might indicate that a neuron is attempting to memorize a data element; generally, it is believed that it would be better to have many neurons contributing to a model’s output, rather than a select few

Regarding the dropout method and how to compensate the de-activated neurons stake in the operation:

```
    example_output = np.array([0.27, -1.03, 0.67, 0.99, 0.05,
    -0.37, -2.01, 1.13, -0.07, 0.73])

    np.sum(example_output)

    sums = []

    for i in range(10000):
        example_output2 = example_output * np.random.binomial(1, 1-dropout_rate, example_output.shape)/(1-dropout_rate)
        sums.append(np.sum(example_output2))

    np.mean(sums)

```

Approximately, for enough iterations we can conclude np.sum(example_output) ~ np.mean(sums)

So this means we can assume that given that we de-activate some neurons, the equivalent TOTAL weight of all the neurons
being active equals the TOTAL weight of the active neurons after applying the dropout multiplied by
(1 / (1-dropout_rate))

If we don t do this we would be making weights bigger than what they should be because of the dropout

### This worth to revisit

- Softmax + Closs Entropy derivative
- Optimizers, particularly the moving averages and momentums and how they are infered
