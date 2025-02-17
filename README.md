# Assignment 1: Studies of Perceptron Learning and Adaline

## Purposes
- Understanding perceptron learning and Adaline algorithms.
- Investigating the performance difference of the two algorithms.
- Understanding the linear classification problem.
- Understanding mini-batch Stochastic Gradient Descent (SGD).

## Implementation Tasks

### 1. (6 points) Comparisons of Perceptron Learning and Adaline
Use the Python programs for perceptron learning and Adaline given in the textbook and compare:
- Loss
- Number of updates
- Margin of the resulting separation hyperplane
- Convergence

What conclusions can you draw? Can you provide mathematical explanations to support them? Use the Iris dataset (first 2 classes with all features) for testing and provide at least 3 mathematically different conclusions. Ensure fair comparisons by using the same initial parameter values, training data, learning rate, and number of epochs for both algorithms.

### 2. (4 points) Modify Perceptron Class
Modify the `Perceptron` class from the textbook so that the bias term `b` is absorbed into the weight vector `w`. Ensure compatibility with the textbook's training program.

### 3. (6 points) Multiclass Classification with Perceptrons
A perceptron is limited to binary classification, but the Iris dataset contains three classes: setosa, versicolor, and virginica. If multiple perceptrons are allowed, how can you achieve multiclass classification using only perceptrons? Implement a demo program using the modified perceptron class from Task 2.

### 4. (4 points) Implement Mini-Batch SGD in AdalineSGD
The textbook explains both SGD and mini-batch GD. Implement a new training function `fit_mini_batch_SGD` in the `AdalineSGD` class, which combines both approaches. In each epoch, update learning parameters based on a randomly selected subset of training data. The batch size, a hyperparameter, should be user-defined.
