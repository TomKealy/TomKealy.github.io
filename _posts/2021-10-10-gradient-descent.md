---
layout: default
title:  "An introduction to Gradient Descent"
subtitle: How to optimise basically any function.
date:   2021-10-10
categories: optimisation
---

Gradient Descent is the simplest learning algorithm. It is very easy to implement, is robust to complex cost functions, and is the foundation of an enormous zoo of variations.

## Introduction

Some ideas are so ubiquitous and all-encompassing that it's difficult to remember that they were once invented. Other ideas are so simple and apparently obvious that they seem less invented and more handed down by God. Gradient descent is one such idea. It there is hardly a branch of applied mathematics which doesn't use it, or not it seems to have appeared fully formed as an algorithmic idea for both the primordial mathematical time.

It's worth noting how Agoras trusted an idea, gradient descent is is it allows the researcher to numerically compute an optimal answer even when they can't analytically describe the function over which they are optimizing. Seen further, all that is required is some measurement of how our solution does as we form the optimal solution, and that the function we are optimizing over be smooth over our domain of interest.

It's seems too good to be true! Yet the core issue in many applications such as signal processing, operations research, economics, and statistics boil down to the following minimization problem

$$ \mathrm{min}_{\theta \in \mathcal{R^m}} J\left(\theta\right) $$

where $$ J\left(\theta\right) $$ is what is called the *cost function,* which measures how well the model parameters $$ \theta $$ fit to a given dataset. A few examples would be

* Logistic regression

$$ J\left(\theta\right) = \sum_{n} \log{\left(1 + \exp{y_i X_{i}^T\theta}\right)} $$

* Linear regression

$$ J\left(\theta\right) = \frac{1}{2}\lvert\lvert X\theta - y \rvert\rvert^2 $$

* Composite

$$ J\left(\theta\right) = f\left(\theta\right) + g\left(\theta\right) $$

with $ f $ being 'well behaved' and $ g $ causing us some trouble.

We assume that $$ J\left(\theta\right) $$ is bounded from below (i.e. the minimum is not $$-\infty $$). This just guarantees the existence of a solution. If $ J $ is convex this is no problem, but if $ J $ is not (and also not smooth) then it's much more difficult. Always check that a solution exists before wasting time minimising!

There are a couple of ways you could find such an $ \theta $, given a cost function. The most straigtforward is to start with some initital value, and then move in the direction of the negative gradient of the cost funtion:  

$$ \theta_{k+1} = \theta_{k} - \eta\nabla_{\theta} J\left(\theta\right) $$

with $ \Theta_0 = 0 $. Here. $\eta$ is the learning rate - a tuneable parameter. We terminate the algorithm once

$$ \lvert \theta_{k+1} -  \theta_{k} \rvert \leq \varepsilon $$

Intuitively, $$ J\left(\theta\right) $$ defines a surface over $$ \theta\ $$. We, the end-users, want to find the lowest point on that surface (or the lowest point subject to some intersection constraint). Gradient descent finds that lowest point by constructing a sequence of approximations to $$\theta^*$$ (the optimal point), with eath $$\theta_{k}$$ (the interations of the algorithm) always in the direction of steepest descent from $$\theta_{k-1}$$. We will formalise this intuition and provide performance guarantees later.

## Linear Regression

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
```

We'll generate some data that we'll use for the rest of this post:

```python
def generate_data(n, m=2.25, b=6.0, stddev=1.5):
    """Generate n data points approximating given line.
    m, b: line slope and intercept.
    stddev: standard deviation of added error.
    Returns pair x, y: arrays of length n.
    """
    x = np.linspace(-2.0, 2.0, n)
    y = x * m + b + np.random.normal(loc=0, scale=stddev, size=n)
    return x.reshape(-1, 1), y

N = 500
X, y = generate_data(N)
plt.scatter(X, y)
```


    
Suppose you have a set of observations of some process you wanted to model, for example the size of a house labelled as $$ x_i \in \mathrm{R}^n $$, and the house price labeleld as $$ y_i \in \mathrm{R} $$, $$ i = 1 \ldots m $$ (i.e. you have $$m$$ examples). One good choice for a model is linear:

$$ \hat{y}\left(x\right) = Ax + b $$

The goal is to find some suitable $$ A \in \mathcal{R}^n $$ and $$ b \in \mathcal{R} $$, to model this process corectly.

For convenience, let's make some new definitions:

$$\Theta = [A: b]$$
$$ X = [x_1: \ldots : x_m :1]$$
$$Y = [y_1: \ldots : y_m :1]$$

What we've done her is to stack our parameters into $$\Theta$$ so that it's a $$ \mathrm{R}^{\left(n+1\right)\times 1} $$ vector, we have stacked our examples into an $$ \mathrm{R}^{\left(n+1\right) \times m} $$ matrix, and we have stacked all the outputs into a $$ \mathrm{R}^{\left(n+1\right) \times 1} $$ vector. Now our hypothesis can be written as 

$$ \hat{Y} = \Theta^T X $$.

Say you also had good reason to believe that the best reconstruction of $$ x $ you could possilby hope to achieve was to minimise the following (mean squared) error measure:

$$ J\left(\Theta\right) = \frac{1}{n} \lVert \hat{Y} - Y\rVert_2^2 $$

A function to compute the cost is below:

```python
def MSE(y, yhat):
    """Function to compute MSE between true values and estimate
    
    y: true values
    yhat: estimate
    """
    assert(y.shape == yhat.shape)
    return np.square(np.subtract(y, yhat)).mean()
```


There are a couple of ways you could find such an $4 \hat{Y} $$, given a cost function. The most straigtforward is to start with some initital value, and then move in the direction of the negative gradient of the cost funtion:  

$$ \Theta_{k+1} = \Theta_{k} - \alpha\nabla_{\Theta} J\left(\Theta\right) $$ 

with $$ \Theta_0 = 0 $$. Here. $$\alpha$$ is the learning rate - a tuneable parameter. We'll have more to say about that later.

The following function does exactly this, 

```python
def gradient_descent(X, y, cost, learning_rate=0.01, num_iters=250):
    m, n = X.shape
    theta = np.zeros((n, 1))
    yhat = theta.T @ X.T
    yield theta, cost(y.reshape(-1, 1), yhat.T)
    
    for i in range(num_iters):
        yhat = theta.T @ X.T
        yhatt = yhat.T
        nabla = np.sum(X.T @ (y.reshape(-1, 1) - yhatt), axis=1).reshape(-1, 1)
        assert(nabla.shape == theta.shape)
        theta +=  (2 * learning_rate / m) * nabla
        yield theta, cost(y.reshape(-1, 1), yhat.T)
```

Some notes on the algorithm: it may be mysterious as to why we divide the learning rate y the number of samples $$ m $$. But this is a way to average the gradient across all samples. This approach has several important implications:

Scaling of the Gradient: When you compute the gradient of the cost function, you're essentially summing up the gradients calculated for each individual sample. If you don't average this sum by dividing by $$m$$, the magnitude of the gradient could become very large, especially with a large number of samples. This large gradient could lead to very large steps when updating your parameters, potentially causing overshooting and failure to converge to the minimum of the cost function.

Consistent Learning Rate By dividing by $$ m $$: you ensure that the learning rate behaves consistently regardless of the number of samples. This means that the learning rate you choose is less dependent on the size of your dataset. Without this scaling, you might need to adjust your learning rate based on the size of your dataset, which is not ideal.

Numerical Stability: Scaling by $$m$$ also contributes to numerical stability. It prevents the gradient values from becoming too large, which can cause numerical issues in computation (like overflow or underflow problems).

Interpretability: It makes the learning rate parameter more interpretable and independent of the sample size. This is beneficial when you want to use the same learning rate for datasets of different sizes or when comparing the performance of the algorithm on different datasets.

Now we check graphically if the fitted parameters make sense:

```python
ones = np.ones_like(X)
X = np.concatenate([X, ones], axis=1)
thetas = gradient_descent(X, y, MSE)
final = [(t, c) for t,c in thetas]
costs = [x[1] for x in final]
theta = final[-1][0]
plt.plot(costs)
```
## Costs figure goes here

```python
theta
    array([[ 2.27769915],
           [ 5.90213934]])

x = np.linspace(-2.0, 2.0)
yhat = x * theta[0] + theta[1]
plt.scatter(X[:,0], y)
plt.plot(x, yhat, "r-")
```

## Fit goes here.

We could also seek to minimise the least absolute deviations of our predictions from the data:

$$ J\left(\Theta\right)  = \frac{1}{n} \lVert \hat{Y} - Y\rVert_1 $$

a function to do this is included below:


```python
def MAE(y, yhat):
    """Function to compute LAE between true values and estimate
    
    y: true values
    yhat: estimate
    """
    assert(y.shape == yhat.shape)
    return np.absolute(y - yhat).mean()
```

Finally, if you are minimising the MSE you can compute it analytically via the normal equations (left as an exercise to the reader):

$$ \hat{\theta} = (X^T X)^{-1} X^Ty $$

We can do this with the following code:

```python
def normal_equation(X, y, ones=False):
    """
    linear regression using the normal equation.
    X: Feature matrix
    y: Target variable array
    ones: Add a col of ones to Feature matrix
    Returns the calculated theta values.
    """
    # Adding a column of ones to X for the intercept term
    if ones:
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate([ones, X], axis=1)

    # Compute the coefficients
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta
  
theta_analytic = normal_equation(X, y)

theta, theta_analytic 
```

You can derive this equation as follows:

1. Take the (vector) derivative of the cost function to find:

$$ \nabla_{\theta} J\left(\theta\right) = X\left(X^T\theta - y\right) $$

2. Set the derivative to zero and rearrange:

$$ X\left(X^T\theta - y\right) = 0 $$

$$ X^TX \theta = X^T y $$

$$ \theta = \left( X^TX  \right)^{-1} X^T y $$

Intuitivelyy the product $$ X^Ty $$ is the projection of $$ y $$ onto the space spanned by the columns of $$ X $$. This operation translates the target values into the "language" of our features.

The matrix $$ X^TX $$ is a feature-feature correlation matrix. It's symmetrical and captures how each feature relates to every other feature. Taking the inverse of this matrix is akin to understanding how to uniquely weigh each feature to best describe the target $$ y $$. The inverse undoes the mixing of feature influences, allowing us to isolate the effect of each feature.

When you multiply the inverse of $$ X^TX $$ with $$ X^Ty $$, you apply the unique weighting to the projection of $$ y $$ in feature space. This results in the coefficients $$ \Theta $$ that best map your features to your target in a least-squares sense.

Note that even though this has a nice interpretation calculating $$\left( X^TX  \right)^{-1} $$ is prohibitively expensive (it's on $$ \mathcal{O}\left(n^3\right) $$ operation). In practice gradient descent is always used.

## Logistic Regression

Logistic regression is used when we need our outputs to be strictly within the interval $$ [0, 1] $$. It's usually taught as a classification algorithm, but it's best thought of as a regression which predictst **probabilities**. Logistic regression minimises the following loss function:

$$ J\left(w\right) = y * p\left( y_i | x_i ; w \right) + (1-y) * (1 - p\left( y_i | x_i ; w \right)) $$

There is a mathematical justification for why this is the right loss to use, but heuristically, this loss minimises the probability error between the predicition classes and the true classes.

The other major difference is that we choose the sigmoid function. $ \sigma\left(X\theta\right) $ with

$$ \sigma\left(z\right) = \frac{1}{1+e^{-z}} $$

instead of $$ X^t\theta $$ as the hypothesis function. The sigmoid function maps any real-valued number into the range $$(0, 1)$$, making it useful for a probability estimate. Essentially what we are doing here with logistic regression is doing a linear regression but then transforming the outputs to probability space. We'll predict the class of each point using softmax (multinomial logistic) regression. The model has a matrix $$ W $$ of weights, which measures for each feature how likely that feature is to be in a particular. It is of size $$ \mathrm{n_{features}} \times \mathrm{n_{classes}} $$. The goal of softmax regression is to learn such a matrix. Given a matrix of weights, $$ W $$, and matrix of points, $$ X $$, it predicts the probability od each class given the samples.

In code we can do it like this:

```python
	def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def logistic_cost(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -(1/m) * jnp.sum(y * jnp.log(h) + (1 - y) * jnp.log(1 - h))
    return cost

def logistic_regression(X, y, cost_fn, learning_rate=0.01, num_iters=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    y = y[:, 0]
    y = y.reshape(-1, 1)
    cost_history = []
    
    for i in range(num_iters):
        z = X @ theta
        h = sigmoid(z)
        gradient = (1/m) * X.T @ (h - y)
        theta -= learning_rate * gradient
        
        cost = cost_fn(theta, X, y)
        cost_history.append(cost)

    return theta, cost_history
    
def make_blobs(num_samples=1000, num_features=2, num_classes=2):
    mu = np.random.rand(num_classes, num_features)
    sigma = np.ones((num_classes, num_features)) * 0.1
    samples_per_class = num_samples // num_classes
    x = np.zeros((num_samples, num_features))
    y = np.zeros((num_samples, num_classes))
    for i in range(num_classes):
        class_samples = np.random.normal(mu[i, :], sigma[i, :], (samples_per_class, num_features))
        x[i * samples_per_class:(i+1) * samples_per_class] = class_samples
        y[i * samples_per_class:(i+1) * samples_per_class, i] = 1
    return x, y
    

def plot_clusters(x, y, num_classes=2):
    temp = np.argmax(y, axis=1)
    colours = ['r', 'g', 'b']
    for i in range(num_classes):
        x_class = x[temp == i]
        plt.scatter(x_class[:, 0], x_class[:, 1], color=colours[i], s=1)
    plt.show()

NUM_FEATURES=50
NUM_CLASSES=2
NUM_SAMPLES=1000

X, y, = make_blobs(num_samples=NUM_SAMPLES, num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
plot_clusters(X, y, num_classes=NUM_CLASSES)
``

## Gradient Descent without Gradients

We can see the outline of what we need: a function which represents the cost of our operation, and something to compute the gradients. Fortunately `autograd` does exactly this! 

According to the webiste, autograd 

> Autograd can automatically differentiate native Python and Numpy code. It can handle a large subset of Python's features, including loops, ifs, recursion and closures, and it can even take derivatives of derivatives of derivatives. It supports reverse-mode differentiation (a.k.a. backpropagation), which means it can efficiently take gradients of scalar-valued functions with respect to array-valued arguments, as well as forward-mode differentiation, and the two can be composed arbitrarily. The main intended application of Autograd is gradient-based optimization.