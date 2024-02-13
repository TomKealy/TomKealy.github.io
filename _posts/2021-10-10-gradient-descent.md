---
layout: post
title: "The Ups and Downs of Gradient Descent"
date:   2021-10-10
categories: optimisation
---

Some ideas are so ubiquitous and all-encompassing that it's difficult to remember that they were once invented. Other ideas are so simple and apparently obvious that they seem less invented and more handed down by God. Gradient descent is one such idea. It there is hardly a branch of applied mathematics which doesn't use it, or not it seems to have appeared fully formed as an algorithmic idea for both the primordial mathematical time.

It's worth noting how Agoras trusted an idea, gradient descent is is it allows the researcher to numerically compute an optimal answer even when they can't analytically describe the function over which they are optimizing. Seen further, all that is required is some measurement of how our solution does as we form the optimal solution, and that the function we are optimizing over be smooth over our domain of interest.

It's seems too good to be true! Yet the core issue in many applications such as signal processing, operations research, economics, and statistics boil down to the following minimization problem

$$ \mathrm{min}_{\theta \in \mathcal{R^m}} J\left(\theta\right) $$

where $ J\left(\theta\right) $ is what is called the *cost function,* which measures how well the model parameters $ \theta $ fit to a given dataset. A few examples would be

* Logistic regression

$$ J\left(\theta\right) = \sum_{n} \log{\left(1 + \exp{y_i X_{i}^T\theta}\right)} $$

* Linear regression

$$ J\left(\theta\right) = \frac{1}{2}\lvert\lvert X\theta - y \rvert\rvert^2 $$

* Composite

$$ J\left(\theta\right) = f\left(\theta\right) + g\left(\theta\right) $$

with $ f $ being 'well behaved' and $ g $ causing us some trouble.

We assume that $ J\left(\theta\right) $ is bounded from below (i.e. the minimum is not $-\infty $). This just guarantees the existence of a solution. If $ J $ is convex this is no problem, but if $ J $ is not (and also not smooth) then it's much more difficult. Always check that a solution exists before wasting time minimising!

There are a couple of ways you could find such an $ \theta $, given a cost function. The most straigtforward is to start with some initital value, and then move in the direction of the negative gradient of the cost funtion:  

$$ \theta_{k+1} = \theta_{k} - \eta\nabla_{\theta} J\left(\theta\right) $$ 

with $ \Theta_0 = 0 $. Here. $\eta$ is the learning rate - a tuneable parameter. We terminate the algorithm once

$$ \lvert \theta_{k+1} -  \theta_{k} \rvert \leq \varepsilon $$

Gradient Descent is the simplest learning algorithm. It is very easy to implement, is robust to complex cost functions, and is the foundation of an enormous zoo of variations.

Intuitively, $ J\left(\theta\right) $ defines a surface over $ \theta\ $. We, the end-users, want to find the lowest point on that surface (or the lowest point subject to some intersection constraint). Gradient descent finds that lowest point by constructing a sequence of approximations to $\theta^*$ (the optimal point), with eath $\theta_{k}$ (the interations of the algorithm) always in the direction of steepest descent from $\theta_{k-1}$. We will formalise this intuition and provide performance guarantees later.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
%config InlineBackend.figure_format = 'retina'
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x109cba630>

    
![png](2021-02-01-Gradietnt-Descent_files/2021-02-01-Gradietnt-Descent_2_1.png)
    
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
```


```python
N = 500
X, y = generate_data(N)
plt.scatter(X, y)
```




    <matplotlib.collections.PathCollection at 0x10ee65438>




    
![png](2021-02-01-Gradietnt-Descent_files/2021-02-01-Gradietnt-Descent_5_1.png)
    


For convenience, let's define $\Theta = [A: b]$ be a $ (n+1) \times 1 $ vector, let $ X = [x_1: \ldots : x_m :1] \in \mathrm{R}^{(n+1) \times m} $, and let $Y = [y_1: \ldots : y_m :1]$ (i.e. we expanded both to include the intercept, and we concatenate all the examples into a single matrix of x's and y's respectively). Now out hypothesis can be written as $ \hat{Y} = \Theta^T X $ 

Say you also had good reason to believe that the best reconstruction of $ x $ you could possilby hope to achieve was to minimise the following (mean squared) error measure:

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

There are a couple of ways you could find such an $ \hat{Y} $, given a cost function. The most straigtforward is to start with some initital value, and then move in the direction of the negative gradient of the cost funtion:  

$$ \Theta_{k+1} = \Theta_{k} - \alpha\nabla_{\Theta} J\left(\Theta\right) $$ 

with $ \Theta_0 = 0 $. Here. $\alpha$ is the learning rate - a tuneable parameter.

The following function does exactly this, using autograd to avoid mathematically computing the gradients.




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


```python
ones = np.ones_like(X)
X = np.concatenate([X, ones], axis=1)
thetas = gradient_descent(X, y, MSE)
```


```python
final = [(t, c) for t,c in thetas]
costs = [x[1] for x in final]
theta = final[-1][0]
plt.plot(costs)
```




    [<matplotlib.lines.Line2D at 0x10eedc898>]




    
![png](2021-02-01-Gradietnt-Descent_files/2021-02-01-Gradietnt-Descent_14_1.png)
    



```python
theta
```




    array([[ 2.27769915],
           [ 5.90213934]])




```python
x = np.linspace(-2.0, 2.0)
yhat = x * theta[0] + theta[1]
```


```python
plt.scatter(X[:,0], y)
plt.plot(x, yhat, "r-")
```




    [<matplotlib.lines.Line2D at 0x10eebad30>]




    
![png](2021-02-01-Gradietnt-Descent_files/2021-02-01-Gradietnt-Descent_17_1.png)
    


Firstly, if you are minimising the MSE you can compute it analytically via

$$ (A^T A)^{-1} A^Ty $$


```python

```
