---
layout: default
title:  "Understanding the Guts of Generalized Additive Models (GAMs) with Hands-on Examples."
subtitle: Generalised Additive Models look harder than they actually are.
date:   2021-08-10
categories: GAMs
toc: true
---

This post will explain  the internals of (generalised) additive models (GAMs): how to estimate the feature functions. We first explain *why* we want to fit additive models, and then we go on to explain how they are estimated in practice via splines (we will also explain what a spline is).  Finally we'll fit simple splines on wage data, then we'll go on to fit more complicated splines on some accelerometer data with a highly non-linear realtionship between in the input and the output.

# Introduction

The additive model for regression expresses the conditional expectation function as a sum of partial response functions, one for each predictor variable. That's a lot to take in in a single sentence! The idea is that each input feature makes a separate contribution to the response, and these contributions just add up (hence the name "partial response function"). However, these contributions don’t have to be strictly proportional to the inputs.

Formally, when the vector $$ \mathbf{X} $$ of predictor variables has $$ p $$ dimensions, $$ x_1, \ldots, x_p $$, the model is given by:

$$
E[Y|\mathbf{X} = \mathbf{x}] = \alpha + \sum_{j=1}^{p} f_j(x_j)
$$

This  includes the linear model as a special case, where $$ f_j(x_j) = \beta_j x_j $$, but it’s clearly more general because the $$ f_j $$'s can be arbitrary nonlinear functions.

To make the model identifiable, we add a restriction. Without loss of generality, we assume:

$$
E[Y] = \alpha \quad \text{and} \quad E[f_j(X_j)] = 0
$$

To see why this restriction is necessary, consider a simple case where $$ p = 2 $$. If we add constants $$ c_1 $$ to $$ f_1 $$ and $$ c_2 $$ to $$ f_2 $$, but subtract $$ c_1 + c_2 $$ from $$ \alpha $$, nothing observable changes in the model. This kind of degeneracy or lack of identifiability is similar to the way collinearity prevents us from defining true slopes in linear regression. However, it’s less harmful than collinearity because we can resolve it by imposing this constraint.

# Why would we do this!?

Before we dive into how additive models work in practice, let’s take a step back and understand *why* we might want to use them. Think of regression models as existing on a spectrum, with linear models on one end and fully nonparametric models on the other. Additive models sit somewhere in between.

Linear regression is fast—it converges quickly as we get more data, but it assumes the world is simple and linear, which it almost never is. Even with infinite data, a linear model will make systematic mistakes due to its inability to capture non-linear relationships. Its mean squared error (MSE) shrinks at a rate of $$\mathcal{O}(n^{-1}) $$, and this speed is unaffected by the number of parameters.

At the opposite extreme, fully nonparametric methods like kernel regression or k-nearest neighbors make no such assumptions. They can capture complex shapes, but they pay for that flexibility with much slower convergence, especially in higher dimensions. In one dimension, these models converge at a rate of $$ \mathcal{O}(n^{-4/5}) $$, but in higher dimensions, the rate drops dramatically to something like $$ \mathcal{O}(n^{-1/26}) $$ in 100 dimensions—this is the notorious "curse of dimensionality." Essentially, the more features you have, the more data you need to maintain the same level of precision, and the data demands grow exponentially.

So where do additive models fit in? They balance these two extremes. Additive models allow for some non-linearity while avoiding the curse of dimensionality. They achieve this by estimating each component function $$ f_j(x_j) $$ independently, using one-dimensional smoothing techniques. The result is a model that converges almost as fast as parametric ones—at a rate of $$ \mathcal{O}(n^{-4/5}) $$, rather than the much slower rate of fully nonparametric models. Sure, there’s a little approximation bias, but the trade-off is worth it. In practice, additive models often outperform linear models once you have enough data, because they capture more of the underlying structure without demanding the impossible amount of data required by fully nonparametric approaches.

This is where additive models shine: they give you enough structure to work with high-dimensional data, but with far less approximation bias than linear models.

```python
import pandas as pd
import patsy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf

%matplotlib inline
```
# Splines

We build the $$ f_i $$ using a type of function called a spline; splines allow us to automatically model non-linear relationships without having to manually try out many different transformations on each variable. The name “spline” comes from a tool used by craftsmen to draw smooth curves—a thin, flexible strip of material like soft wood. You pin it down at specific points, called knots, and let it bend between them.

Bending a spline takes energy—the stiffer the material, the more energy is needed to bend it into shape. A stiffer spline creates a straighter curve between points. For smoothing splines, increasing $$\lambda$$ corresponds to using a stiffer material.

We have data points $$(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$$, and we want to find a good approximation $$\hat{\mu}$$ to the true conditional expectation or regression function $$\mu$$. In many regression setting we control how smooth we made $$\hat{\mu}$$ indirectly. For instance, in kernel regression, we use a bandwidth parameter to adjust the size of the neighborhood around each data point that contributed to the prediction. A larger bandwidth would create a smoother curve by averaging over more data points, but this could lead to underfitting, where the model fails to capture important details in the data. On the other hand, a smaller bandwidth creates a more jagged, detailed curve, which might overfit the data by following every minor fluctuation.

In $$k$$-nearest neighbors (k-NN) regression, the smoothness of the prediction depends on how many neighbors $$k$$ we use to make predictions. A higher value of $$k$$ effectively increases the "bandwidth," as we are averaging over more points, which smooths out the predictions. This can lead to a smoother but less flexible model that might miss important patterns in the data (underfitting). A lower value of $$k$$ results in predictions based on fewer neighbors, leading to more jagged, less smooth results, as the model reacts to small variations in the data (overfitting).

Why not be more direct, and control smoothness itself?

A natural way to do this is by minimizing the spline objective function:

$$
L(m, \lambda) \equiv \frac{1}{n} \sum_{i=1}^{n} \left( y_i - m(x_i) \right)^2 + \lambda \int \left( m''(x) \right)^2 dx
$$

The first term here is just the mean squared error (MSE) of using the curve $$m(x)$$ to predict $$y$$. The second term, penalises the MSE in the direction of smoother curves. $$m''(x)$$ is the second derivative of $$m$$ with respect to $$x$$—it would be zero if $$m$$ were linear, so this term measures the curvature of $$m$$ at $$x$$. The sign of $$m''(x)$$ tells us whether the curvature is concave or convex, but we don’t care about that, so we square it. Then, we integrate this over all $$x$$ to get the total curvature. Finally, we multiply by $$lambda$$ and add that to the MSE. This adds a penalty to the MSE criterion—given two functions with the same MSE, we prefer the one with less average curvature. In practice, we’ll accept changes in $$m$$ that increase the MSE by 1 unit if they reduce the average curvature by at least $$lambda$$.

The curve or function that solves this minimization problem:

$$
\hat{\mu}_{\lambda} = \arg \min_m L(m, \lambda) \tag{7.2}
$$

is called a smoothing spline or spline curve.

All solutions to the spline cost equation no matter what the data, are piecewise cubic polynomials that are continuous and have continuous first and second derivatives. That is, $$\hat{\mu}$$, $$\hat{\mu}'$$, and $$\hat{\mu}''$$ are all continuous. The boundaries between the pieces are the original data points, and, like the craftsman’s spline, these boundary points are called the knots of the smoothing spline. The function remains continuous beyond the largest and smallest data points but is always linear in those regions.

Let's think a little more about the optimisation proboem. There are two limits to consider: as $$ \lambda \to \infty $$ and As $$ \lambda \to 0 $$. As $$ \lambda \to \infty $$, any curvature at all becomes infinitely costly, and only linear functions are allowed. This makes sense because minimizing mean squared error (MSE) with linear functions is simply **Ordinary Least Squares (OLS)** regression. So, we understand that in this limit, the smoothing spline behaves just like a linear regression model.

On the other hand, as $$ \lambda \to 0 $$, we don’t care about curvature at all. In this case, we can always come up with a function that just **interpolates between the data points**—this is called an **interpolation spline**, which passes exactly through each point. More specifically, of the infinitely many functions that interpolate between those points, we pick the one with the minimum average curvature.

At intermediate values of $$ \lambda $$, the smoothing spline $$ \hat{\mu}_{\lambda} $$ becomes a compromise between fitting the data closely and keeping the curve smooth. The larger we make $$ \lambda $$, the more heavily we penalize curvature. There’s a clear **bias-variance trade-off** here:

- As $$ \lambda $$ grows, the spline becomes less sensitive to the data. It’s **less wiggly**, meaning lower variance in its predictions but higher bias since it may fail to capture intricate patterns.
- As $$ \lambda $$ shrinks, the bias decreases, but the spline becomes **more sensitive** to fluctuations in the data, resulting in higher variance.

For example, if $$ \lambda $$ is large, the spline may resemble a simple line that overlooks small dips and peaks in the data. If $$ \lambda $$ is small, the spline may start capturing noise and overfitting, creating a curve that twists excessively between points.

For consistency, we want to let $$ \lambda \to 0 $$ as $$ n \to \infty $$, just as we allow the bandwidth $$ h \to 0 $$ in kernel smoothing when $$ n \to \infty $$. This allows the spline to capture more complexity as we gather more data, while still avoiding overfitting.

The way to think about smoothing splines is as functions that minimize MSE, but with a constraint on the average curvature. This follows a general principle in optimization where **penalized optimization** corresponds to **optimization under constraints**.  The short version is that each level of $$ \lambda $$ corresponds to setting a cap on how much curvature the function is allowed to have on average. The spline we fit for a particular $$ \lambda $$ is the MSE-minimizing curve, subject to that constraint. As we get more data, we can **relax the constraint** (by letting $$ \lambda $$ shrink) without sacrificing reliable estimation.

It should come as no surprise that we select $$ \lambda $$ by **cross-validation**. Ordinary **k-fold cross-validation** works, but **leave-one-out CV** performs well for splines. In fact, the default in most spline software is either leave-one-out CV or the faster approximation called **generalized cross-validation (GCV)**. 

# Fitting Splines

Splines are piecewise cubic polynomials. To see how to fit them, let’s think about how to fit a global cubic polynomial. We would define four basis functions,

$$
B_1(x) = 1 
$$

$$
B_2(x) = x 
$$

$$
B_3(x) = x^2 
$$

$$
B_4(x) = x^3 
$$

and choose to only consider regression functions that are linear combinations of the basis functions:

$$
\mu(x) = \sum_{j=1}^4 \beta_j B_j(x)
$$

Such regression functions would be linear in the transformed variables $$B_1(x), \dots, B_4(x)$$, even though it is nonlinear in $$x$$.

To estimate the coefficients of the cubic polynomial, we would apply each basis function to each data point $$x_i$$ and gather the results in an $$n \times 4$$ matrix $$B$$:

$$
B_{ij} = B_j(x_i)
$$

Then we would do OLS using the $$B$$ matrix in place of the usual data matrix $$x$$:

$$
\hat{\beta} = (B^T B)^{-1} B^T y 
$$

Since splines are piecewise cubics, things proceed similarly, but we need to be a little more careful in defining the basis functions. Recall that we have $$n$$ values of the input variable $$x$$, $$x_1 , x_2 , \dots, x_n$$. Assume that these are in increasing order, because it simplifies the notation. These $$n$$ “knots” define $$n + 1$$ pieces or segments: $$n - 1$$ of them between the knots, one from $$-\infty$$ to $$x_1$$, and one from $$x_n$$ to $$+\infty$$. A third-order polynomial on each segment would seem to need a constant, linear, quadratic, and cubic term per segment. So the segment running from $$x_i$$ to $$x_{i+1}$$ would need the basis functions

$$
1(x_i,x_{i+1})(x), \, (x - x_i) 1(x_i,x_{i+1})(x), \, (x - x_i)^2 1(x_i,x_{i+1})(x), \, (x - x_i)^3 1(x_i,x_{i+1})(x)
$$

where the indicator function $$1(x_i,x_{i+1})(x)$$ is 1 if $$x \in (x_i,x_{i+1})$$ and 0 otherwise. This makes it seem like we need $$4(n + 1) = 4n + 4$$ basis functions.

However, we know from linear algebra that the number of basis vectors we need is equal to the number of dimensions of the vector space. The number of adjustable coefficients for an arbitrary piecewise cubic with $$n + 1$$ segments is indeed $$4n + 4$$, but splines are constrained to be smooth. The spline must be continuous, which means that at each $$x_i$$, the value of the cubic from the left, defined on $$(x_{i-1},x_i)$$, must match the value of the cubic from the right, defined on $$(x_i, x_{i+1})$$. This gives us one constraint per data point, reducing the number of adjustable coefficients to at most $$3n + 4$$. Since the first and second derivatives are also continuous, we are down to just $$n + 4$$ coefficients. Finally, we know that the spline function is linear outside the range of the data, i.e., on $$(-\infty,x_1)$$ and on $$(x_n,\infty)$$, lowering the number of coefficients to $$n$$. There are no more constraints, so we end up needing only $$n$$ basis functions. And in fact, from linear algebra, any set of $$n$$ piecewise cubic functions which are linearly independent can be used as a basis.

One common choice is:

$$
B_1(x) = 1
$$

$$
B_2(x) = x
$$

$$
B_{i+2}(x) = \frac{(x - x_i)^3_+ - (x - x_n)^3_+}{x_n - x_i} - \frac{(x - x_{n-1})^3_+ - (x - x_n)^3_+}{x_n - x_{n-1}}
$$

where $$ (a)_+ = a $$ if $$ a > 0 $$, and $$ = 0 $$ otherwise. This rather unintuitive-looking basis has the nice property that the second and third derivatives of each $$B_j$$ are zero outside the interval $$(x_1, x_n)$$.

Now that we have our basis functions, we can once again write the spline as a weighted sum of them:

$$
m(x) = \sum_{j=1}^m \beta_j B_j(x)
$$

and put together the matrix $$B$$ where $$B_{ij} = B_j(x_i)$$. We can write the spline objective function in terms of the basis functions:

$$
L = (y - B\beta)^T (y - B\beta) + n\lambda\beta^T \Omega\beta
$$

where the matrix $$\Omega$$ encodes information about the curvature of the basis functions:

$$
\Omega_{jk} = \int B_j''(x) B_k''(x) dx \tag{7.16}
$$

Notice that only the quadratic and cubic basis functions will make non-zero contributions to $$\Omega$$. With the choice of basis above, the second derivatives are non-zero on, at most, the interval $$(x_1,x_n)$$, so each of the integrals in $$\Omega$$ is going to be finite. This is something we (or, realistically, R) can calculate once, no matter what $$\lambda$$ is. Now we can find the smoothing spline by differentiating with respect to $$\beta$$:

$$
0 = -2B^T y + 2B^T B\hat{\beta} + 2n\lambda\Omega\hat{\beta}
$$

$$
B^T y = (B^T B + n\lambda\Omega) \hat{\beta}
$$

$$
\hat{\beta} = (B^T B + n\lambda\Omega)^{-1} B^T y
$$

Notice, incidentally, that we can now show splines are linear smoothers:

$$
\hat{\mu}(x) = B\hat{\beta} = B(B^T B + n\lambda\Omega)^{-1} B^T y \tag{7.20}
$$

Once again, if this were ordinary linear regression, the OLS estimate of the coefficients would be $$(x^T x)^{-1} x^T y$$. In comparison to that, we’ve made two changes. First, we’ve substituted the basis function matrix $$B$$ for the original matrix of independent variables, $$x$$—a change we’d have made already for a polynomial regression. Second, the "denominator" is not $$x^T x$$, or even $$B^T B$$, but $$B^T B + n\lambda\Omega$$. Since $$x^T x$$ is $$n$$ times the covariance matrix of the independent variables, we are taking the covariance matrix of the spline basis functions and adding some extra covariance—how much depends on the shapes of the functions (through $$\Omega$$) and how much smoothing we want to do (through $$\lambda$$). The larger we make $$\lambda$$, the less the actual data matters to the fit.

In addition to explaining how splines can be fit quickly (do some matrix arithmetic), this illustrates two important tricks. One, which we won’t explore further here, is to turn a nonlinear regression problem into one which is linear in another set of basis functions. This is like using not just one transformation of the input variables, but a whole library of them, and letting the data decide which transformations are important. There remains the issue of selecting the basis functions, which can be quite tricky. In addition to the spline basis, most choices are various sorts of waves—sine and cosine waves of different frequencies, various wave-forms of limited spatial extent ("wavelets"), etc. The ideal is to choose a function basis where only a few non-zero coefficients would need to be estimated, but this requires some understanding of the data.

The other trick is that of stabilizing an unstable estimation problem by adding a penalty term. This reduces variance at the cost of introducing some bias.

# Example: Wage Data

First of all, we'll use `patsy` to construct a few spline bases and fit generalised linear models with `statsmodels`. Then, we'll dive into constructing splines ourselves; following Simon Wood's book we'll use penalised regression splines.

Firstly, we'll use `patsy` to create some basic pline models. The data we're using comes from https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Wage.html. It's plotted below:


```python
df = pd.read_csv('Wage.csv')
age_grid = np.arange(df.age.min(), df.age.max()).reshape(-1,1)
plt.scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.1)
```




    <matplotlib.collections.PathCollection at 0x11d0a5898>




    
![png](2021-08-10-The-Guts-Of-GAMs_files/2021-08-10-The-Guts-Of-GAMs_3_1.png)
    


GAMs are essentially linear models, but in a very special (and useful!) basis made of regression splines. We can use the `bs()` function in `patsy` to create such a basis for us:


```python
transformed_x1 = patsy.dmatrix("bs(df.age, knots=(25,40,60), degree=3, include_intercept=False)", {"df.age": df.age}, return_type='dataframe')
fit1 = sm.GLM(df.wage, transformed_x1).fit()
```


```python
fit1.params
```




    Intercept                                                               60.493714
    bs(df.age, knots=(25, 40, 60), degree=3, include_intercept=False)[0]     3.980500
    bs(df.age, knots=(25, 40, 60), degree=3, include_intercept=False)[1]    44.630980
    bs(df.age, knots=(25, 40, 60), degree=3, include_intercept=False)[2]    62.838788
    bs(df.age, knots=(25, 40, 60), degree=3, include_intercept=False)[3]    55.990830
    bs(df.age, knots=(25, 40, 60), degree=3, include_intercept=False)[4]    50.688098
    bs(df.age, knots=(25, 40, 60), degree=3, include_intercept=False)[5]    16.606142
    dtype: float64




```python
age_grid = np.arange(df.age.min(), df.age.max()).reshape(-1,1)
pred = fit1.predict(patsy.dmatrix("bs(age_grid, knots=(25,40,60), include_intercept=False)",
{"age_grid": age_grid}, return_type='dataframe'))
plt.scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.1)
plt.plot(age_grid, pred, color='b', label='Specifying three knots')
plt.xlim(15,85)
plt.ylim(0,350)
plt.xlabel('age')
plt.ylabel('wage')
```




    Text(0,0.5,'wage')




    
![png](2021-08-10-The-Guts-Of-GAMs_files/2021-08-10-The-Guts-Of-GAMs_7_1.png)
    


Here we have prespecified knots at ages 25, 40, and 60. This produces a spline with six basis functions. A cubic spline has 7 degrees of freedom: one for the intercept, and two for each order. We could also have specified knot points at uniform quantiles of the data:


```python
# Specifying 6 degrees of freedom
transformed_x2 = patsy.dmatrix("bs(df.age, df=6, include_intercept=False)",
{"df.age": df.age}, return_type='dataframe')
fit2 = sm.GLM(df.wage, transformed_x2).fit()
fit2.params
```




    Intercept                                       56.313841
    bs(df.age, df=6, include_intercept=False)[0]    27.824002
    bs(df.age, df=6, include_intercept=False)[1]    54.062546
    bs(df.age, df=6, include_intercept=False)[2]    65.828391
    bs(df.age, df=6, include_intercept=False)[3]    55.812734
    bs(df.age, df=6, include_intercept=False)[4]    72.131473
    bs(df.age, df=6, include_intercept=False)[5]    14.750876
    dtype: float64




```python
age_grid = np.arange(df.age.min(), df.age.max()).reshape(-1,1)
pred = fit2.predict(patsy.dmatrix("bs(age_grid, df=6, include_intercept=False)",
{"age_grid": age_grid}, return_type='dataframe'))
plt.scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.1)
plt.plot(age_grid, pred, color='b', label='Specifying three knots')
plt.xlim(15,85)
plt.ylim(0,350)
plt.xlabel('age')
plt.ylabel('wage')
```




    Text(0,0.5,'wage')




    
![png](2021-08-10-The-Guts-Of-GAMs_files/2021-08-10-The-Guts-Of-GAMs_10_1.png)
    


Finally, we can also fit natural splines with the `cr()` function:


```python
# Specifying 4 degrees of freedom
transformed_x3 = patsy.dmatrix("cr(df.age, df=4)", {"df.age": df.age}, return_type='dataframe')
fit3 = sm.GLM(df.wage, transformed_x3).fit()
fit3.params
```




    Intercept             -6.970341e+13
    cr(df.age, df=4)[0]    6.970341e+13
    cr(df.age, df=4)[1]    6.970341e+13
    cr(df.age, df=4)[2]    6.970341e+13
    cr(df.age, df=4)[3]    6.970341e+13
    dtype: float64




```python
pred = fit3.predict(patsy.dmatrix("cr(age_grid, df=4)", {"age_grid": age_grid}, return_type='dataframe'))
plt.scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.1)
plt.plot(age_grid, pred, color='g', label='Natural spline df=4')
plt.legend()
plt.xlim(15,85)
plt.ylim(0,350)
plt.xlabel('age')
plt.ylabel('wage')
```




    Text(0,0.5,'wage')




    
![png](2021-08-10-The-Guts-Of-GAMs_files/2021-08-10-The-Guts-Of-GAMs_13_1.png)
    


Let's see how these fits all stack together:


```python
# Generate a sequence of age values spanning the range
age_grid = np.arange(df.age.min(), df.age.max()).reshape(-1,1)
# Make some predictions
pred1 = fit1.predict(patsy.dmatrix("bs(age_grid, knots=(25,40,60), include_intercept=False)",
{"age_grid": age_grid}, return_type='dataframe'))
pred2 = fit2.predict(patsy.dmatrix("bs(age_grid, df=6, include_intercept=False)",
{"age_grid": age_grid}, return_type='dataframe'))
pred3 = fit3.predict(patsy.dmatrix("cr(age_grid, df=4)", {"age_grid": age_grid}, return_type='dataframe'))
# Plot the splines and error bands
plt.scatter(df.age, df.wage, facecolor='None', edgecolor='k', alpha=0.1)
plt.plot(age_grid, pred1, color='b', label='Specifying three knots')
plt.plot(age_grid, pred2, color='r', label='Specifying df=6')
plt.plot(age_grid, pred3, color='g', label='Natural spline df=4')
plt.legend()
plt.xlim(15,85)
plt.ylim(0,350)
plt.xlabel('age')
plt.ylabel('wage')
```




    Text(0,0.5,'wage')




    
![png](2021-08-10-The-Guts-Of-GAMs_files/2021-08-10-The-Guts-Of-GAMs_15_1.png)
    



```python
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import patsy
import scipy as sp
import seaborn as sns
from statsmodels import api as sm

%matplotlib inline
```


```python
df = pd.read_csv('mcycle.csv')
df = df.drop('Unnamed: 0', axis=1)
```


```python
fig, ax = plt.subplots(figsize=(8, 6))
blue = sns.color_palette()[0]
ax.scatter(df.times, df.accel, c=blue, alpha=0.5)
ax.set_xlabel('time')
ax.set_ylabel('Acceleration')
```




    Text(0,0.5,'Acceleration')




    
![png](2021-08-10-The-Guts-Of-GAMs_files/2021-08-10-The-Guts-Of-GAMs_18_1.png)
    




As discussed earlier: GAMs are smooth, semi-parametric models of the form:
​
$$ y = \sum_{i=0}^{n-1} \beta_i f_i\left(x_i\right) $$
​
where \\(y\\) is the dependent variable, \\(x_i\\) are the independent variables, \\(\beta\\) are the model coefficients, and \\(f_i\\) are the feature functions.
​
We build the \\(f_i\\) using a type of function called a spline. Since our data is 1D, we can model it as:

$$ y = \beta_0 + f\left( x \right) + \varepsilon $$

We must also choose a basis for \\( f \\):

$$ f \left( x \right) = \beta_1 B_1\left(x\right) + \ldots + \beta_k B_k\left(x\right) $$

We define 

$$ X = \left[1, x_1,  \ldots,  x_k \right] $$

so we can write:

$ y = \beta_0 + f\left( x \right) + \varepsilon = X\beta + \varepsilon $$

We choose to minimise the sum of squares again, this time with a regularisation term:

$$ \frac{1}{2} \lVert y - X\beta \rVert + \lambda \int_0^1 f''\left(x\right)^2 dx $$

You can show (you, not me!) that the second term can always be written:

$$ \int_0^1 f''\left(x\right)^2 dx = \beta^T S \beta $$

where \\( S \\) is a postive (semi)-definiate matrix (i.e. all it's eigenvalues are positive or 0). Therefore our objective function becomes:

$$ \frac{1}{2} \lVert y - X\beta \rVert + \lambda \beta^T S \beta dx $$
 
and we can use the techniques we've developed fitting linear models to fit additive models! We'll start by fitting a univariate spline, then maybe something more complicated.


```python
def R(x, z):
    return ((z - 0.5)**2 - 1 / 12) * ((x - 0.5)**2 - 1 / 12) / 4 - ((np.abs(x - z) - 0.5)**4 - 0.5 * (np.abs(x - z) - 0.5)**2 + 7 / 240) / 24

R = np.frompyfunc(R, 2, 1)

def R_(x):
    return R.outer(x, knots).astype(np.float64)
```


```python
q = 20

knots = df.times.quantile(np.linspace(0, 1, q))
```


```python
y, X = patsy.dmatrices('accel ~ times + R_(times)', data=df)
```


```python
S = np.zeros((q + 2, q + 2))
S[2:, 2:] = R_(knots)
```


```python
B = np.zeros_like(S)
B[2:, 2:] = np.real_if_close(sp.linalg.sqrtm(S[2:, 2:]), tol=10**8)
```

    /Users/thomas.kealy/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: ComplexWarning: Casting complex values to real discards the imaginary part
      



```python
def fit(y, X, B, lambda_=1.0):
    # build the augmented matrices
    y_ = np.vstack((y, np.zeros((q + 2, 1))))
    X_ = np.vstack((X, np.sqrt(lambda_) * B))
    
    return sm.OLS(y_, X_).fit()
```


```python
min_time = df.times.min()
max_time = df.times.max()

plot_x = np.linspace(min_time, max_time, 100)
plot_X = patsy.dmatrix('times + R_(times)', {'times': plot_x})

results = fit(y, X, B)

fig, ax = plt.subplots(figsize=(8, 6))
blue = sns.color_palette()[0]
ax.scatter(df.times, df.accel, c=blue, alpha=0.5)
ax.plot(plot_x, results.predict(plot_X))
ax.set_xlabel('time')
ax.set_ylabel('accel')
ax.set_title(r'$\lambda = {}$'.format(1.0))
```




    Text(0.5,1,'$\\lambda = 1.0$')




    
![png](2021-08-10-The-Guts-Of-GAMs_files/2021-08-10-The-Guts-Of-GAMs_27_1.png)
    



```python

```
