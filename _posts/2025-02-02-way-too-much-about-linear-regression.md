---
layout: default
title:  Way too much about linear regression.
subtitle: There's a lot of details
date:   2025-02-02
categories: regression
toc: true
---

# Introduction
There are a lot of ways to understand (multiple) linear regression, but it takes a lot of work to integrate them all together. This guide attempts to do that. It starts off deriving the solution to the linear regression problem—how linear regression is the maximum likelihood solution for ordinary least squares—and then moves on to the Frisch-Waugh-Lovell theorem as a tool to understand how the perspectives all hang together.

# An algebraic perspective.
Suppose you have a set of observations of some process you wanted to model, for example the size of a house labelled as $x_i \in \mathbb{R}^n$, and the house price labeled as $y_i \in \mathbb{R}$, $i=1\ldots m$ (i.e. you have $m$ examples). One good choice for a model is linear:

$$
\hat{y}(x) = Ax + b
$$

The goal is to find some suitable $A \in \mathbb{R}^n$ and $b \in \mathbb{R}$, to model this process correctly.

For convenience, let's make some new definitions:

$$
\Theta = [A:b]
$$

$$
X = [x_1:\ldots:x_m:1]
$$

$$
Y = [y_1:\ldots:y_m:1]
$$

What we've done here is to stack our parameters into $\Theta$ so that it's a $\mathbb{R}^{(n+1)\times 1}$ vector, we have stacked our examples into an $\mathbb{R}^{(n+1)\times m}$ matrix, and we have stacked all the outputs into a $\mathbb{R}^{(n+1)\times 1}$ vector. Now our hypothesis can be written as

$$
\hat{Y} = \Theta^T X
$$

Say you also had good reason to believe that the best reconstruction of $x$ you could possibly hope to achieve was to minimise the following (mean squared) error measure:

$$
J(\Theta) = \frac{1}{n}\|\hat{Y}-Y\|_2^2
$$

This is **linear regression**—the fundamental tool in statistical modeling. It's the swiss army knife of statistical modeling, and using it can get you really far. Real-world applications often require more sophisticated approaches.

You can do some calculus and show that the closed-form solution for Ordinary Least Squares (OLS) regression with matrices of arbitrary size is:

$$
\beta = (X'X)^{-1}X'y
$$

However this solution is a little opaque. The solution defines a hyperplane in the feature space, but that's no better than just staring at the solution. It's difficult to interpret why this specific hyperplane has been chosen.

If you squint, you can make algebraic sense of the solution: pre-multiplying by $X'$ will give us $X'y = \beta X'X$ and then you have to 'divide' by the matrix $X'X$ to get the solution for $\beta$. But this is unsatisfactory: following some algebraic steps won't tell you why this is the solution, and it won't help in making generalisations (such as ridge regession or the Lasso) legible.

# A statistical persective.

In simple linear regression (with one independent variable and a constant term), you can do a little bit of different algebra to show that the coefficient $\beta$ can be expressed as:

$$
\beta = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}
$$

This makes some more intuitive sense!

A linear regression coefficient tells us: If predictor variable $x$ increases by 1, what is the expected increase in outcome variable $y$?

The answer to this question depends in large part on the scales on which $x$ and $y$ are measured. E.g. $x$ is a measure of length, imagine measuring in millimeters or centimeters; the variance of measurements in millimeters will be 10^2 times the variance of the same measurements in centimeters; the covariance will be multiplied by 10. Note, $\text{Cov}(x,y)$ is determined by three things:

1. the linear association between $x$ and $y$.
2. the scale of $x$.
3. the scale of $y$.

Because of 2 and 3, the covariance is an unstandardized measure of association. Its value is difficult to interpret, because what would be a large and what would be a small value depends on the scales of $x$ and $y$. The correlation coefficient, however, gives us a standardized measure of association: It is 'corrected' for the scales on which $x$ and $y$ are measured.

$$
\text{Cor}(x,y) = \frac{\text{Cov}(x,y)}{\sqrt{\text{Var}(x)\cdot\text{Var}(y)}}
$$

The correlation coefficient tells us: If $x$ increases by $\sqrt{\text{Var}(x)}$ how many $\sqrt{\text{Var}(y)}$'s will outcome $y$ increase? Thus, with a correlation coefficient of 1, an increase of 1 SD in $x$ $ is associated with an increase of 1 SD in $y$.

Now, the regression coefficient quantifies the expected increase in $y$, when $x$ increases by 1. We thus need to 'correct' the covariance between $x$ and $y$ for the scale of $x$. We can do that by simply dividing:

$$
\frac{\text{Cov}(x,y)}{\text{Car}(x)}
$$

# Projections

A projection matrix $P$ is any matrix with property $P^2=P$. For vector $v$ in space and point $x$, projection $Px$ returns closest point $\bar{x}$ along $v$.

## How do I find a projection matrix for a given $v$?

That's not at all that helpful. Instead we want to find the projection matrix for a given $v$. To begin, we can take some point in $n$-dimensional space, $x$, and the vector line $v$ along which we want to project $x$. The goal is the following:

$$
\begin{align}
\text{argmin}_c \sqrt{\sum_i(\bar{x}_i-x)^2} &= \text{argmin}_c \sum_i(\bar{x}_i-x)^2 \\
&= \text{argmin}_c \sum_i(cv_i-x)^2
\end{align}
$$

This rearrangement follows since the square root is a monotonic transformation, such that the optimal choice of $c$ is the same in both minimisations. Since any potential $x^*$ along the line drawn by $v$ is some scalar multiplication of that line $cv$, we can express the function to be minimised with respect to $c$, and then differentiate:

$$
\begin{align}
\frac{d}{dc}\sum_i(cv_i-x)^2 &= \sum_i 2v_i(cv_i-x) \\
&= 2(\sum_i cv_i^2 - \sum_i v_ix) \\
&= 2(cv'v-v'x) \\
&= 0
\end{align}
$$

So

$$
\begin{align}
2(cv'v-v'x) &= 0 \\
&\Rightarrow cv^Tv-v^Tx = 0 \\
&\Rightarrow cvTv = v^Tx \\
&\Rightarrow c = (v^Tv)^{-1}v^Tx \\
\end{align}
$$

So we end up with:

$$
P_v = v(v^Tv)^{-1}v^T
$$

Which looks suspiciously similar to the formula for the optimal linear regression coefficients!

# Linear Regression as a projection.

Suppose we have a vector of outcomes $y \in \mathcal{R}^n$, and some $p$-dimensional matrix $X$ of predictors. We write the linear regression model as:

$$
y=X^T\beta+\varepsilon
$$ 
where $\beta$ is a vector of coefficients, and $\varepsilon$ is additive Gaussian white noise. Linear regression minimises the sum of the squared residuals:

$$
\hat{y}=argmin_{\beta}(y−Xβ)′(y−Xβ)
$$
  
Differentiating with respect to and solving:

$$
\begin{align}
\frac{d}{d\beta}(y−Xβ)^T(y−Xβ) &= −2X(y−Xβ) \\
&= 2X^TXβ−2X^Ty \\
&= 0 \\
&\rightarrow \\
X^TXβ &= X^Ty\\
\beta &=(X^TX)^{−1}X^Ty.
\end{align}
$$

To get our prediction of $y$, i.e. $\hat{y}$ , we multiply our $\beta$ coefficient by the matrix $X$:

$$
\hat{y} = X\beta = X(X^TX)^{-1}X^Ty.
$$

The OLS derivation of $\hat{y}$ is very similar to $P=X(X^TX)^{-1}X$, the orthogonal prediction matrix. The two differ only in that that $\hat{y}$ includes the original outcome vector $y$. But, $Py = X(X^TX)^{-1}X^Ty=\hat{y}$! Hence the predicted values from a linear regression simply are an orthogonal projection of $y$ onto the space defined by $X$

## Geometric interpretation

An alternative way to think about linear regression is by flipping our usual perspective—instead of thinking of. Instead of thinking about each feature (X, Y) as dimensions and plotting data points as observations in this feature space, we can treat each observation as a dimension and view features as vectors in this observation-space.

As a concrete example we can think of houses for sale having a set of features (floorspace, rooms, location, sale price etc) and we record each house's values of those features. Instead we can think of each house as a dimension, and represent each feature (floorspace, rooms, location, sale price) as a vector in this house-space.

if we had 100 houses in our dataset:

- Rather than thinking of each house as a point in feature-space (e.g. plotting houses on axes of price vs. floorspace)
- We'd have a 100-dimensional space (one dimension per house)
- Each feature would be a vector in this 100-dimensional space
- The sale price vector would be some point in this space
- And linear regression would find the combination of feature vectors that gets closest to the sale price vector.

In this dual space linear regression becomes a geometric problem where:

- Your outcome variable $y$ is a vector pointing somewhere in this observation-space
- Your predictor variables (including a constant term) are also vectors
- The combinations of your predictor vectors create a "span" or "column space" - imagine this like a flat surface in the observation-space
- Linear regression finds the point on this surface that's closest to your Y vector
- The regression coefficients are just the scaling factors you need to combine your predictor vectors to reach this closest point

Linear regression is mathematically identical to an "orthogonal projection" - you're literally projecting your Y vector onto the surface defined by your predictors in the most direct (perpendicular) way possible.

While this might seem abstract, it helps explain why linear regression works the way it does - it's finding the best possible approximation of Y that can be created by combining your predictor variables, where "best" means "closest in terms of Euclidean distance."

![Two views of linear regression](/assets/images//LinearProjections/linear_projections.png "The two views of linear regression")
*Figure 1: Two views of linear regression. (a) Traditional view: Each point represents an observation. (b) Vector view: Each axis represents an observation.*

# Frisch-Waugh-Lovell

Any multiple linear regression is just a series of simple linear regressions in a trenchcoat.

This isn't so important today, where every researcher has access to a computer that can do the heavy lifting for them. Statisticians can spend more time thinking about what they put into regressions than how to do them. But back before there were computers, calculating regressions took a long time. There columns of numbers that you had to add up, and then you would have had to do some division as well. Knowing the details about the OLS model really sped up those hours with a slide rule and a table of logs.

A lot of the regressions statisticians and economists were concerned with involved panel data: observations of the same person over time. So one natural question you needed to ask, was:

- Should I detrend each variable and then perform a regression on the detrended variables?

Or

- Add time as a variable to your model without detrending?

The Frisch-Waugh-Lovell theorem says that these procedures are the same thing. In other words, time as a variable detrends the other parameters. The theorem is a version of the geometric intuition of [[Linear regression as a Projection]]: OLD projects $y$ onto the space spanned by the features $X$ to get the estimate $\hat{y}$.

Formally, FWL states that each predictor’s coefficient in a multivariate regression explains that variance of $y$ not explained by both the other $k-1$ predictors’ relationship with the outcome and their relationship with that predictor, i.e. the independent effect of $x_j$.

The [proof](https://bookdown.org/ts_robinson1994/10EconometricTheorems/frisch.html) is pretty boring and uses a the properties of projection matrices.

# Putting it all together

We had two questions earliner:

1. The $\frac{1}{N}$ Factor: Variance is typically calculated with a $\frac{1}{N}$ or $\frac{1}{N-1}$ factor, while $X^TX$ doesn't include this normalization. Why does the statistical formula use $\text{Var}(x)$?

2. Mean Centering: Variance is calculated using centered values: $\text{Var}(X) = \frac{1}{N} \sum(x-\bar{x})^2$ not just $\frac{1}{N} \sum x^2$. Where does this mean-centering come from?

The first question has a satisfying answer: covariance also includes a $\frac{1}/{N}$ factor! These factors cancel out in the ratio $ \frac{\text{Cov}(x,y)}{\text{Var}(x)}$, making it a convenient way to express the relationship.

The second question leads us to a deeper understanding of regression's underlying mechanics.

## The Role of Orthogonalization

The mean-centering in variance calculations isn't arbitrary - it's a consequence of how OLS handles the constant term. OLS orthogonalizes effects with respect to each column vector in X.

To understand why orthogonalization matters, consider a simple example:

$$
y = 3 + 4x + 5z
$$

If we set $x=1$ and $z=1$, our predicted $y$ is 12. Increasing $z$ to 2 increases our prediction by exactly 5. This shows that $∂y/∂z = 5$, demonstrating the orthogonality of our coefficients. (If the variables weren't orthogonal, changing one variable would affect the contribution of others.)

In simple regression, we have two columns:

* A constant vector (all 1's)
* Our independent variable $x$

The coefficient $\beta$ must be orthogonalized with respect to the constant term. But what does orthogonality mean in this context?

Two vectors are orthogonal if and only if their dot product is 0. For a constant vector $c=[1,1,1...]$ and our $x$ $[x_1,x_2,x_3...]$ their dot product is:

$$
x_1 + x_2 + x_3 ... = \sum_{i}^N x_j
$$

To make this sum zero, we subtract the mean $\bar{x}$ from each element:

$$
(x_1 - \bar{x}) + (x_2 - \bar{x}) + ... = (x_1 + x_2 + ...) - N \bar{x} * N = 0
$$

This reveals why we use centered values in variance calculations: centering a vector is equivalent to orthogonalizing it with respect to a constant term.

Now we can see how $X^TX^{-1} relates to $\frac{1}{\text{Var}(x)}$. The mean-centering in variance calculations isn't just a statistical convention - it's a fundamental consequence of how regression handles the constant term through orthogonalization.

This helps explain why $\beta = \frac{\text{Cov}(x,y)}{\text{Var}(x)}$ works, and provides insight into the geometric foundations of linear regression.
