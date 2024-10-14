---
layout: post
title:  "Bayesian Sequential Hypothesis Testing"
date:   2024-02-04
categories: hypothesis-testing
---

{% newthought 'tl; dr We show how mSPRT' %} can be thought of as a Bayesian Algorithm..<!--more--> 

### A Bayesian Model
Consider the following Bayesian model for the data:
$$Y_1|\alpha, \delta, M_1 \sim \mathcal{N}\left(\frac{1}{n_1} \alpha + \frac{1}{2n_1} \delta, \sigma^2 I_{n_1}\right)$$

$$Y_2|\alpha, \delta, M_1 \sim \mathcal{N}\left(\frac{1}{n_2} \alpha - \frac{1}{2n_2} \delta, \sigma^2 I_{n_2}\right)$$

$$\delta|M_1 \sim \mathcal{N} (0, \tau^2)$$

$$p(\alpha|M_1) \propto 1$$

Note that the prior distribution for the lift $$\delta$$ is equal to the mixing distribution in the mSPRT. If you are not familiar with this vectorized notation, an alternative notation is

$$y_{1i}|\alpha, \delta, M_1 \sim \mathcal{N} (\alpha + \frac{\delta}{2}, \sigma^2) \text{ } \forall i = 1, \ldots, n_1$$
$$y_{2j}|\alpha, \delta, M_1 \sim \mathcal{N} (\alpha - \frac{\delta}{2}, \sigma^2) \text{ } \forall j = 1, \ldots, n_2$$

In this model for the "alternative" we have a grand mean $$\alpha$$ and a lift parameter $$\delta$$. 

Following a Bayesian approach, we assign $$\alpha$$ a uniform prior, and $$\delta$$ a Normal prior. Under this model, the interpretation of $$\delta$$ is still unchanged as $$E[y_1]-E[y_2]$$ i.e., the expected difference in means. Moreover, it also follows under this model that $$\bar{Y}_1 - \bar{Y}_2 \sim \mathcal{N} (\delta, \sigma^2(\frac{1}{n_1} + \frac{1}{n_2}))$$. Notice, however, that we have conditioned on this model being the alternative model $$M_1$$. The "null" model can be expressed as

$$Y_1|\alpha, M_0 \sim \mathcal{N} (\frac{1}{n_1} \alpha, \sigma^2 I_{n_1})$$
$$Y_2|\alpha, M_0 \sim \mathcal{N} (\frac{1}{n_2} \alpha, \sigma^2 I_{n_2})$$
$$p(\alpha|M_0) \propto 1$

in which there is no $$\delta$$ i.e., it is the same as the alternative model except that our prior on $$\delta$$ concentrates all of its mass at zero. A Bayesian would test the hypothesis that $$\delta \neq 0$$ by looking at the Bayes Factor between models $$M_1$$ and $$M_0$$

$$
\frac{p(Y_1, Y_2|M_1)}{p(Y_1, Y_2|M_0)} = \frac{\int p(Y_1, Y_2|\alpha, \delta)d\alpha d\delta}{\int p(Y_1, Y_2|\alpha)d\alpha},
$$

note that we are explicitly integrating out the nuisance intercept parameter $$\alpha$$ w.r.t. a uniform prior. Let’s go ahead and compute these integrals!

To make the mathematics easier, it is helpful to stack the observations $$Y_1, Y_2$$ into a larger multivariate normal:

$$Y = [Y_1', Y_2']' \sim \mathcal{N} (1_n\alpha + X\delta, \sigma^2I_n)$$

where $$n = n_1 + n_2$$, $$1_n$$ is a vector of 1’s, $$I_n$$ is an identity matrix and $$X' = \frac{1}{2}[1_{n_1}', -1_{n_2}']$$.

The first trick is to marginalize out the intercept parameter by computing the component of $$Y$$ that is orthogonal to the column space of $$1_n$$. Dropping the $$n$$'s from now on, let $$P_1 = \frac{1}{n}1(1'1)^{-1}1' = \frac{1}{n} 11'$$ be the projection operator onto the column space of $$1$$. This neatly isolates the component of $$\alpha$$ in the quadratic form i.e.

$$
\begin{aligned}
\|Y - 1\alpha - X\delta\|^2_2 &= (Y - 1\alpha - X\delta)'(Y - 1\alpha - X\delta) \\
&= (Y - 1\alpha - X\delta)'(P_1 + I - P_1)(Y - 1\alpha - X\delta) \\
&= \|P_1(Y - 1\alpha - X\delta)\|^2_2 + \|(I - P_1)(Y - 1\alpha - X\delta)\|^2_2 \\
&= n(\alpha - \bar{Y} - \bar{X}\delta)^2 + \|Y_c - X_c\delta\|^2_2
\end{aligned}
$$

where $$Y_c$$ and $$X_c$$ are the centered observations and design matrix

$$
Y_c = (I - P_1)Y = Y - 1 \bar{Y}
$$

$$
X_c = (I - P_1)X = \frac{1}{2}
\begin{bmatrix}
1_{n_1} \\
-1_{n_2}
\end{bmatrix}
- \frac{1}{2}
\frac{n_1 - n_2}{n_1 + n_2}
\begin{bmatrix}
1_{n_1} \\
1_{n_2}
\end{bmatrix}
= \frac{1}{n_1 + n_2}
\begin{bmatrix}
n_2 1_{n_1} \\
-n_1 1_{n_2}
\end{bmatrix}.
$$

As a foreshadowing of a result later it is also suggestive to note at this point that

$$
X_c'X_c = \frac{n_1n_2}{n_1 + n_2} = \frac{1}{\frac{1}{n_1} + \frac{1}{n_2}}.
$$

This has rearranged the quadratic form in the likelihood into two terms, one a quadratic in $$\alpha$$, which makes the integration step easy.

$$
p(Y_1, Y_2|\delta, M_1) = \frac{1}{n} \left(\frac{1}{2\pi\sigma^2}\right)^{\frac{n-1}{2}} \exp\left(-\frac{1}{2\sigma^2} \|Y_c - X_c\delta\|^2\right)
$$

$$
p(Y_1, Y_2|M_0) = \frac{1}{n} \left(\frac{1}{2\pi\sigma^2}\right)^{\frac{n-1}{2}} \exp\left(-\frac{1}{2\sigma^2} \|Y_c\|^2\right)
$$

Integrating out the intercept from the model had the effect of losing a degree of freedom and working with the centered observations and design matrix. The final integration is the marginalization step of $$\delta$$ w.r.t. the prior $$N (0, \tau^2)$$:

$$
p(Y_1, Y_2|M_1) = \frac{1}{n} \left(\frac{1}{2\pi\sigma^2}\right)^{\frac{n-1}{2}} \exp\left(-\frac{1}{2\sigma^2} \|Y_c\|^2\right) \left(\frac{1}{2\pi\tau^2}\right)^{\frac{1}{2}} \exp\left(\frac{1}{2} \left(X_c'X_c\sigma^2 + 1\tau^2\right)^{-1} (X_c'Y_c\sigma^2)^2\right)
$$

When forming the Bayes factor, most of these terms cancel:

$$
\frac{p(Y_1, Y_2|M_1)}{p(Y_1, Y_2|M_0)} = \left(\frac{1}{\tau^2} \frac{1}{\tau^2 + X_c'X_c\sigma^2}\right)^{\frac{1}{2}} \exp\left(\frac{1}{2} \left(X_c'X_c\sigma^2 + 1\tau^2\right)^{-1} (X_c'Y_c\sigma^2)^2\right),
$$

yet $$X_c'X_c/\sigma^2 = 1/(\sigma^2(\frac{1}{n_1} + \frac{1}{n_2}))$$ which is $$1/\rho^2$$ under our earlier definition of $$\rho^2$$ and

$$
X_c'Y_c =X_c'Y_c + X_c'1 \bar{Y} = X_c'Y = \frac{n_2n_1}{n_1 + n_2} (\bar{Y}_1 - \bar{Y}_2) = \frac{\bar{Y}_1 - \bar{Y}_2}{\frac{1}{n_1} + \frac{1}{n_2}}.
$$

The Bayes factor then simplifies to
$$
\frac{p(Y_1, Y_2|M_1)}{p(Y_1, Y_2|M_0)} = \left(\frac{\rho^2}{\rho^2 + \tau^2}\right)^{\frac{1}{2}} \exp\left(\frac{1}{2} \frac{\tau^2}{\rho^2 + \tau^2} \frac{(\bar{Y}_1 - \bar{Y}_2)^2}{\rho^2}\right)
=\Lambda
$$