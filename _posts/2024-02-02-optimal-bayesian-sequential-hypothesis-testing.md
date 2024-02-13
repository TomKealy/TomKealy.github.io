---
layout: post
title:  "Optimal Bayesian Sequential Hypothesis Testing"
date:   2024-02-02
categories: hypothesis-testing
---

{% newthought 'tl; dr We introduce the mSPRT' %} and give a derivation from a Bayesian point of view.<!--more--> 

# Sequential Testing

Normally when you go about hypothesis testing, after a random sample is observed one of two possible actions are taken: accept the null hypothesis $$H_0$$, or accepct the tive hypothesis $$H_1$$. In some cases the evidence may strongly support one of the hypotheses, whilst in other cases the evidence may be less convincing. Nevertheless, a decision must he made. All this assumes that all the data has been collected, and no more is available. It doesn't have to be this way. There is a class of hypothesis tests where you can safely collect more data when the evidence is ambiguous. Such a test typically continues until the evidence strongly favors one of the two hypotheses.

In case of simple hypotheses the strength of the evidence for $$H$$ is given by the ratio of the probability of the data under $$H_0$$, to the probability of the data under $$H_1$$. We denote this likelihood ratio by $$\Lambda$$. The Neyman-Pearson lemma implies that for a given amount of information the likelihood ratio test is the most powerful test. Such a rule decides to accept $$H_1$$ if $$\Lambda$$ is big enough, and decides to accept $$H_0$$ otherwise. How big $$\Lambda$$ must get to lead to the decicion $$H_1$$, danends on, amond other things, its sampling distribution under $$H_0$$ and $$H_1$$. It is not unusual for "big enough" to mean $$\Lambda \geq 1$$. Such tests could easily decide $$H_0$$ or $$H_1$$ when the actual evidence is neutral.

## SPRT

In 1943 Wald proposed the *sequential probability ratio test* (SPRT). Suppose that $$Y$$ is a random variable with unknown distribution $$f$$. We want to test the following hypotheses:

* $$H_0: f=f_0$$
* $$H_1: f=f_1$$

where $$f_0$$ and $$f_1$$ are specified. We observe values of $$Y$$ successively: $$y_1, y_2, \ldots$$ the random variables $$Y_i$$ corresponding to the $y_i$$ are i.i.d with common distribution $$f$$. Let

$$
\Lambda = \prod_{i=1}^n \frac{f_1\left(x_i\right)}{f_0\left(x_i\right)}
$$

be the likelihood ratio at stage $$n$$. We choose two decision boundaries $$A$$ and $$B$$ such that $$0 < B < A < \infty $$,  we accept $$H_0$$ if $$\Lambda \leq B$$ and $$H_1$$ if $$\Lambda \geq A$$, and we continue if $$B \leq \Lambda \leq A$$. The constants $$A$$ and $$B$$ are determined by the desired false positive and false negative rates of the experimenter.

### Example: IID case

Assume that $$X_1 , X_2 \ldots$$ are independent and identically distributed with distributions $$P$$, and $$Q$$, under $$H_0$$ and $$H_1$$ respectively. Then $$L_n =\Lambda\left(X_n\right)$$. Let $$Z_i=\log{\left(\Lambda\left(X_n\right)\right)}$$. Since $$\log{L_n} = \sum_i^n Z_i$$ the SPRT can be viewed as z random walk (or more properly a family of random walks), with steps $$Z_i$$ which proceeds until it crosses $$\log{B}$$ or $$\log{A}$$.

We now focus on the more general case where $$P$$ and $$Q$$, have either densities or probability mass functions of the form:

$$ f\left(x; \theta\right) = h\left(x\right)\exp{C\left(\theta\right)x - D\left(\theta\right)} $$

where $$\theta $$ is a real valued parameter. The *exponential families* are to work with and include many common distributions. Let $$P$$, be determined by $$f\left(x; \theta_0\right)$$, and let $$Q$$, be determined by $$f\left(x; \theta_1\right)$$. Then 
$$Z_i = \left[C\left(\theta_1\right) - C\left(\theta_0\right) \right]X_i - \left[D\left(\theta_1\right)-D\left(\theta_0\right)\right]$$. 

In terms of the random walk with steps $$Z_i$$ the SPRT continues until fixed boundaries are crossed.

It is somtimes easier to perform the test using the sums of the $$X_i$$'s. Let $$S_n = \sum X_i$$. Assuming that $$ C\left(\theta_1\right) > C\left(\theta_0\right) $$ the test continues until

$$ S_n \geq \frac{ \log{A} }{ C\left(\theta_1\right) - C\left(\theta_0\right) } + n\frac{ D\left(\theta_1\right)-D\left(\theta_0\right) }{ C\left(\theta_1\right) - C\left(\theta_0\right) } $$

Or

$$ S_n \leq \frac{ \log{A} }{ C\left(\theta_1\right) - C\left(\theta_0\right) } + n\frac{ D\left(\theta_1\right)-D\left(\theta_0\right) }{ C\left(\theta_1\right) - C\left(\theta_0\right) } $$

The following tableshow the SPRT for some common distributions:

{% marginnote 'tableID-3' 'Table: SPRTs for various common distributions' %}

|Distribution | $$ f\left(x; \theta\right) $$  |  $$ C\left(\theta\right) $$  |   $$ D\left(\theta\right) $$   |
|:----------------|----:|-----:|-------:|
|Normal        |$$ \frac{1}{\sqrt{2\pi}}e^\left(-\left(x-\theta\right)^2/2\right) $$ | $$\theta$$ |$$\frac{\theta^2}{2} $$ |
|Bernoulli    |$$ \theta^x\left(1-\theta\right)^(1-x) $$   | $$ \log{\left(\frac{\theta}{1-\theta}\right)} $$     | $$ -\log{\left(1-\theta\right)} $$|
|Exponential    |$$ \frac{1}{\theta}e^-x/\theta $$   | $$ \log{\theta}$$     | $$ \theta $$|
|Poisson    |$$ \frac{e^-\theta \theta^x}{x!} $$   | $$ -\frac{1}{\theta}$$     | $$ \log{\theta} $$|

#### Example: Exponential Distribution

A textbook example is parameter estimation of a probability distribution function. Consider the exponential distribution:

$$ f\left(x; \theta\right) = \frac{1}{\theta}e^-x/\theta $$

The hypotheses are

* $$H_0: \theta = \theta_0 $$
* $$H_1: \theta = \theta_1 $$

Where $$ \theta_1 > \theta_1 $$.

Then the log-likelihood function (LLF) for one sample is

$$
\begin{aligned}
\log{\Lambda\left(x\right)} &= \log{ \frac{1}{\theta_1}e^-x/\theta_1  }{ \frac{1}{\theta_0}e^-x/\theta_0  } }
&= -\log{\frac{\theta_1}{\theta_0}} + \frac{\theta_1 - \theta_0}{\theta_1\theta_0}x
\end{aligned}
$$

The cumulative sum of the LLFs for all x is

$$ S_n = \sum_n \log{\Lambda\lefgt(x_i\right)} = -n\log{\frac{\theta_1}{\theta_0}} + \frac{\theta_1 - \theta_0}{\theta_1\theta_0}\sum_n x_i  $$

Accordingly, the stopping rule is:

$$ a <  -n\log{\frac{\theta_1}{\theta_0}} + \frac{\theta_1 - \theta_0}{\theta_1\theta_0}\sum_n x_i < b $$

After re-arranging we finally find

$$ a + n\log{\frac{\theta_1}{\theta_0}} < \frac{\theta_1 - \theta_0}{\theta_1\theta_0}\sum_n x_i < b + n\log{\frac{\theta_1}{\theta_0}} $$

The thresholds are simply two parallel lines with slope $$ \log{\frac{\theta_1}{\theta_0}}$$. Sampling should stop when the sum of the samples makes an excursion outside the continue-sampling region.

#### Example: Binomial Distribution

Suppose 𝑋1,𝑋2,…, are IID Bernoulli(𝑝) random variables and let 𝑝1>𝑝0

.

Set LR←1
and 𝑗←0

.

    Increment 𝑗

If 𝑋𝑗=1
, LR←LR×𝑝1/𝑝0

.

If 𝑋𝑗=0
, LR←LR×(1−𝑝1)/(1−𝑝0)

    .

What’s LR
at stage 𝑚? Let 𝑇𝑚≡∑𝑚𝑗=1𝑋𝑗

.
𝑝1𝑚𝑝0𝑚≡𝑝𝑇𝑚1(1−𝑝1)𝑚−𝑇𝑚𝑝𝑇𝑚0(1−𝑝0)𝑚−𝑇𝑚.

This is the ratio of binomial probability when 𝑝=𝑝1
to binomial probability when 𝑝=𝑝0

(the binomial coefficients in the numerator and denominator cancel). It simplifies further to
𝑝1𝑚𝑝0𝑚=(𝑝0/𝑝1)𝑇𝑚((1−𝑝0)/(1−𝑝1))𝑚−𝑇𝑚.

Wald's SPRT for $p$ in iid Bernoulli trials

Conclude 𝑝>𝑝0

if
𝑝1𝑚𝑝0𝑚≥1−𝛽𝛼.

Conclude 𝑝≤𝑝0

if
𝑝1𝑚𝑝0𝑚≤𝛽1−𝛼.

Otherwise, draw again.

The SPRT approximately minimizes the expected sample size when 𝑝≤𝑝0
or 𝑝>𝑝1. For values in (𝑝1,𝑝0), it can have larger sample sizes than fixed-sample-size tests.

#### Example: Normal Distribution

Consider the Normal distribution:

$$ f\left(x; \theta\right) = \frac{1}{\sqrt{2\pi}}e^\left(-\left(x-\theta\right)^2/2\right) $$

The hypotheses are

* $$H_0: \theta = \theta_0 $$
* $$H_1: \theta = \theta_1 $$

Where $$ \theta_1 > \theta_1 $$.

Then the log-likelihood function (LLF) for one sample is

$$
\begin{aligned}
\log{\Lambda\left(x\right)} &= \log{ \frac{1}{\sqrt{2\pi}}e^\left(-\left(x-\theta_1\right)^2/2\right)  }{ \\frac{1}{\sqrt{2\pi}}e^\left(-\left(x-\theta_0\right)^2/2\right)  } }
&= -\frac{1}{2}\log{\frac{\theta_1}{\theta_0}} + \frac{\theta_1 - \theta_0}{2\theta_1\theta_0}x_i^2 - \frac{\theta_1 - \theta_0}{2}
\end{aligned}
$$

The cumulative sum of the LLFs for all x is

$$ S_n = \sum_n \log{\Lambda\lefgt(x_i\right)} = -n\log{\frac{\theta_1}{\theta_0}} + \frac{\theta_1 - \theta_0}{\theta_1\theta_0}\sum_n x_i  $$

Accordingly, the stopping rule is:

$$ a <  -n\log{\frac{\theta_1}{\theta_0}} + \frac{\theta_1 - \theta_0}{\theta_1\theta_0}\sum_n x_i < b $$

After re-arranging we finally find

$$ a + n\log{\frac{\theta_1}{\theta_0}} < \frac{\theta_1 - \theta_0}{\theta_1\theta_0}\sum_n x_i < b + n\log{\frac{\theta_1}{\theta_0}} $$

The thresholds are simply two parallel lines with slope $$ \log{\frac{\theta_1}{\theta_0}}$$. Sampling should stop when the sum of the samples makes an excursion outside the continue-sampling region.


## The Mixture Sequential Probability Ratio Test (mSPRT)

The main limitation of the Sequential Probability Ratio Test (SPRT) is its requirement for specifying an explicit alternative hypothesis, which might not always align with the goal of merely rejecting a null hypothesis. In contrast, the modified Sequential Probability Ratio Test (mSPRT) offers flexibility in determining sample sizes at the start of an experiment, unlike the traditional SPRT, where sample size calculations are not feasible. Consequently, parameters such as $$N$$, $$\alpha$$, and $$\beta$$ are fixed at the outset of an experiment.

The mixture Sequential Probability Ratio Test (mSPRT) amends these limitations. The test is defined by a "mixing" distribution $$H$$ over a parameter space $$\Theta$$, where $$H$$ is assumed to have a density $$h$$ that is positive everywhere. Utilizing $$H$$, we first compute the following mixture of likelihood ratios against the null hypothesis $$\theta = \theta_0$$:

$$
\Lambda_n(s_n) = \int_{\Theta} \left(\frac{f_{\theta}(s_n)}{f_{\theta_0}(s_n)}\right)^n dH(\theta).
$$

The mSPRT is parameterized by a mixing distribution $$H$$ over $$\Theta$$, which is restricted to have an everywhere continuous and positive derivative. Given an observed sample average $$s_n$$ up to time $$n$$, the likelihood ratio of $$\theta$$ against $$\theta_0$$ is $$\left(\frac{f_{\theta}(s_n)}{f_{\theta_0}(s_n)}\right)^n$$. Thus, we define the mixture likelihood ratio with respect to $$H$$ as:

$$
\Lambda_n(s_n) = \int_{\Theta} \left(\frac{f_{\theta}(s_n)}{f_{\theta_0}(s_n)}\right)^n dH(\theta).
$$

### Application to Normal Data

Considering normal data, for any \(\mu_A\) and \(\mu_B\), the difference \(Z_n = Y_n - X_n\) follows a normal distribution \(\sim N(\theta, 2\sigma^2)\). We can apply the one-variation mSPRT to the sequence \(\{Z_n\}\), leading to the following definition:

$$
\sqrt{\frac{2\sigma^2}{2\sigma^2 + n\tau^2}} \exp{\left(\frac{n^2\tau^2\left(X_n - Y_n - \theta_0\right)^2}{4\sigma^2\left(2\sigma^2 + n\tau^2\right)}\right)}.
$$

```python
def compute_mSPRT(y_data, x_data, sigma_squared, tau_squared, theta_0):
    n = len(y_data)
    y_mean = np.mean(y_data)
    x_mean = np.mean(x_data)
    lambda_n = np.sqrt(2 * sigma_squared / (2 * sigma_squared + n * tau_squared)) * \
               np.exp(n * tau_squared * (y_mean - x_mean - theta_0) ** 2 / (4 * sigma_squared * (2 * sigma_squared + n * tau_squared)))
    return lambda_n
```


```python
# Parameters for synthetic data generation
mu_X = 0
mu_Y = 3  # Assuming a non-zero delta for illustration
sigma = 1.0  # Standard deviation for both groups
n_samples = 100  # Number of samples per group

# Generate synthetic data
np.random.seed(42)
x_data = np.random.normal(mu_X, sigma, n_samples)
y_data = np.random.normal(mu_Y, sigma, n_samples)
```


```python
sigma_squared = sigma ** 2
tau_squared = 1  # Assumed variance of the prior
theta_0 = 0  # Difference in means under H0

# Initialize list to store mSPRT values
msprt_values = []

for n in range(1, n_samples + 1):
    lambda_n = compute_mSPRT(y_data[:n], x_data[:n], sigma_squared, tau_squared, theta_0)
    msprt_values.append(lambda_n)

# Convert mSPRT values to always valid p-values
p_values = [1]
for i, lambda_val in enumerate(msprt_values):
    p_values.append(min(p_values[i-1], 1 / lambda_val))

# Plot always valid p-values
plt.figure(figsize=(10, 6))
plt.plot(p_values[1:], label='Always Valid p-values')
plt.xlabel('Number of Observations')
plt.ylabel('p-value')
plt.title('Always Valid p-values Over Time')
plt.legend()
plt.show()


# Plot always valid p-values
plt.plot(msprt_values, label='likelihood')
plt.xlabel('Number of Observations')
plt.ylabel('likelihood')
plt.title('Likelihood Over Time')
plt.legend()
plt.show()
```

![png](img/Optimal%20Bayesian%20Sequential%20Hypothesis%20Testing_7_0.png)

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