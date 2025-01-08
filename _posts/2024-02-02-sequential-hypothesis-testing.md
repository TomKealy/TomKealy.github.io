---
layout: default
title:  "Sequential Hypothesis Testing"
date:   2024-02-02
categories: hypothesis-testing
---
We introduce the mSPRT and give a derivation from a Bayesian point of view.

# Sequential Testing

Normally when you go about hypothesis testing, after a random sample is observed one of two possible actions are taken: accept the null hypothesis $$H_0$$, or accepct the tive hypothesis $$H_1$$. In some cases the evidence may strongly support one of the hypotheses, whilst in other cases the evidence may be less convincing. Nevertheless, a decision must he made. All this assumes that all the data has been collected, and no more is available. It doesn't have to be this way. There is a class of hypothesis tests where you can safely collect more data when the evidence is ambiguous. Such a test typically continues until the evidence strongly favors one of the two hypotheses.

In case of simple hypotheses the strength of the evidence for $$H$$ is given by the ratio of the probability of the data under $$H_0$$, to the probability of the data under $$H_1$$. We denote this likelihood ratio by $$\Lambda$$. The Neyman-Pearson lemma implies that for a given amount of information the likelihood ratio test is the most powerful test. Such a rule decides to accept $$H_1$$ if $$\Lambda$$ is big enough, and decides to accept $$H_0$$ otherwise. How big $$\Lambda$$ must get to lead to the decicion $$H_1$$, danends on, amond other things, its sampling distribution under $$H_0$$ and $$H_1$$. It is not unusual for "big enough" to mean $$\Lambda \geq 1$$. Such tests could easily decide $$H_0$$ or $$H_1$$ when the actual evidence is neutral.

## SPRT

In 1943 Wald proposed the *sequential probability ratio test* (SPRT). Suppose that $$Y$$ is a random variable with unknown distribution $$f$$. We want to test the following hypotheses:

* $$H_0: f=f_0$$
* $$H_1: f=f_1$$

where $$f_0$$ and $$f_1$$ are specified. We observe values of $$Y$$ successively: $$y_1, y_2, \ldots$$ the random variables $$Y_i$$ corresponding to the $$y_i$$ are i.i.d with common distribution $$f$$. Let

$$
\Lambda = \prod_{i=1}^n \frac{f_1\left(x_i\right)}{f_0\left(x_i\right)}
$$

be the likelihood ratio at stage $$n$$. We choose two decision boundaries $$A$$ and $$B$$ such that $$0 < B < A < \infty $$,  we accept $$H_0$$ if $$\Lambda \leq B$$ and $$H_1$$ if $$\Lambda \geq A$$, and we continue if $$B \leq \Lambda \leq A$$. The constants $$A$$ and $$B$$ are determined by the desired false positive and false negative rates of the experimenter. In general it can  be shown that the boundaries A and B can be calculated as with very good approximation as

$$A=\log\left(\frac{\beta}{1=\alpha}\right)$$

$$B=\log\left(\frac{1-\beta}{\alpha}\right)$$

so the SPRT is really very simple to apply in practice.

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

|Distribution | $$ f\left(x; \theta\right) $$  |  $$ C\left(\theta\right) $$  |   $$ D\left(\theta\right) $$   |
|:----------------|----:|-----:|-------:|
|Normal        |$$ \frac{1}{\sqrt{2\pi}}e^\left(-\left(x-\theta\right)^2/2\right) $$ | $$\theta$$ |$$\frac{\theta^2}{2} $$ |
|Bernoulli    |$$ \theta^{x}\left(1-\theta\right)^{(1-x)} $$   | $$ \log{\left(\frac{\theta}{1-\theta}\right)} $$     | $$ -\log{\left(1-\theta\right)} $$|
|Exponential    |$$ \frac{1}{\theta}e^-x/\theta $$   | $$ \log{\theta}$$     | $$ \theta $$|
|Poisson    |$$ \frac{e^-\theta \theta^x}{x!} $$   | $$ -\frac{1}{\theta}$$     | $$ \log{\theta} $$|

### Example: Exponential Distribution

A textbook example is parameter estimation of a probability distribution function. Consider the exponential distribution:

$$ f\left(x; \theta\right) = \frac{1}{\theta}e^-x/\theta $$

The hypotheses are

* $$H_0: \theta = \theta_0 $$
* $$H_1: \theta = \theta_1 $$

Where $$ \theta_1 > \theta_1 $$.

Then the log-likelihood function (LLF) for one sample is

$$
\begin{aligned}
\log{\Lambda\left(x\right)} &= \log{ \frac{1}{\theta_1}e^-x/\theta_1  }{ \frac{1}{\theta_0}e^-x/\theta_0  } 
&= -\log{\frac{\theta_1}{\theta_0}} + \frac{\theta_1 - \theta_0}{\theta_1\theta_0}x
\end{aligned}
$$

The cumulative sum of the LLFs for all x is

$$ S_n = \sum_n \log{\Lambda\left(x_i\right)} = -n\log{\frac{\theta_1}{\theta_0}} + \frac{\theta_1 - \theta_0}{\theta_1\theta_0}\sum_n x_i  $$

Accordingly, the stopping rule is:

$$ a <  -n\log{\frac{\theta_1}{\theta_0}} + \frac{\theta_1 - \theta_0}{\theta_1\theta_0}\sum_n x_i < b $$

After re-arranging we finally find

$$ a + n\log{\frac{\theta_1}{\theta_0}} < \frac{\theta_1 - \theta_0}{\theta_1\theta_0}\sum_n x_i < b + n\log{\frac{\theta_1}{\theta_0}} $$

The thresholds are simply two parallel lines with slope $$ \log{\frac{\theta_1}{\theta_0}}$$. Sampling should stop when the sum of the samples makes an excursion outside the continue-sampling region.

### Example: Binomial Distribution


### 1. Likelihood Ratio for Bernoulli Distribution

For a Bernoulli trial, $$ X \in \{0, 1\} $$ with probability $$ \theta $$ for success ($$ \theta = 1 $$) and $$ 1 - \theta $$ for failure ($$ \theta = 0 $$).

- Under hypothesis $$ H_1 $$ with parameter $$ \theta_1 $$:  
  $$
  P(X = x | \theta_1) = \theta_1^x (1 - \theta_1)^{1 - x}
  $$

- Under hypothesis $$ H_0 $$ with parameter $$ \theta_0 $$:  
  $$
  P(X = x | \theta_0) = \theta_0^x (1 - \theta_0)^{1 - x}
  $$

The likelihood ratio is:
$$
\Lambda(x) = \frac{P(X = x | \theta_1)}{P(X = x | \theta_0)} = \frac{\theta_1^x (1 - \theta_1)^{1 - x}}{\theta_0^x (1 - \theta_0)^{1 - x}}
$$


### 2. Log-Likelihood Ratio

The log-likelihood ratio is:
$$
\log \Lambda(x) = \log \left( \frac{\theta_1^x (1 - \theta_1)^{1 - x}}{\theta_0^x (1 - \theta_0)^{1 - x}} \right)
$$

Breaking this into cases for $$ x = 0 $$ and $$ x = 1 $$:

If $$ x = 1 $$:  

  $$
  \log \Lambda(1) = \log \frac{\theta_1}{\theta_0}
  $$

If $$ x = 0 $$:  

  $$
  \log \Lambda(0) = \log \frac{1 - \theta_1}{1 - \theta_0}
  $$

Thus, the general form for $$ x $$ is:

$$
\log \Lambda(x) = x \log \frac{\theta_1}{\theta_0} + (1 - x) \log \frac{1 - \theta_1}{1 - \theta_0}
$$


### 3. Cumulative Log-Likelihood for $$ n $$ Observations

Now we calculate the cumulative log-likelihood ratio for \( n \) independent observations \( x_1, x_2, \ldots, x_n \):
$$
S_n = \log \frac{\theta_1}{\theta_0} \sum_{i=1}^{n} x_i + \log \frac{1 - \theta_1}{1 - \theta_0} \sum_{i=1}^{n} (1 - x_i)
$$


This simplifies to:
\[
S_n = \log \frac{\theta_1}{\theta_0} \sum_{i=1}^{n} x_i + \log \frac{1 - \theta_1}{1 - \theta_0} \sum_{i=1}^{n} (1 - x_i)
\]

### 4. Stopping Rule

The stopping rule in this context is when the cumulative sum \( S_n \) crosses specific thresholds. The inequality becomes:
$$
a < \log \frac{\theta_1}{\theta_0} \sum_{i=1}^{n} x_i + \log \frac{1 - \theta_1}{1 - \theta_0} \sum_{i=1}^{n} (1 - x_i) < b
$$

Rearranging this gives:
$$
a - n \log \frac{1 - \theta_1}{1 - \theta_0} < \log \frac{\theta_1}{\theta_0} \sum_{i=1}^{n} x_i < b - n \log \frac{1 - \theta_1}{1 - \theta_0}
$$


### 5. Interpretation

In this case, the thresholds are two parallel lines with slopes based on the log-ratios $$ \log \frac{\theta_1}{\theta_0} $$ and $$ \log \frac{1 - \theta_1}{1 - \theta_0} $$. The sampling process stops when the sum of the observed successes $$ \sum x_i $$ exits the continuation region bounded by these two lines.

We can summarise this in the following algorithm:

Set $$LR \leftarrow 1$$ and $$j \leftarrow 0$$

1. Increment $$j$$
2. If $$𝑋_j=1$$ then set $$LR \leftarrow LR \frac{\theta_1}{\theta_0} $$
3. If $$𝑋_j=0$$ then set $$ LR \leftarrow LR \frac{(1−\theta_1)}{(1−\theta_0)}$$

What’s $$LR$$ at stage 𝑚? Let $$ T_m≡ \sum_{j=1}^m 𝑋_j$$ then:

$$
\frac{\theta_{1𝑚}}{\theta_{0m}} = \frac{\theta_1^{𝑇_𝑚}(1−\theta_1)^{𝑚−𝑇_𝑚}}{\theta_0^{𝑇_𝑚}(1−\theta_0)^{𝑚−𝑇_𝑚}}
$$

This is the ratio of binomial probability when $$\theta=\theta_1$$ to binomial probability when $$\theta=\theta_0$$ (the binomial coefficients in the numerator and denominator cancel). It simplifies further to

$$
\frac{\theta_{1𝑚}}{\theta_{0𝑚}}=\left(\frac{(\theta_0}{\theta_1}\right)^{𝑇_m}\left(\frac{1−\theta_0}/{1−\theta_1})\right)^{𝑚−𝑇_𝑚}
$$

We conclude $$\theta > \theta_0$$ if

$$
\frac{\theta_1m}{\theta_0m} \geq \frac{1-\beta}{\alpha}
$$

And we conclude $$\theta < \theta_0$$ if

$$
\frac{\theta_1m}{\theta_0m} \geq \frac{\beta}{1-\alpha}
$$

Otherwise, we draw again.

The SPRT approximately minimizes the expected sample size when $$\theta > \theta_0$$ or $$\theta > \theta_1$$. For values in $$\left(\theta_1,\theta_0\right)$$, it can have larger sample sizes than fixed-sample-size tests.

### Example: Normal Distribution

Consider the Normal distribution:

$$ f\left(x; \theta\right) = \frac{1}{\sqrt{2\pi}}e^\left(-\left(x-\theta\right)^2/2\right) $$

The hypotheses are

* $$H_0: \theta = \theta_0 $$
* $$H_1: \theta = \theta_1 $$

Where $$ \theta_1 > \theta_1 $$.

Then the log-likelihood function (LLF) for one sample is

$$
\begin{aligned}
\log{\Lambda\left(x\right)} &= \log{ \frac{1}{\sqrt{2\pi}}e^\left(-\left(x-\theta_1\right)^2/2\right)  }{ \\frac{1}{\sqrt{2\pi}}e^\left(-\left(x-\theta_0\right)^2/2\right)  } 
&= -\frac{1}{2}\log{\frac{\theta_1}{\theta_0}} + \frac{\theta_1 - \theta_0}{2\theta_1\theta_0}x_i^2 - \frac{\theta_1 - \theta_0}{2}
\end{aligned}
$$

The cumulative sum of the LLFs for all x is

$$ S_n = \sum_n \log{\Lambda\left(x_i\right)} = -n\log{\frac{\theta_1}{\theta_0}} + \frac{\theta_1 - \theta_0}{\theta_1\theta_0}\sum_n x_i  $$

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

Considering normal data, for any $$\mu_A$$ and $$\mu_B$$, the difference $$Z_n = Y_n - X_n$$ follows a normal distribution $$\sim N(\theta, 2\sigma^2)$$. We can apply the one-variation mSPRT to the sequence $$\{Z_n\}$$, leading to the following definition:

$$
\sqrt{\frac{2\sigma^2}{2\sigma^2 + n\tau^2}} \exp{\left(\frac{n^2\tau^2\left(X_n - Y_n - \theta_0\right)^2}{4\sigma^2\left(2\sigma^2 + n\tau^2\right)}\right)}.
$$

Here's a small simulation:

```python
def compute_mSPRT(y_data, x_data, sigma_squared, tau_squared, theta_0):
    n = len(y_data)
    y_mean = np.mean(y_data)
    x_mean = np.mean(x_data)
    lambda_n = np.sqrt(2 * sigma_squared / (2 * sigma_squared + n * tau_squared)) * \
               np.exp(n * n * tau_squared * (y_mean - x_mean - theta_0) ** 2 / (4 * sigma_squared * (2 * sigma_squared + n * tau_squared)))
    return lambda_n
```

```python
# Parameters for synthetic data generation
mu_X = 0
mu_Y = 3  
sigma = 1.0  
n_samples = 100  

np.random.seed(42)
x_data = np.random.normal(mu_X, sigma, n_samples)
y_data = np.random.normal(mu_Y, sigma, n_samples)
```

```python
sigma_squared = sigma ** 2
tau_squared = 1  
theta_0 = 0  

msprt_values = []
for n in range(1, n_samples + 1):
    lambda_n = compute_mSPRT(y_data[:n], x_data[:n], sigma_squared, tau_squared, theta_0)
    msprt_values.append(lambda_n)

p_values = [1]
for i, lambda_val in enumerate(msprt_values):
    p_values.append(min(p_values[i-1], 1 / lambda_val))

plt.figure(figsize=(10, 6))
plt.plot(p_values[1:], label='Always Valid p-values')
plt.xlabel('Number of Observations')
plt.ylabel('p-value')
plt.title('Always Valid p-values Over Time')
plt.legend()
plt.show()

plt.plot(msprt_values, label='likelihood')
plt.xlabel('Number of Observations')
plt.ylabel('likelihood')
plt.title('Likelihood Over Time')
plt.legend()
plt.show()
```