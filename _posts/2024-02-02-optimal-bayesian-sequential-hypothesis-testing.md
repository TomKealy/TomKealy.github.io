---
layout: post
title:  "Optimal Bayesian Sequential Hypothesis Testing"
date:   2024-02-02
categories: hypothesis-testing
---

{% newthought 'tl; dr We introduce the mSPRT' %} and give a derivation from a Bayesian point of view.<!--more--> 


### A Bayesian Model
Consider the following Bayesian model for the data:
$$Y_1|\alpha, \delta, M_1 \sim \mathcal{N}\left(\frac{1}{n_1} \alpha + \frac{1}{2} \frac{1}{n_1} \delta, \sigma^2 I_{n_1}\right)$$
$$Y_2|\alpha, \delta, M_1 \sim \mathcal{N}\left(\frac{1}{n_2} \alpha - \frac{1}{2} \frac{1}{n_2} \delta, \sigma^2 I_{n_2}\right)$$
$$\delta|M_1 \sim \mathcal{N} (0, \tau^2)$$
$$p(\alpha|M_1) \propto 1$$

Note that the prior distribution for the lift $\delta$ is equal to the mixing distribution in the mSPRT. If you are not familiar with this vectorized notation, an alternative notation is
$$y_{1i}|\alpha, \delta, M_1 \sim \mathcal{N} (\alpha + \frac{\delta}{2}, \sigma^2) \forall i = 1, \ldots, n_1$$
$$y_{2j}|\alpha, \delta, M_1 \sim \mathcal{N} (\alpha - \frac{\delta}{2}, \sigma^2) \forall j = 1, \ldots, n_2.$$

In this model for the "alternative" we have a grand mean $$\alpha$$ and a lift parameter $$\delta$$. Following a Bayesian approach, we assign $$\alpha$$ a uniform prior, and $$\delta$$ a Normal prior. Under this model, the interpretation of $$\delta$$ is still unchanged as $$E[y_1]-E[y_2]$$ i.e., the expected difference in means. Moreover, it also follows under this model that $$\bar{Y}_1 - \bar{Y}_2 \sim \mathcal{N} (\delta, \sigma^2(\frac{1}{n_1} + \frac{1}{n_2}))$$. Notice, however, that we have conditioned on this model being the alternative model $$M_1$$. The "null" model can be expressed as
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
\|Y - 1\alpha - X\delta\|^2_2 = (Y - 1\alpha - X\delta)'(Y - 1\alpha - X\delta) = (Y - 1\alpha - X\delta)'(P_1 + I - P_1)(Y - 1\alpha - X\delta) = \|P_1(Y - 1\alpha - X\delta)\|^2_2 + \|(I - P_1)(Y - 1\alpha - X\delta)\|^2_2 = n(\alpha - \bar{Y} - \bar{X}\delta)^2 + \|Y_c - X_c\delta\|^2_2,
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

Here is some code to illustrate all of this:

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu1, mu2 = 1.0, 2.0  # Means of Y1 and Y2
sigma = 0.5  # Standard deviation (same for both groups)
rho_squared = 1 / sigma**2  # Precision of observations
tau_squared = 1  # Variance of prior on delta (assumed)
n_samples = 100  # Number of samples to simulate streaming

# Generate synthetic data
np.random.seed(42)
y1_data = np.random.normal(mu1, sigma, n_samples)
y2_data = np.random.normal(mu2, sigma, n_samples)

import numpy as np

def calculate_lambda(y1, y2, sigma_squared, tau_squared):
    """Calculate Bayes Factor (Lambda) based on current data using logarithms."""
    n1, n2 = len(y1), len(y2)
    
    # Compute means
    mean_y1 = np.mean(y1)
    mean_y2 = np.mean(y2)
    
    # Compute mean difference squared
    mean_diff_squared = (mean_y1 - mean_y2) ** 2
    
    # Corrected rho_squared using logarithms for numerical stability
    log_rho_squared = np.log(1 / sigma_squared) - np.log(1/n1 + 1/n2)
    
    # Compute components of the lambda expression using logarithms
    log_numerator = log_rho_squared
    log_denominator = np.log(rho_squared + tau_squared)
    
    # Compute the exponential term separately to ensure numerical stability
    exp_term = 0.5 * (tau_squared / (np.exp(log_rho_squared) + tau_squared) * mean_diff_squared / np.exp(log_rho_squared))
    
    # Calculate log of Lambda (for numerical stability)
    log_lambda_val = 0.5 * (log_numerator - log_denominator) + exp_term
    
    # Convert log_lambda back to lambda for the probability calculation
    lambda_val = np.exp(log_lambda_val)
    
    # Probability interpretation (for illustrative purposes)
    prob_b_better = lambda_val / (lambda_val + 1)
    
    return lambda_val, prob_b_better

# Initialize storage for Lambda values
lambda_values = []
prob_values = []

# Stream data and calculate Lambda
for i in range(1, n_samples + 1):
    lambda_current, prob = calculate_lambda(y1_data[:i], y2_data[:i], rho_squared, tau_squared)
    lambda_values.append(lambda_current)
    prob_values.append(prob)

# Convert list to NumPy array for plotting
lambda_values = np.array(lambda_values)
prob_values = np.array(prob_values)
```


```python
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, label='Bayes Factor ($\Lambda$)')
plt.xlabel('Number of Observations')
plt.ylabel('Bayes Factor ($\Lambda$)')
plt.title('Evolution of Bayes Factor Over Time')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](img/Optimal%20Bayesian%20Sequential%20Hypothesis%20Testing_2_0.png)
    



```python
plt.figure(figsize=(10, 6))
plt.plot(prob_values, label='Prob')
plt.xlabel('Number of Observations')
plt.ylabel('Probability')
plt.title('Evolution of Probability Over Time')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](img/Optimal%20Bayesian%20Sequential%20Hypothesis%20Testing_3_0.png)
    



```python
plt.figure(figsize=(10, 6))
plt.scatter(y1_data, y2_data, label='Y')
plt.xlabel('$Y_{C}$')
plt.ylabel('$Y_{T}$')
plt.title('Raw Data')
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](img/Optimal%20Bayesian%20Sequential%20Hypothesis%20Testing_4_0.png)
    



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
    



    
![png](img/Optimal%20Bayesian%20Sequential%20Hypothesis%20Testing_7_1.png)
    



```python
# Assuming sigma_squared and tau_squared are defined as before
theta_0 = 0  # Under H0, the difference in means is zero

# Calculate Bayes Factor over the data stream
lambda_values = [calculate_lambda(y_data[:i], x_data[:i], 1/sigma_squared, tau_squared) for i in range(1, len(x_data) + 1)]

# Optional: Convert the list of Lambda values into a NumPy array for easier handling
import numpy as np
lambda_values_array = np.array(lambda_values)

# Plot Bayes Factor over observations
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(lambda_values_array[:, 0], label='Bayes Factor ($\Lambda$) over Observations')
plt.xlabel('Number of Observations')
plt.ylabel('Bayes Factor ($\Lambda$)')
plt.title('Bayes Factor Evolution in A/B Testing')
plt.legend()
plt.grid(True)
```


    
![png](img/Optimal%20Bayesian%20Sequential%20Hypothesis%20Testing_8_0.png)
    



```python
baseline_conversion_rate = 0.05  # Baseline conversion rate for control group
improved_conversion_rate = 0.05 + 0.05 * baseline_conversion_rate  # Improved rate for treatment group
daily_visitors = 1000  # Number of people arriving per day
total_days = 30  # Duration of the experiment in days
sigma_squared = baseline_conversion_rate * (1 - baseline_conversion_rate)  # Assuming binomial variance formula
tau_squared = sigma_squared  # Variance of the prior, can be adjusted based on prior belief

import numpy as np

np.random.seed(42)  # For reproducibility

# Generate synthetic data
control_conversions = np.random.binomial(1, baseline_conversion_rate, (total_days, daily_visitors))
treatment_conversions = np.random.binomial(1, improved_conversion_rate, (total_days, daily_visitors))
```


```python
treatment_conversions
```




    array([[0, 0, 1, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 1, 0],
           [0, 0, 1, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]])




```python
lambda_vals = []
prob_b_better_vals = []

for day in range(total_days):
    # Cumulative data up to the current day
    cumulative_control = control_conversions[:day+1].flatten()
    cumulative_treatment = treatment_conversions[:day+1].flatten()
    
    # Calculate Lambda and probability B is better for the current cumulative data
    lambda_val, prob_b_better = calculate_lambda(cumulative_treatment, cumulative_control, sigma_squared, tau_squared)
    lambda_vals.append(lambda_val)
    prob_b_better_vals.append(prob_b_better)
    
    print(f"Day {day+1}: Lambda = {lambda_val:.4f}, Probability B is better = {prob_b_better:.4f}")

```

    Day 1: Lambda = 1.0000, Probability B is better = 0.5000
    Day 2: Lambda = 1.0000, Probability B is better = 0.5000
    Day 3: Lambda = 1.0000, Probability B is better = 0.5000
    Day 4: Lambda = 1.0000, Probability B is better = 0.5000
    Day 5: Lambda = 1.0000, Probability B is better = 0.5000
    Day 6: Lambda = 1.0000, Probability B is better = 0.5000
    Day 7: Lambda = 1.0000, Probability B is better = 0.5000
    Day 8: Lambda = 1.0000, Probability B is better = 0.5000
    Day 9: Lambda = 1.0000, Probability B is better = 0.5000
    Day 10: Lambda = 1.0000, Probability B is better = 0.5000
    Day 11: Lambda = 1.0000, Probability B is better = 0.5000
    Day 12: Lambda = 1.0000, Probability B is better = 0.5000
    Day 13: Lambda = 1.0000, Probability B is better = 0.5000
    Day 14: Lambda = 1.0000, Probability B is better = 0.5000
    Day 15: Lambda = 1.0000, Probability B is better = 0.5000
    Day 16: Lambda = 1.0000, Probability B is better = 0.5000
    Day 17: Lambda = 1.0000, Probability B is better = 0.5000
    Day 18: Lambda = 1.0000, Probability B is better = 0.5000
    Day 19: Lambda = 1.0000, Probability B is better = 0.5000
    Day 20: Lambda = 1.0000, Probability B is better = 0.5000
    Day 21: Lambda = 1.0000, Probability B is better = 0.5000
    Day 22: Lambda = 1.0000, Probability B is better = 0.5000
    Day 23: Lambda = 1.0000, Probability B is better = 0.5000
    Day 24: Lambda = 1.0000, Probability B is better = 0.5000
    Day 25: Lambda = 1.0000, Probability B is better = 0.5000
    Day 26: Lambda = 1.0000, Probability B is better = 0.5000
    Day 27: Lambda = 1.0000, Probability B is better = 0.5000
    Day 28: Lambda = 1.0000, Probability B is better = 0.5000
    Day 29: Lambda = 1.0000, Probability B is better = 0.5000
    Day 30: Lambda = 1.0000, Probability B is better = 0.5000



```python
import matplotlib.pyplot as plt

# Plot Lambda values over time
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(range(1, total_days + 1), lambda_vals, marker='o', linestyle='-')
plt.title('Bayes Factor Over Time')
plt.xlabel('Day')
plt.ylabel('Bayes Factor ($\Lambda$)')

# Plot Probability B is better values over time
plt.subplot(1, 2, 2)
plt.plot(range(1, total_days + 1), prob_b_better_vals, marker='o', linestyle='-', color='orange')
plt.title('Probability B is Better Than A Over Time')
plt.xlabel('Day')
plt.ylabel('Probability B is Better')

plt.tight_layout()
plt.show()

```


    
![png](img/Optimal%20Bayesian%20Sequential%20Hypothesis%20Testing_12_0.png)
    



```python
def compute_mSPRT_binomial(y_data, x_data, baseline_rate, improved_rate, daily_visitors):
    """Compute mSPRT for binomial data, assuming known conversion rates."""
    n = len(y_data)  # Number of days so far
    sigma_squared = baseline_rate * (1 - baseline_rate) / daily_visitors  # Variance of binomial distribution per visitor
    tau_squared = sigma_squared  # Assuming the same variance for the prior for simplicity
    
    # Conversion rate differences
    y_rate = np.mean(y_data)
    x_rate = np.mean(x_data)
    
    # mSPRT calculation adapted for binomial data
    theta_0 = 0  # Null hypothesis: no difference
    lambda_n = np.sqrt(2 * sigma_squared / (2 * sigma_squared + n * tau_squared)) * \
               np.exp(n * tau_squared * (y_rate - x_rate - theta_0) ** 2 / (4 * sigma_squared * (2 * sigma_squared + n * tau_squared)))
    return lambda_n

```


```python
msprt_values = []

for day in range(1, total_days + 1):
    # Cumulative data up to the current day
    cumulative_control = control_conversions[:day].flatten()
    cumulative_treatment = treatment_conversions[:day].flatten()
    
    # Compute mSPRT for the cumulative data
    msprt_value = compute_mSPRT_binomial(cumulative_treatment, cumulative_control, baseline_conversion_rate, improved_conversion_rate, daily_visitors * day)
    msprt_values.append(msprt_value)

# Plot mSPRT values over time
plt.figure(figsize=(10, 6))
plt.plot(range(1, total_days + 1), msprt_values, marker='o', linestyle='-', color='green')
plt.title('mSPRT Values Over Time for A/B Testing')
plt.xlabel('Day')
plt.ylabel('mSPRT Value')
plt.grid(True)
plt.show()

```


    
![png](img/Optimal%20Bayesian%20Sequential%20Hypothesis%20Testing_14_0.png)
    



```python

```
