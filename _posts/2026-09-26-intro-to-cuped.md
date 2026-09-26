---
layout: default
section: stats
title: CUPED — making your A/B tests more sensitive
subtitle: How pre-experiment data can help you detect smaller effects with fewer users.
categories: statistics, experimentation
bibliography: references.bib
toc: true
---

# Introduction

Every A/B test is a signal-to-noise problem. The signal is the effect of your intervention: the lift in conversion rate, the change in revenue per customer. The noise is everything else: the natural variation in how people behave, regardless of what you showed them.

When noise is high relative to the signal, you need more data to be confident that what you're seeing is real. This creates a familiar constraint: larger sample sizes mean longer experiments, and longer experiments mean slower decisions.

CUPED — Controlled-experiment Using Pre-Experiment Data — is a technique for reducing noise without collecting more data {% cite deng2013improving %}. The core idea is that a lot of the variation you observe during an experiment is predictable from how users were already behaving before it started. If you can subtract that predictable component, what's left is a cleaner signal.

# The problem with variance

To understand why variance matters, consider a simple example. Suppose you're testing a new checkout flow and you expect it to increase revenue per customer by €0.50. Your users vary a lot — some spend €5 per order, some spend €150. That variation exists completely independently of your experiment.

When you run a t-test on the results, you're comparing the means of two groups. The wider the spread of values in each group, the harder it is to tell whether a difference in means is real or just noise. This is why experiments with high-variance metrics — revenue, order value, any metric with a long tail — tend to require large sample sizes.

The formal measure of this is the standard error of your estimate, which scales with the standard deviation of your metric divided by the square root of your sample size:

$$SE = \frac{\sigma}{\sqrt{n}}$$

You can reduce the standard error two ways: increase $$n$$ (more data) or decrease $$\sigma$$ (less noise). CUPED is a way to do the second.

# The intuition behind CUPED

Consider a bookworm who buys several books every month regardless of any experiment. If they are in the test group, their high spending makes the test group look better than it is. If they are in the control group, the reverse. Either way, their habitual behaviour is noise — it obscures the true effect of whatever you're testing.

Before the experiment started, you already had data on how this person behaves. That history is a good predictor of how they'll behave during the experiment, independent of the treatment. CUPED uses that pre-experiment history as a covariate — a piece of information that helps explain variation in the outcome — and adjusts the results accordingly.

The adjusted estimator is:

$$\hat{Y}_{CUPED} = \bar{Y} - \theta(\bar{X} - \mathbb{E}[X])$$

Where:

- $$\bar{Y}$$ is the observed mean outcome during the experiment
- $$X$$ is the pre-experiment covariate (typically the same metric, measured
  before the experiment)
- $$\mathbb{E}[X]$$ is the expected value of $$X$$ (in practice, its overall
  mean)
- $$\theta$$ is chosen to minimise variance, and works out to
  $$\frac{Cov(Y, X)}{Var(X)}$$

The term $$\theta(\bar{X} - \mathbb{E}[X])$$ removes the part of the outcome that was predictable from pre-experiment behaviour. What remains has lower variance.

# How much variance does it remove?

The variance of the CUPED estimator is:

$$Var(\hat{Y}_{CUPED}) = Var(\bar{Y})(1 - \rho^2)$$

Where $$\rho$$ is the correlation between the pre-experiment covariate and the experiment outcome. The higher the correlation, the more variance CUPED removes.

This has a direct practical implication. If your pre- and post-experiment metrics are correlated at $$\rho = 0.7$$, the variance reduction is $$1 - 0.7^2 = 51\%$$. A 51% reduction in variance means you need roughly half as many users to achieve the same statistical power — or, equivalently, you can detect a smaller effect with the same number of users.

| Correlation ($$\rho$$) | Variance reduction | Sample size saving |
|---|---|---|
| 0.3 | 9% | ~9% |
| 0.5 | 25% | ~25% |
| 0.7 | 51% | ~51% |
| 0.9 | 19% | ~81% |

At HelloFresh, revenue metrics often have relatively high week-to-week correlation for returning customers, which makes CUPED particularly effective.

# Choosing your covariate

The covariate does not have to be the same metric you are measuring. Any variable that correlates with your outcome and was measured before the experiment can be used. In practice, the pre-experiment value of the same metric is almost always the best choice: it tends to have the highest correlation, and it is straightforward to justify.

A few principles for covariate selection:

**The covariate must be measured before the experiment starts.** Using a variable measured during the experiment risks contamination — the covariate itself could be affected by the treatment, which would corrupt the adjustment.

**Higher correlation means more variance reduction.** If you have multiple candidate covariates, the one most correlated with your outcome metric will give the greatest benefit. You can also combine multiple covariates using regression adjustment, though the marginal gains beyond the single best covariate are usually small.

**The covariate must be independent of the treatment assignment.** Pre- experiment data satisfies this by definition, as long as it is collected before randomisation.

# Choosing the lookback window

The lookback window — how far back you go to calculate the pre-experiment covariate — affects the correlation and therefore the variance reduction.

A window that is too short may not reflect stable behaviour, particularly for metrics that are bursty or seasonal. A window that is too long may include periods where user behaviour was structurally different from the present, reducing the predictive power of the covariate.

In practice, a lookback of the same duration as the experiment is a reasonable starting point. For subscription businesses or contexts with strong loyalty patterns, the average customer lifetime or the average repurchase cycle is a natural anchor. A window of four to eight weeks tends to capture stable purchasing behaviour without reaching back into seasonally different periods.

# Limitations

CUPED requires pre-experiment data. For new users who have no prior history — a common case for acquisition experiments — no covariate is available, and CUPED cannot be applied. In these situations, stratified randomisation or other variance reduction techniques are more appropriate.

The quality of the pre-experiment data matters. If the covariate is measured with error, or if the lookback period includes anomalies (a major promotional event, a platform outage), the adjustment can be noisy or biased. Garbage in, garbage out.

Finally, CUPED assumes a linear relationship between the covariate and the outcome. In most practical cases this is a reasonable approximation, but where the relationship is strongly non-linear, more flexible adjustment methods — such as MLRATE or doubly robust estimators — may perform better.

# Summary

CUPED reduces the variance of your experiment estimates by removing the component of variation that is predictable from pre-experiment behaviour. The degree of variance reduction is determined by the correlation between your covariate and your outcome metric. In practical terms, this translates into either faster experiments (smaller required sample sizes) or more sensitive ones (ability to detect smaller effects).

It is not a silver bullet — it requires pre-experiment data, is most effective for returning users, and depends on a linear relationship between covariate and outcome. But for metric-heavy experimentation platforms where reducing sample size requirements has real business value, it is one of the most impactful tools available.

# References
{% bibliography --cited %}