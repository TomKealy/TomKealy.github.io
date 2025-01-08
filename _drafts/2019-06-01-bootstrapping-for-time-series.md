---
layout: post
title:  "Bootstrapping for Time Series"
date:   2024-10-11
categories: hypothesis-testing, multiple-testing
---

# Introduction

Remember (from basic stats) that the key to dealing with uncertainty in parameters and functionals is the **sampling distribution of estimators**. Knowing what distribution we’d get for our estimates on repeating the experiment would give us things like **standard errors**. Efron’s insight was that we can simulate replication.

After all, we’ve already fitted a model to the data, which is a guess at the mechanism that generated the data. Running that mechanism generates simulated data that, by hypothesis, has the same distribution as the real data. Feeding the simulated data through our estimator gives us one draw from the sampling distribution. Repeating this many times gives us the full sampling distribution. Since we’re using the model to estimate its own uncertainty, Efron called this **bootstrapping**. Unlike Baron Munchhausen’s attempt to get out of a swamp by pulling on his own bootstraps, this actually works.

Figure 6.1 sketches the process: fit a model to data, use the model to calculate the functional, then get the sampling distribution by generating new, synthetic data from the model and repeating the estimation on the simulated output.

To fix notation, let’s say the original data is $ x $ (in general, this is a whole data frame, not just a single number). Our parameter estimate from the data is $ \hat{\theta} $, and surrogate data sets simulated from the fitted model are $ \tilde{X}_1, \tilde{X}_2, \dots, \tilde{X}_B $.

The corresponding re-estimates of the parameters on the surrogate data are $ \tilde{\theta}_1, \tilde{\theta}_2, \dots, \tilde{\theta}_B $.

The functional of interest is estimated by the statistic $$ T $$, with a sample value $$ \hat{t} = T(x) $$, and values of the surrogates:

$$
\tilde{t}_1 = T(\tilde{X}_1), \tilde{t}_2 = T(\tilde{X}_2), \dots, \tilde{t}_B = T(\tilde{X}_B)
$$

The statistic $$ T $$ may be a direct function of the estimated parameters and only indirectly a function of $$ x $$. Everything that follows applies without modification when the functional of interest is the parameter itself or some component of the parameter.

In this section, we’ll assume that the model is correct for some value of $$ \theta $$, which we’ll call $$ \theta_0 $$, meaning we’re employing a **parametric model-based bootstrap**. The true (population or ensemble) value of the functional is likewise $$ t_0 $$.

# Bootstrapping for time series

The big picture of bootstrapping doesn’t change: simulate a distribution that’s close to the true one, repeat our estimate (or test, or whatever) on the simulation, and then examine the distribution of this statistic over many simulations. The catch is that the surrogate data from the simulation has to have the same sort of dependence as the original time series. Simple resampling is wrong (unless the data are independent), and our simulations will have to be more complicated.

### Parametric or Model-Based Bootstrap

Conceptually, the simplest case is when we fit a full, generative model—something we could step through to generate a new time series. If we are confident in the model specification, we can bootstrap by simulating from the fitted model. This is the parametric bootstrap we saw in Chapter 6.

### Block Bootstraps

Simple resampling doesn’t work because it destroys the dependence between successive values in the time series. However, there’s a clever trick due to Künsch (1989) that works and is almost as simple. Take the full time series $$ x_1:n $$ and divide it into overlapping blocks of length $$ k $$, i.e., $$ x_1:k $$, $$ x_2:k+1 $$, and so on, down to $$ x_{n-k+1}:n $$. Now, draw $$ m = n/k $$ of these blocks with replacement, and set them down in order. Call the new time series $$ \tilde{x}_{1:n} $$.

Within each block, we’ve preserved all of the dependence between observations. Successive observations are now independent, which introduces some inaccuracy since this wasn’t true of the original data. But it’s certainly better than just resampling individual observations (which would be $$ k = 1 $$). Moreover, we can make this inaccuracy smaller by letting $$ k $$ grow as $$ n $$ grows. One can show that the optimal block size is:

$$ k = O(n^{1/3}) $$

This gives a growing number $$ O(n^{2/3}) $$ of increasingly long blocks, capturing more and more of the dependence. (We’ll discuss how to pick $$ k $$ later.)

The block bootstrap scheme is extremely clever and has inspired many variants, three of which are worth mentioning:

1. **Circular Block Bootstrap**: We “wrap the time series around a circle,” so that it goes $$ x_1, x_2, \dots, x_n, x_1, x_2, \dots $$. We then sample $$ n $$ blocks of length $$ k $$ rather than just the $$ n - k $$ blocks of the simple block bootstrap. This makes better use of the information about dependence for distances less than $$ k $$.

2. **Block-of-Blocks Bootstrap**: First, divide the series into blocks of length $$ k_2 $$, then subdivide each of those into sub-blocks of length $$ k_1 < k_2 $$. To generate a new series, sample blocks with replacement and then sample sub-blocks within each block with replacement. This gives a better idea of longer-range dependence, though you have to pick two block lengths.

3. **Stationary Bootstrap**: The length of each block is random, chosen from a geometric distribution with mean $$ k $$. After choosing the sequence of block lengths, we sample the appropriate blocks with replacement. The idea here is to address a shortcoming of the regular block bootstrap, which doesn’t quite give us a stationary time series (because the distribution gets weird around block boundaries). Averaging over random block lengths fixes this. It’s slightly slower to converge than the block or circular bootstrap, but it can be helpful when the surrogate data needs to be strictly stationary.

### Sieve Bootstrap

A compromise between model-based and resampling bootstraps is the sieve bootstrap. This simulates from models, but we don’t have to fully believe in the model; we just want something easy to fit and flexible enough to capture a wide range of processes. We gradually let the model grow more complex as we get more data.

A popular choice is to use linear AR($p$) models and let $p$ grow with $n$, but there’s nothing special about AR models. They’re just easy to fit and simulate from. Additive autoregressive models, for instance, would often work just as well.

# The Maximum Entropy Bootstrap


