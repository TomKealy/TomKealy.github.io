---
layout: default
title:  "A short introduction to Martingales"
subtitle: Calculations for 
date:   2021-03-10
categories: time-series
---

# Introduction

Martingales are elegant and powerful tools to study sequences of dependent random variables. It is originated from gambling, where a gambler can adjust the bet according to the previous results.

Martingales are one of the more abstract parts of probability. Which is a pity, since they are also one of the most useful bits of probability. Martingales are one of the simplest models of dependent random variables, and that is why they come up in examples over and over again. If you can identify when a stocastic process as a martingale, then you know some rather general things about that process that may lead you to being able to get specific answers to your questions.

If you have a stocastic process whos expected value is time-dependent, then you can potentially convert your process into a martingale. All you need to do is to add the proper conditional expected value. Then you can apply all sorts of martingal theorems (convergence, optional stopping etc) which can lead to explict numerical answers. However approaching martingales with the attitude that you __will__ calculate things will leave you dissapointed. There are very few explicit examples.

You should approach your problem from the point of view that "process $$X_t$$ is a martingale, so any stopping strategy will not affect its expected value". This realization is useful for an applied as it stops you ways wasting time optimising the expected value.

## The maths, unfortunately

We denote our [sample space](https://en.wikipedia.org/wiki/Probability_space) as $S$ (for example $S = {+1, -1}$). Let $F_n$ be a filtration of the event space. 

Let $X_0, X_1, X_2, \dots$ be a sequence of random variables. We will imagine that we are acquiring information about $S$ in stages. The random variable $X_n$ is what we know at stage $n$. If $Z$ is any random variable, let

$$
E\left[Z \mid F_nn\right]
$$

denote the conditional expectation of $Z$ given all the information that is available to us on the $n$th stage. If you saw all the information you could obtain by stage $n$, and you made a Bayesian update to your probability distribution on $S$ in light of this information, then $E\left[Z\mid F_n\right]$ would represent the expected value of $Z$ with respect to this revised probability.

If we don’t specify otherwise, we assume that the information available at stage n consists precisely of the values $X_0, X_1, X_2, \dots$ so that

$$
E\left[Z \mid F_nn\right] = E\left[Z \mid X_0, X_1, X_2, \dots, X_n].
$$

However in some applications, one could imagine there are other things known as well at stage $n$. For example, maybe $X_n$ represents the price of an asset on the $n$th day and $Y_n$ represents the price of asset $Y$ on the $n$th day. If you have access to the sequence $X_0, X_1, X_2, \dots, X_n]$ and the sequence $Y_0, Y_1, Y_2, \dots, Y_n]$. Then $E\left[Z \mid F_nn\right]$ would be our revised expectation of $Z$ after we have incorporated what we know about both sequences.

We say that sequence $X_n$ is a martingale if:

1. $E\left[\mid X_n \mid\right] < \infty$ for all $n$.
2. $E\left[X_{n+1}|F_n\right] = X_n$  for all $n$.

Informally $X_0, X_1, \ldots$ is a martingale if the following is true: taking into account all the information I have at stage $n$, the conditional expected value of $X_{n+1}$ is just $X_n$. Basically, if your process is a martingale, you can't know anything about the future other that everything you know in the present moment (and the entire history of the process).

To motivate this definition, imagine that $X_n$ represents the price of a stock on day $n$. In this context, the martingale condition states informally that “The expected value of the stock tomorrow, given all I know today, is the value of the stock today.” After all, if the stock price today were 50 and I expected it to be 60 tomorrow, then I would have an easy way to make money in expectation (buy today, sell tomorrow). But if the public had the same information I had, then other investors would also try to cash in on this by buying the stock today at 50, and people holding the stock would be reluctant to sell for 50. Indeed, we’d expect the price to be quickly bid up to about 60 today.

### Example

Let $S = {+1, -1}$ and let $X_0, X_1, X_2, \dots$ be a sequence of random variables taking values in $S$. $X_n$ is +1 with probability 0.5 and -1 with probability 0.5 We define:

$$
M_n = \sum_{i=0}^n X_n
$$

$M_n$ is a martingale since:

$$
E\left[M_{n+1} \mid F_n\right] = E\left[M_{n} + X_{n+1} \mid F_n\right]  = E\left[M_{n}\mid F_n\right] + E\left[X_{n+1} \mid F_n\right]
$$

Since $M_n$ is known at stage n, we have $ E\left[M_{n}\mid F_n\right] = M_n$. Since we know nothing more about $X_{n+1}$ at stage $n$ than we originally knew, we have $E\left[X_{n+1} \mid F_n\right] =0$. So

$$ 
E\left[M_{n+1} \mid F_n\right] = M_n
$$

So the sequence $M_n$ is a martingale.

## Stopping times, and the Optional Stopping theorem

A **stopping time** is a (non-negative integer-valued) random variable, $T$, such that for all $n$ the event that $T = n$ depends only on the information available to us at time $n$. Stopping times can be interprete as the time that you sell an asset, given that the sequence of prices is $X_0, X_1, X_2, \dots$

Saying that $T$ is a stopping time means that the decision to sell at time $n$ depends only the information we have up to time $n$, and not on future prices. Specifying a stopping time can be interpreted as specifying a strategy for deciding when to sell the asset.

For example, let $X_0, X_1, X_2, \dots$ be i.i.d. random variables equal to −1 with probability .5 and 1 with probability .5 and let $X_0 = 0$ and $M_n = \sum_{i=0}^n X_n $ for $n ≥ 0$. These four statements:

1. The smallest $T$ for which $\mid X_T \mid = 50$
2. The smallest $T$ for which $X_tT \in {−30, 100}$
3. The smallest $T$ for which $X_T = 17$.
4. The $T$ at which the $X_n$ sequence achieves the value 17 for the 9th time.

all define stopping times.

### Optional Stopping Theorem

The optional stopping theorem says that the expected value of a martingale at a stopping time is equal to its initial expected value. It tells us that you can’t make money (in expectation) by buying and selling an asset whose price is a martingale. Precisely, if you buy the asset at some time and adopt any strategy at all for deciding when to sell it, then the expected price at te time you sell is the price you originally paid. In other words, if the market price is a martingale, you cannot make money in expectation by “timing the market.”

**Definition** Boundedness. We say a random sequence $X_0, X_1, X_2, \dots$ is **bounded** when there exists some $C > 0$, we have that with probability one $\mid X_n \mid ≤ C$ for all $n \geq 0$. A stopping time $T$ is bounded if there exists some $C > 0$ such that $T \geq C$ with probability one.

**Optional Stopping Theorem (first version)**: Suppose that $X_$ is a known constant, that $X_0, X_1, X_2, \dots$ is a bounded martingale, and that $T$ is a stopping time. Then $E\left[X_T\right] = X_0$.

**Optional Stopping Theorem (second version):** Suppose that $X_$ is a known constant, that $X_0, X_1, X_2, \dots$ is a martingale, and that $T$ is a bounded stopping time. Then $E\left[X_T\right] = X_0$.

These boundedness assumptions are actually very important. Without them the theorem would not be true. 

For a counterexample, recall that if $X_0 = 0$ and $X_n$ goes up or down by 1 at each time step (each with probability .5) then $X_0, X_1, X_2, \dots$ is a martingale. If we let $T$ be the first $n$ for which $X_n = 100$, then $T$ is a finite number with probability one. (That is, with probability one $X_n$ reaches T eventually.) But then $X_T$ is always 100, which means that $E\left[X_T\right]  \neq X_0$.