---
layout: default
title:  "AB testing Google Ads Bidding strategies"
date:   2025-01-15
categories: hypothesis-testing
---

# Introduction

Google (and Meta, and Twitter) serves ads to users browsing the internet. For example, when you make a Google search, you will see sponsored content at the top of your search results. Similarly if you brose Instagram, some of the images you see will be products to buy. The ads you see when you are browsing the internet are paid for by companies who want you to buy their products. The companies bid, in automated auctions, for their ads to be served to you as you browse.

A **bid** is the amount of money you are willing to spend for a single click on a specific keyword in the Google Ads campaign. Those bids determine where your ads appear in search results. Bidders compete for the highest possible position in Google search rankings for the specific search query. Every time someone does a search in Google, the system runs all offers through Google’s algorithm and ranks the ads according to how much particular advertisers are willing to pay for that term as well as what is its quality score. In short, it is the auction that determines which ads will appear at that moment in that space and your bid puts you in the auction.

Google and Meta provide several ways to control the outcome of each auction: interested parties need to specify a target Return on Ad Spend (tROAS). This is the target ammount that you wish to make on each ad. For example, if a business spends 200 USD and this generates 1,000 USD in revenue (or conversion value), then the ROAS is 500%. Setting a tROAS of 500% says that the business needs to make 5 dollars for every dollar that’s spent on Google Ads.

Alternatively a company could control a target Cost per Acquisition (tCPA). Choosing a tCP Abidding strategy will generate the maximum number of conversions using a campaign’s budget. It will also account for the desired target cost per action, which is chosen by the advertiser. For example, if an advertiser has chosen a 20 USD Target CPA, Google will adjust bids to drive conversions and aim to achieve that tCPA.

So you can choose ROAS if you want the maximum conversion value per conversions for your ad budget. You can choose tCPA if you want the maximum number of conversion for a given ad budget.

Companies who bid for these ad spaces are interested in optimising their **ad spend.** That is would a 'set and forget' strategy—where you set a _fixed_ tROAS value and leave it throughout a campaign—be outperformed by an adaptive strategy, where the tROAS is adjusted (possibly by an ML model) during the campaign. This is useful, because maybe you set the tCPA to bid 50 USD per conversion, buy you could have gotten away with paying 40 USE instead. A 'set and forget it' approach would therefore pay 10 USD extra, and over 1000s of conversions this strategy can cost millions extra over the course of a campaign.

One way to decide whether either strategy is better, is to conduct an A/B test. The challenge is that we cannot directly control the amount of money each strategy spends. The target determines the total spend, which in turn affects the number of conversions and ultimately, the return on investment (ROI). The only aspect I can control is the duration of the experiment. That is, conditioned on the history of outcomes, do I continue for another day?

Our goal is to determine whether the adaptive strategy can outperform the fixed strategy by at least 10\% in terms of ROI.

Here are a few key questions I have:

1. What is the appropriate randomisation unit in this case? We can't be certain that the dollar ammounts or the number of conversions are at all comparible between experiment variations.
2. Given the uncertainty over the randomisation unit, is the simple total profit (revenue - spend) an appropriate metric to determine the outcome of the experiment? What is an appropriate method for statistical inference in this case.
3. Sample size & duration: How do I determine the appropriate sample size or experiment duration to detect a 10% uplift in ROI with statistical significance? Is it even correct to think of randomisation units being days?
4. Variance estimation: Given that spend is not directly controlled but influenced by target CPA, what metrics or techniques should I use to accurately estimate the variance in ROI across both groups?
5. Bias mitigation: How can I account for differences in spend or conversion patterns that might create bias, especially if one strategy adapts faster than the other?
6. Stopping rules: Since I can only control how long the experiment runs, what would be appropriate stopping criteria or statistical tests to apply during and after the experiment to ensure the results are reliable?

# Data

Google and Facebook collect data about each ad which is served to you. So, the data is at the level of the impression. I.e. each time somebody sees see an ad, that is recorded as a separate event. This means that a single person could see the same ad multiple times before clicking through (if they ever do so).

# Sequential Testing by Betting: A Martingale Approach

We are interested in testing which of two hypoteses are true:

1. The null hypothesis $H_0: P \in \mathcal{P}_{\text{null}}$.
2. The alternative $H_1: P \in \mathcal{P}_{\text{alt}}$.

We have observations of some procces denoted by $Z_1, Z_2, \ldots$ lying in some space $\mathcal{Z}$, and drawn i.i.d. according to $P$.

Testing by betting reframes hypothesis testing into a gambilng problem, instead of creating a function to produce test statistics. Totest the null $H_0$, a bettor may place repeated bets on the outcomes $\{Z_t : t \geq 1\}$ starting with an initial wealth $K_0 = 1$. 

A single round of betting (say at time $t$) involves the following two steps:

Firstly, the bettor selects a payoff function $S_t: \mathcal{Z} \to [0, \infty)$. The payoff has to satisfy $\mathbb{E}_P[S_t(Z_t)|\mathcal{F}_{t-1}] = 1$. I.e. the bet is fair when the null is true.

Then, the outcome $Z_t$ is revealed, and the bettor's wealth grows (or possibly shrinks) by a factor of $S_t(Z_t)$. Thus, the bettor's wealth after $t$ rounds of betting is $K_t = K_0 \prod_{i=1}^t S_i(Z_i)$.

The restriction on the conditional expectation of the payoff functions implies that under the null, $\{K_t : t \geq 0\}$ is a test martingale, which is a nonnegative martingale with an initial value 1. Due to this fact, $K_t$ is unlikely to take large values for any $t \geq 1$.

On the other hand, when $H_1$ is true, the bettor's choice of payoff functions, $\{S_t : t \geq 1\}$ should ensure that the wealth process grows exponentially. Such a wealth process naturally leads to the following sequential test: reject the null if $K_t \geq 1/\alpha$, where $\alpha \in (0, 1)$ is the desired confidence level. Ville's maximal inequality[^1] ensures that this test controls the type-I error at level $\alpha$.

When testing simple hypotheses ($H_0: Z_t \sim P$ and $H_1: Z_t \sim Q$ with $P$ and $Q$ known), the payoff fucntion $S_t$ is just the likelihood ratio $dQ/dP$. With this choice of payoff functions, we have $\mathbb{E}_P\left[S_t|\mathcal{F}_{t-1}\right] = 1$, meaning it is a fair bet under the null.

Under $H_1$, the wealth process with this payoff grows exponentially, with an optimal (expected) growth rate of $\text{KL}(Q, P)$: the KL-divergence between $Q$ and $P$.

When dealing with cases where either one or both of $H_0$ and $H_1$ are composite and nonparametric there is no obvious choice for the payoff functions. So there are a couple of design choices we need to make before we can use the testing by betting framework:

1. In which function class should $S_t$ lie?
2. How to ensure $\mathbb{E}\left[S_t|\mathcal{F}_{t-1}\right] = 1$ uniformly over $\mathcal{P}_{\text{null}}$?
3. How to ensure fast growth of $K_t$ under the alternative?

Before we continue with the exposition, we first define a sequential test:

> **Definition 1** (sequential-test). A level-$\alpha$ sequential test can be represented by a random stopping time $\tau$ taking values in ${1, 2, \ldots} \cup {\infty}$, and satisfying the condition $\mathbb{P}(\tau < \infty) \leq \alpha$, under the null $H_0$. Thus, $\tau$ denotes the random time at which the null hypothesis is rejected.