---
layout: default
title:  "Local Linear Trend models with seasonality"
date:   2021-03-10
categories: time-series
---

# Introduction

* Ads served to people via Google/Meta
* Companies need to bid for their ads to be served.
* Several ways of doing this: tROAS vs tCPA
* We are interested in optimising the ad spend, can we do better than 'set and forget' with a fixed tROAS/tCPA?

I have an application in mind. 

I'm conducting an A/B test comparing two bidding strategies for online auctions. One strategy (the fixed strategy) simply bids a fixed ammount of money (on average) per auction. The other strategy adapts the ammount bid based on performance over the last few auctions.

The challenge is that I cannot directly control the amount of money each strategy spends. The target cost per activation (CPA) determines the total spend, which in turn affects the number of conversions and ultimately, the return on investment (ROI). The only aspect I can control is the duration of the experiment.

My goal is to determine whether the adaptive strategy can outperform the fixed strategy by at least 10% in terms of ROI.
Here are a few key questions I have:

1. What is the appropriate randomisation unit in this case? We can't be certain that the dollar ammounts or the number of conversions are at all comparible between experiment variations.
2. Given the uncertainty over the randomisation unit, is the simple total profit (revenue - spend) an appropriate metric to determine the outcome of the experiment? What is an appropriate method for statistical inference in this case (I have read this post as a starter).
3. Sample size & duration: How do I determine the appropriate sample size or experiment duration to detect a 10% uplift in ROI with statistical significance? Is it even correct to think of randomisation units being days?
4. Variance estimation: Given that spend is not directly controlled but influenced by target CPA, what metrics or techniques should I use to accurately estimate the variance in ROI across both groups?
5. Bias mitigation: How can I account for differences in spend or conversion patterns that might create bias, especially if one strategy adapts faster than the other?
6. Stopping rules: Since I can only control how long the experiment runs, what would be appropriate stopping criteria or statistical tests to apply during and after the experiment to ensure the results are reliable?

# Sequential Testing by Betting: A Martingale Approach

consider a hypothesis testing problem with a null hypothesis H0 : P ∈ Pnull and alternative H1 : P ∈ Palt, and observations denoted by Z1,Z2,... lying in some space Z, and drawn i.i.d. according to P. To test the null H0, a bettor may place repeated bets on the outcomes {Zt : t ≥ 1} starting with an initial wealth K0 = 1. A single round of betting (say at time t) involves the following two steps. (i) First, the bettor selects a payoff function St : Z → [0, ∞), under the restriction that it ensures a fair bet if the null is true. Formally, this is imposed by requiring St to satisfy EP [St(Zt)|Ft−1] = 1 (or more generally, EP [St(Zt)|Ft−1] ≤ 1) for all P ∈ Pnull, where Ft−1 = σ(Z1,...,Zt−1). (ii) Then, the outcome Zt is revealed, and the bettor’s wealth grows (or posQsibly shrinks) by a factor of St(Zt). Thus, the bettor’s wealth after t rounds of betting is Kt = K0 ti=1 Si(Zi).
The two key technical pieces that underpin the framework are test martingales [Shafer et al., 2011] and Ville’s inequality [Ville, 1939]. To elaborate, the restriction on the conditional expectation of the payoff functions implies that under the null, {Kt : t ≥ 0} is a test martingale, which is a nonnegative martingale with an initial value 1. Due to this fact, Kt is unlikely to take large values for any t ≥ 1. On the other hand, when H1 is true, the bettor’s choice of payoff functions, {St : t ≥ 1} should ensure that the wealth process (or equivalently, the amount of evidence against the null)
4
grows at a fast rate, ideally exponentially. Such a wealth process naturally leads to the following sequential test: reject the null if Kt ≥ 1/α, where α ∈ (0, 1) is the desired confidence level. Ville’s maximal inequality (recalled in Fact 1 in Appendix A) ensures that this test controls the type-I error at level α.
The discussion in the previous paragraph highlights some key design choices that must be made to use these ideas for two-sample testing: in which function class should St lie; how to ensure E[St|Ft−1] = 1 uniformly over Pnull; and how to ensure fast growth of Kt under the alternative? When testing simple hypotheses H0 : Zt ∼ P and H1 : Zt ∼ Q with P and Q known, an obvious choice of St is the likelihood ratio dQ/dP. Indeed, with this choice of payoff functions, we have EP [St|Ft−1] = 1, meaning it is a fair bet under the null. Furthermore, it is easy to check that under H1, the wealth process with this payoff grows exponentially, with an optimal (expected) growth rate of dKL(Q, P ): the KL-divergence between Q and P . However, when dealing with cases where either one or both of H0 and H1 are composite and nonparametric (as is the case with the two-sample testing problem considered in this paper), there is no obvious choice for the payoff functions.

## The Core Insight

The key insight from the paper is that hypothesis testing can be reframed as a betting game: if you can consistently make money betting against the null hypothesis, you have evidence against it. This idea gives us a mathematically rigorous but intuitively clear way to do sequential testing.
## How It Works
### The Basic Setup

Start with $1 of betting capital
For each new observation, you place a bet
If your betting capital reaches $1/α (e.g., $20 for α=0.05), you reject the null hypothesis

This setup satisfies a crucial property: under the null hypothesis, your betting capital is a martingale - meaning your expected future wealth equals your current wealth. This property ensures proper Type I error control.

### The Betting Process
For each observation:

Choose a betting amount (constrained to ensure you can't lose all your money)
Observe the outcome
Update your wealth based on the bet's outcome

Mathematically, at time t:


The martingale approach handles this in the context of your bidding strategies.
In the paper, Section 2.1 covers this through the filtration {Ft-1} concept. Here's what's happening:

At each time t, the betting decision (λt) and prediction strategy (gt) must be Ft-1-measurable, meaning they can only use information available up to time t-1. This accounts for time dependencies because:

$$
E[gt(Xt) - gt(Yt)|Ft-1] = 0
$$

For your bidding scenario specifically:

Each day's bids and outcomes aren't independent (today's bidding may be influenced by yesterday's performance)
But the relative performance difference between strategies each day can still be used for betting

The key adaptation for your case would be:

```python
def calculate_bet(day_data, history):
    # Use history to normalize daily performance
    normalized_roi_diff = (
        (day_data['adaptive_roi'] - day_data['fixed_roi']) /
        get_historical_volatility(history)
    )
    return max(-0.5, min(0.5, normalized_roi_diff))
```

Crucially, in the paper's Remark 7 (page 13):
"The results stated above in Theorem 1 are valid under a much weaker assumption that the stream {(Xt,Yt) : t ≥ 1} consists of independent pairs of observations satisfying (Xt,Yt) d= (Yt,Xt) under the null."
This means we only need:

Each day's comparison to be a fair test under the null
The betting strategy to only use past information

We don't need full independence between days, which makes this approach particularly suitable for your adaptive bidding scenario.