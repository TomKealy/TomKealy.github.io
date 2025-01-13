---
layout: default
title:  Hypothesis Testing by Betting
subtitle: A simpler alternative to $p$-values.
date:   2025-01-13
categories: time-series
---

# Introduction

The $p$-value is the most misunderstood, misused, and maligned scientific quantity. The $p$-value is merely a summary of your data; it tells you how likely you are to see data at least as extreme as the data you have collected when your null hypothesis is correct. However it often does double duty in statistical analyses: it can tell us whether or not to reject the null hypothesis (e.g. if $ p \leq \alpha$) as well as how confident we should be in the alternative hypothesis. Altogether, this makes the $p$-value a poor tool for scientific communication.

A simpler alternative to $p$-values is to report the result of a bet against the null hypothesis. This is **testing by betting.** This method is closely related to the likelihood method, and leads to alternatives for statistical confidence and power.

Testing by betting is straightforward. First we select a payoff (which can be either be zero or a positive number). Then we buy this payoff for its (hypothesized) expected value. If this bet multiplies the money it risks, we have evidence against the hypothesis. The factor--called the **betting score**--measures the strength of this evidence. Multiplying our money by 5 merits attention; multiplying it by 100 or by 1000 might be considered conclusive.

 Testing by betting is simpler than reporting a $p$-value, because a $p$-value represents the probability of obtaining test results at least as extreme as those observed, assuming the null hypothesis is true. The key phrase here is "at least as extreme"--you're not just looking at the probability of getting exactly your observed result, but rather the probability of getting your result or any more extreme result. A $p$-value is thus the result of a _family_ of tests:

* For any observed test statistic, you need to consider all the possible outcomes that would be "more extreme"
* You have to decide what counts as "more extreme" (one-tailed vs two-tailed tests)
* You're essentially conducting multiple implicit hypothesis tests - one for your actual result and one for each more extreme possible outcome.

The betting score  is simpler because it just looks at one specific bet and its outcome--did you win or lose, and by how much? There's no need to consider a family of hypothetical outcomes or define what counts as "more extreme." To take an example:

1. With a $p$-value: If you observe a z-score of 2.5, you need to calculate the probability of observing a z-score ≥ 2.5 (or ≤ -2.5 for a two-tailed test)
2. With a betting score: You just calculate how much money your specific bet made or lost (e.g. 10x the initial bet).

Betting scores also have a number of other advantages:

1. There is less uncertainty when you get a large betting score, compared with a small $p$-value. You will not forget that a long shot can succeed by sheer luck.
2. When we make a bet, we create an implied alternative hypothesis. The betting score is the likelihood ratio with respect to this alternative. What the betting score means is aligned with what a likelihood ratio means.
3. As well as implying an alternative hypothsis, the bet also implies a **target.** This is the value we desire to make the bet "worthwhile." Implied targets are useful than power calculations, because an implied target along with an actual betting score tells a coherent story. To interpret a statistical test, you need to know the test's actual statistical power. This, in turn, requires a fixed significance level. In traditional statistical inference, there's a  awkward relationship between power calculations and $p$-values: Power calculations are done before the study; but after running the study, you report a $p$-value, which might be any value between 0 and 1. Betting scores are superior because they have a more coherent narrative -both your planning and your results are in the same units (betting scores for your money).
4. Testing by betting is a more agile approach to science, because you don't have to plan your entire analysis in advance.