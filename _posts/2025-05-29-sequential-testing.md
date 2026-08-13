---
layout: default
title: Sequential testing — making decisions before your experiment ends
subtitle: Why peeking at results inflates your false positive rate, and how sequential testing fixes it.
categories: statistics, experimentation
bibliography: references.bib
toc: true
---

# Introduction

The textbook version of an A/B test is clean: you decide on a sample size
upfront, run the experiment until you have enough data, then analyse the results
once. In practice, almost nobody does this. Experiments take weeks. Metrics move
in unexpected directions. There is pressure to ship. So people check.

Checking your results before the experiment ends — peeking — is intuitive and
feels harmless. If the effect is already large and significant, why wait? The
problem is that peeking is not harmless. Done naively, it silently inflates your
false positive rate in ways that are difficult to detect and easy to forget about.

Sequential testing is the statistical framework that makes interim monitoring
safe. It lets you look at your data as it accumulates and make early decisions
when the evidence is strong enough — without compromising the reliability of
your results.

# The peeking problem

To understand why peeking is dangerous, consider what a *p*-value looks like
over time when there is no real effect.

Suppose you are running an A/A test — two identical groups, no treatment
difference. By construction, the null hypothesis is true: there is no effect.
If you analyse the data once at the end, you will observe a false positive
roughly 5% of the time (at α = 0.05). That is exactly what the test is
calibrated to do.

Now suppose instead you check the *p*-value every day for 20 days, and stop the
moment it drops below 0.05. *p*-values fluctuate — they are random variables
that wander up and down as data accumulates. Even when the null hypothesis is
true, the *p*-value will dip below any fixed threshold with increasing
probability the more times you look. In an A/A test monitored daily for 20
days, the probability of observing at least one false positive result exceeds
25%. Check for long enough, and a random walk will eventually cross any fixed
boundary.

This matters because the date you choose to end the experiment is no longer
arbitrary — it is selected based on the results. You are, in effect, cherry-
picking the readout from a sequence of tests and claiming the significance of
a single one. The procedure is no longer calibrated to α = 0.05; it just looks
like it is.

## Why it's worse than it seems

The false positive inflation from peeking compounds in ways that are not
obvious. Consider a team that monitors ten metrics daily across a two-week
experiment. Even if no individual metric is being explicitly tested at interim
points, the act of scanning results and flagging "interesting" movements
introduces exactly this problem. Selective attention to significant results is
equivalent to testing for them.

The other risk is the opposite error: stopping early because a metric looks
good, before there has been enough time to observe regressions in secondary
metrics. An experiment that is significant for conversion rate after one week
may still be underpowered to detect a negative effect on retention two weeks
later.

# How sequential testing works

Sequential testing solves the peeking problem by adjusting the significance
threshold at each look to account for the fact that multiple looks are being
taken. The total false positive rate is controlled across all interim analyses,
not just the final one.

## Alpha spending

The key concept is the alpha spending function. You start with a budget of
false positive probability — typically α = 0.05 or α = 0.10. Each time you
look at the data, you spend some of that budget. The spending function
determines how much you spend at each look.

Two classical approaches illustrate the tradeoff:

**O'Brien-Fleming** spends very little alpha early and more later. The
significance threshold is stringent for early looks — a *p*-value of 0.001
or lower is required to stop at the first interim — and relaxes toward the
planned final alpha as the experiment matures. This approach closely preserves
the power of the original fixed-horizon test, at the cost of making early
stopping difficult.

**Pocock** spends alpha equally across all looks, using the same adjusted
threshold at every interim analysis. Early stopping is more achievable, but
the final threshold is also more stringent than the original α, which reduces
power.

Both approaches require you to specify the number of looks in advance, which
limits flexibility.

## Always-valid inference

A more flexible approach, and one increasingly common in industry
experimentation platforms, is always-valid inference — the ability to monitor
results continuously without pre-specifying the number or timing of looks.

The theoretical basis is the mixture Sequential Probability Ratio Test
(mSPRT) {% cite johari2022always %}. Rather than spending a fixed alpha budget
across planned interim analyses, mSPRT constructs a test statistic that
remains valid at any stopping time. The guarantee is that the false positive
rate is controlled no matter when you look or how many times — you can check
daily, hourly, or continuously, and the stated α is correct.

The practical implication is that you can stop an experiment as soon as the
evidence is strong enough, or run it longer than planned if you need more
power, without compromising the integrity of the results.

# The cost of sequential testing

Sequential testing is not free. To control the false positive rate across
multiple looks, the significance threshold at any individual interim analysis
is more stringent than the threshold you would use in a fixed-horizon test.
This means it takes longer to accumulate enough evidence to stop early.

Put differently: sequential testing gives you flexibility in when you stop, but
it does not give you a shortcut to significance. If an effect is genuinely small,
you will still need to run the experiment close to its original planned duration.
Where sequential testing adds value is for large effects, where evidence
accumulates quickly, or for regressions, where you want to catch problems early
rather than waiting for a scheduled readout.

There is also a precision cost. Estimates from experiments stopped early tend
to be inflated — a phenomenon known as the winner's curse. When you stop
because a metric crossed a significance threshold, you are conditioning on
observing a large estimate, which is more likely when the true effect is near
the boundary of detectability. Sequential test statistics are designed to
control the false positive rate, but the point estimates at stopping time
should be treated with some caution.

# When to use sequential testing

**Catching regressions early.** If an experiment introduces a bug or has
unintended consequences on a key metric, sequential testing lets you identify
this in the first days rather than waiting for the full experiment duration.
This is one of the strongest arguments for continuous monitoring: the cost of
running a broken experiment for three weeks is much higher than the cost of
stopping early.

**High opportunity cost of waiting.** If there is a time-sensitive decision
— launching ahead of a major event, fixing an issue that is costing revenue —
and the evidence already appears strong, sequential testing provides a
principled basis for acting early. Use this carefully: early significance on
one metric does not guarantee adequate power across all metrics.

**Large expected effects.** If your prior is that the effect will be large
(for example, a major redesign rather than a button colour change), sequential
testing makes it practical to stop as soon as that large effect manifests.

# When not to use sequential testing

**When many metrics matter.** Early significance on your primary metric does
not mean secondary or guardrail metrics have reached sufficient power. Stopping
early on the basis of one metric can leave regressions in others undetected.
If the decision depends on a broad set of metrics, run the full duration.

**When the effect is expected to be small.** Sequential testing does not
accelerate evidence accumulation — it just lets you act on it sooner when it
is strong. For experiments designed to detect subtle effects near the minimum
detectable effect, you are unlikely to stop materially early.

**When the bake-in period has not elapsed.** Some metrics take time to
stabilise — a change to a recommendation algorithm may not show its full
effect on retention for several weeks. Sequential testing does not change
this. Stopping an experiment before the metric has had time to reflect the
true effect will produce misleading results regardless of the *p*-value.

# Summary

Peeking at experiment results without a principled framework inflates the
false positive rate in proportion to how many times you look and how selectively
you stop. The intuition that "if it's already significant, why wait?" is
seductive but statistically invalid under a fixed-horizon testing framework.

Sequential testing restores validity by adjusting the significance threshold
across interim looks, spending a controlled alpha budget so the overall error
rate remains calibrated. Always-valid approaches like mSPRT extend this to
continuous monitoring without requiring you to pre-specify when you will look.

The result is not faster experiments — sequential testing cannot manufacture
evidence that is not in the data. What it provides is the ability to act
quickly when large effects are present, and to catch regressions before they
run for their full planned duration.

# References
{% bibliography --cited %}