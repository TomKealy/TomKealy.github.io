---
layout: default
title: Is this Simpson's Paradox?
subtitle: A common misdiagnosis in A/B test results, and how to tell the difference.
categories: statistics, experimentation
bibliography: references.bib
toc: true
---

# Introduction

You're running an A/B test. The overall results look good — the test group is outperforming control. Then you break the results down by device type and something strange appears: Desktop users are going in the *opposite* direction.

Your first instinct is Simpson's Paradox. But is it?

# The scenario

Imagine you're analysing an A/B test with 300,000 users. Split by device, the data looks like this:

| Segment | Group   | Allocated | Converted | Rate  |
|---------|---------|-----------|-----------|-------|
| Mobile  | Control | 200,000   | 10,000    | 5.00% |
| Mobile  | Test    | 200,000   | 10,775    | 5.39% |
| Desktop | Control | 50,000    | 2,500     | 5.00% |
| Desktop | Test    | 50,000    | 2,350     | 4.70% |

The Mobile segment shows a **+7.24%** improvement. The Desktop segment shows a **-5.04%** decline. And overall?

| Group   | Allocated | Converted | Rate  | Change  |
|---------|-----------|-----------|-------|---------|
| Control | 250,000   | 12,500    | 5.00% |         |
| Test    | 250,000   | 13,125    | 5.25% | +4.76%  |

The overall result follows Mobile — the larger segment — while Desktop quietly goes the other way. Paradox, right?

Not quite.

# What Simpson's Paradox actually is

Simpson's Paradox occurs when a trend present in *every* subgroup *reverses* when the groups are combined. The canonical example comes from a 1986 medical study comparing two treatments for kidney stones[^1]:

| Segment      | Treatment A       | Treatment B        |
|--------------|-------------------|--------------------|
| Small stones | **93%** (81/87)   | 87% (234/270)      |
| Large stones | **73%** (192/263) | 69% (55/80)        |
| Overall      | 78% (273/350)     | **83%** (289/350)  |

Treatment A is better for small stones. Treatment A is better for large stones. And yet Treatment B appears better overall.

This is Simpson's Paradox. The reversal happens because Treatment B was applied disproportionately to small stones — the easier cases — which inflated its overall success rate. Stone size is a confounding variable that was invisible until you looked for it.

# What's actually happening in our A/B test

In the kidney stones example, *both* subgroups agree: Treatment A wins in small stones, Treatment A wins in large stones. The paradox is that the overall result contradicts both subgroups.

In our A/B test, the subgroups *disagree with each other*: Mobile improves, Desktop declines. The overall result doesn't contradict either subgroup — it simply reflects the majority. With 80% of users on Mobile, the overall conversion rate is dominated by Mobile's performance. This is intuitive, not paradoxical.

For it to be Simpson's Paradox, you'd need to see Mobile *and* Desktop both decline individually, while the overall result showed an improvement — or vice versa. That's not what we have here.

# Why it still matters

Misdiagnosing this as Simpson's Paradox is easy, but the correct diagnosis doesn't make the Desktop result any less real. A chi-squared test on the Desktop segment gives p = 0.03 — the underperformance there is statistically significant and unlikely to be noise.

The overall positive result is genuine. So is the Desktop decline. Both can be true simultaneously, and both need to inform the decision about whether to ship the change. An improvement that helps 80% of users while harming the other 20% 
is a very different outcome from a uniform improvement. Treating the overall number as the whole story would miss that.

Simpson's Paradox, when it genuinely occurs,signals a confounding variable that needs to be brought into the analysis. What we have here is something simpler but equally important: a treatment that works differently for different users. 

Stratifying your results isn't just a way to catch paradoxes — it's how you find out *who* your experiment is actually helping.

# References
{% bibliography --cited %}

[^1]: Charig CR, Webb DR, Payne SR, Wickham JE. Comparison of treatment of renal 
calculi by open surgery, percutaneous nephrostomy, and extracorporeal shockwave 
lithotripsy. *BMJ*. 1986;292(6524):879–882.