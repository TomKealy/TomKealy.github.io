---
layout: default
title:  How wide should your confidence interval be?
subtitle: A short, hands-on, guide.
date: 2024-11-19
categories: causal-inference, experimentation
toc: true
---

# Introduction
Every experiment we run is asking a question: did this change actually make a difference, or did we just get lucky?

To answer that question rigorously, we use statistical tests. These tests help us understand whether the patterns we observe in our data occur by chance or if they represent real insights. There is nothing magical about a test, they are just measuring tools. Like any measuring instrument, those tests need to be calibrated. We have to decide upfront how much uncertainty we're willing to tolerate. That calibration is what a confidence interval is about.

# Calibration of Statistical Tests
Calibrating statistical tests involves setting up two key probabilities: alpha and beta.

## Two ways to be wrong
A statistical test can be wrong in two ways: 

__False positive:__ You conclude the change worked, but it didn't. You got lucky with the numbers.

__False negative:__ You conclude the change didn't work, but it actually did. You missed a real effect.

You can't eliminate both risks at once. Making it harder to get a false positive (demanding stronger evidence) also makes it easier to miss real effects (unless you run the experiment on more people).

When you run an experiment you have to set these two probabilities and trade off the false positive rate against the false negative rate. All experiments balance the two types of errors. If the trade off is unacceptable, then you will have to use a bigger sample.

## What a confidence interval tells you
Confidence intervals (CIs) describe the range within which we expect the true value of our measurement to fall. Instead of saying "the change increased revenue by €1 per customer," it says "the true effect is most likely somewhere between €X and €Y."

The confidence level determines how wide that range is. Here's a concrete example: imagine you ran an experiment on 10,000 customers per variant and observed a mean revenue increase of €1, with typical variability in the data.

* A 90% confidence interval would give you a range of [€0.40, €1.60]
* A 95% confidence interval would give you a range of [€0.27, €1.72]

The 95% interval is wider because it's making a stronger guarantee — it's saying the true value falls within this range in 95 out of 100 repetitions of the experiment, rather than 90. More certainty requires more room.

## The cost of more certainty
Wider intervals aren't free. If you want the same precision at a higher confidence level, you need a bigger sample.

To get a 95% CI as narrow as the 90% CI in the example above — that is, to be more certain while keeping the range equally tight — you'd need 15,000 customers per variant instead of 10,000. That's a 50% increase in sample size for roughly €0.12 of extra precision on each side of the estimate.

Whether that tradeoff is worth it depends on the stakes.

# What other industries do

There's no universal right answer here. Different fields have different tolerances for error:

* Pharmaceuticals typically use a 99% CI or stricter — the consequences of a false positive can be severe.
* Tech companies commonly use 90–95% CIs. Ronny Kohavi, who literally wrote the book on online experimentation (*Trustworthy Online Controlled Experiments*), recommends a 95% CI for most decisions.

The right level for any organisation depends on how reversible its decisions are, how many experiments it runs, and how costly each type of error actually is.


