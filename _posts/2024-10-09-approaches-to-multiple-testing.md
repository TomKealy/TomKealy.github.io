---
layout: default
title:  "A Grab Bag of Approaches to Frequentist Multiple Testing."
date:   2024-10-09
categories: hypothesis-testing, multiple-testing
---

We show how to handle more than one hypothesis at a time. 

## Introduction

This note is to explain the statistical background behind multiple testing, which experimenters (be they researchers in academia, NGOs, or tech)  need to  consider when implementing their experiments. Understanding multiple testing is crucial because it determines the reliability of your results and influences how much your end users can trust your papers or products.

The presentation will be in terms of p-values as the algorithms for multiple metric adjustments can be done by hand. Their purpose is to develop intuition for the decision theoretic issues underlying the algorithms. Here we are focussing on the why, rather than on any specific how.

## The Challenge of Multiple Testing

When a researcher runs an experiment, they will typically choose a single metric to decide whether the intervention has been successful or not. This is called the primary decision metric or endpoint in the experimentation literature. Some examples include: the ratio of the total number of conversions to unique visitors on a website (the conversion rate); or the test score of pupils in a classroom (as in the STAR study).

If you run A/B tests on your website and regularly check ongoing experiments for significant results, you might be falling prey to what statisticians call repeated significance testing errors. As a result, even though your dashboard says a result is statistically significant, there’s a good chance that it’s actually insignificant.
 
To determine the outcome of the experiment, the researcher specifies a binary decision rule tied to their primary metric. For example the conversion rate of the test variation increases by at least 5% (a typical hypothesis in tech). Or that pupils in smaller classes have 20% higher scores on a standardised test at the end of the school year (STAR study).

When an A/B testing dashboard says there is a “95% chance of beating original” or “90% probability of statistical significance,” it’s asking the following question: Assuming there is no underlying difference between A and B, how often will we see a difference like we do in the data just by chance? The answer to that question is called the significance level, and “statistically significant results” mean that the significance level is low, e.g. 5% or 1%. Dashboards usually take the complement of this (e.g. 95% or 99%) and report it as a “chance of beating the original” or something like that.

However, the significance calculation makes a critical assumption that you have probably violated without even realizing it: that the sample size was fixed in advance. If instead of deciding ahead of time, “this experiment will collect exactly 1,000 observations,” you say, “we’ll run it until we see a significant difference,” all the reported significance levels become meaningless. This result is completely counterintuitive and all the A/B testing packages out there ignore it, but I’ll try to explain the source of the problem with a simple example.

There is more than one thing that researchers can measure, and alongside the primary metric many other things are measured about the experimentation subjects. This includes financial metrics in our conversion rate example, or the rates at which pupils dropped out of education altogether in the STAR study.

It’s tempting, then, to have multiple decision metrics to decide if an intervention has been successful. For example we might decide that a change in web page design is an improvement if it increases the conversion rate by at least 5%, and also increases the average revenue per user (ARPU) by 3%. This is an absolutely reasonable thing to do. When we add more metrics with statistical information to an experiment we think that we are increasing the number of endpoints by 1; however this is incorrect, we are increasing the number of endpoints for our experiment by an order of magnitude. Consider going from 1 to 2 metrics in an experiment.  Think of it like this: with one metric, under any binary decision rule you have two endpoints for the experiment. Yes, or no. 0 or 1. When you add a second metric, you now have four endpoints which the experiment can take: 00, 01, 10, 11. With three metrics you now have eight endpoints: 000, 001, 010, 100, 011, 101, 110, 111. Since statistical tests are probabilistic, with each independent test you add you increase the probability that one of them will say yes, even if there is no effect present in the data.  This is the challenge of multiple testing: how can you add more metrics whilst controlling the overall experiment-wise false probability rate?

There are a number of statistical approaches to handle this inflation of error rates, so that the conclusions from any statistical analysis are sound.

### What is a false positive and why is it important?

In hypothesis testing, a false positive occurs when a statistical result returns a ‘yes’ decision even when this is not warranted by the data. In other words, we conclude an effect is present in the data when it actually isn’t. It’s important to remember that statistical tests are not infallible, they are instead probabilistic and so, no matter how stringent you make the requirements of the analysis, they can always go awry. 
This type of error is crucial because it can lead to unnecessary actions or interventions based on a perceived effect that isn't real. There is a feeling that false positives are costless, and that increasing the number of false positives is no big matter. However, consider that most statistical tests are two sided and that they do not distinguish between the upper and lower tails of the distribution. In this situation a false positive could be in the negative direction, and in fact lose the company money.

#### What is the Family-wise error rate?

The false positive rate (FPR) of a single metrics is the proportion of all negatives that still yield positive test outcomes. It's the probability that a null hypothesis is incorrectly rejected given that it is true. The FPR is commonly set at 5% (0.05), meaning we are willing to accept a 5% chance of falsely claiming a significant effect or association.

When we start adding more metics to our statistical analyses the probability that any single negative yields a positive outcome is called the family-wise error rate (FWER). The FWER is the probability of making at least one Type I error (i.e., falsely rejecting a true null hypothesis) across a family of tests. In other words, it controls the total rate of false positive results to keep it below a predefined threshold. Many primary sources use FWER and the FPR for multiple tests interchangeably. Keeping the FWER at an acceptable level is challenging, as it is a very stringent requirement.

### Family Wise Error Rate (FWER)
One of the best-known methods to control the FWER is the Bonferroni correction, which involves dividing the desired significance level by the number of comparisons. For example: if you set an overall alpha of 0.1 and you wanted statistical information on 10 metrics in your experiment, then you would then use an alpha of 0.01 (i.e. divide by 10) for all of the 10 independent tests. This way the overall FWER is constrained to 0.1.  This method is quite stringent (but by far the safest) as it dramatically reduces the window that any single test is a false positive. Though can reduce the risk of any false positives at the cost of potentially missing true positive results (increasing Type II errors). 
For a business AB testing platform, this means that we could reduce the number of rollouts by a factor proportional to the number of metrics we are looking at. In medicine this may be a good trade-off to make.

#### What is the false discovery rate?

In maths one useful move that's always available to you is to define yourself out of a problem. The false discovery rate is an example of one such move. The false discovery rate (FDR) is the expected proportion of false positives among all declared positives. Unlike the FPR, the FDR controls the expected proportion of false discoveries, rather than the chance of any false discoveries.

Colloquially, what this means is that you tolerate a cost of slightly more false positives overall, with the benefit that any adjustment scheme for the FDR is more forgiving.

### False Discovery Rate (FDR)
The FDR, on the other hand, is the expected proportion of Type I errors among all rejected hypotheses. Instead of trying to avoid any single false positive like FWER, the FDR control procedure tries to limit the proportion of false positives among all discoveries.

The Benjamini-Hochberg procedure is a common method used to control the FDR. It is generally less conservative than methods that control the FWER, allowing for more false positives but increasing the power to detect true positives. It works by calculating a sequence of increasing adjusted p-values for each different metric.

In essence, the key difference lies in what each rate seeks to control: FWER controls the probability of at least one false positive, while FDR controls the expected proportion of false positives. Your choice between the two would depend on the balance you wish to strike between avoiding false positives and not missing true positives.

### What is a false negative and why is it important?

A false negative, on the other hand, occurs when we fail to reject a false null hypothesis. In simpler terms, we conclude that there is no effect or association when there actually is. False negatives are important because they can prevent us from taking necessary actions or recognizing significant associations or effects.
This error is when we leave money on the table. Statistical tests are designed to give us the most statistical power (to minimise the number of false negatives), subject to constraints on the chance of detecting a false positive. Again, remember that no matter the parameters of your system (for example, you might increase the sample size dramatically to reduce the probabilities above) you will always be faced with some inherent uncertainty with a statistical test.

### How do they vary by the number of comparisons?

As we perform more and more independent statistical comparisons, we increase the likelihood that our results are false positives (and also increase the number of false negatives we leave by the wayside). This is referred to as the "multiple testing problem." For each individual test, the probability of a false positive or false negative may be small. However, when we perform multiple tests, these probabilities multiply, exponentially  increasing the chance of one or more false results. For example, if we have 10 independent tests, the probability that a single one is positive purely by change is 1 - (1-0.05)^10 ~ 40%.

There are a few ways to avoid this, but first we need to define a few terms:

### How can we prevent this from happening?

We can mitigate the issue of multiple testing by applying various correction methods. Some popular ones include the Bonferroni Correction for the FWER and the Benjamini-Hochberg procedure for the FDR. These techniques adjust our threshold for significance (e.g., lowering the p-value) based on the number of comparisons being made to maintain an appropriate error rate.

#### Methods to control the Family Wise Error Rate

#### Methods to control the False Discovery Rate

### Must we apply the adjustments equally.

No. It’s not required that we apply adjustments equally. In fact the Bejamini-Hochberg algorithm explicitly does not apply equally to all metrics.

A good example is to consider the following situation. You have 11 metrics you would like to compute p-values for in an experiment, with an overall alpha of 0.1. However, for one of them you wish to allocate an alpha of 0.5, and for the rest of them you want to allocate a p-value of 0.005 (0.05/10). This is a perfectly reasonable situation, as it keeps the overall alpha to 0.1 (the sum of all the individual alpha values). You could also extend this: your primary metric gets an alpha of 0.5, some set of secondary metrics gets an alpha of 0.3 (and the p-value thresholds are calculated via the Benjamini-Hochberg procedure) and even more tertiary metrics have an alpha of 0.2 (and the p-value thresholds are calculated via the Benjamini-Hochberg procedure). This tiering strategy is also appropriate. What wouldn’t be appropriate would be leaving the primary metric unadjusted, and still claiming the overall alpha is 0.1. In that case, the overall alpha would be inflated (possibly as high as 0.2).
