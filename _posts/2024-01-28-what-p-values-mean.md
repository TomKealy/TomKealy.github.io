---
layout: post
title: “What p-values really mean“
categories: statistics, hypothesis testing
---

{% newthought 'tl;dr Most people get _p_-values wrong.' %} This is how to understand and apply them correctly.<!--more--> 

## Introduction

In the branch of statistics known as hypothesis testing,  a statistical hypothesis is a claim about the value of a parameter in a statistical model: for example  the mean difference in revenue from two random groups of customers visiting a website is exactly zero. In statistics we call the hypothesis you want to reject the null hypothesis. A p-value is a summary of your data; it tells you how likely you are to see data at least as extreme as the data you have collected when your null hypothesis is correct.

A statistical hypothesis test is an algorithm used to calculate a test statistic and a p-value. A large p-value just means that the data was likely to occur under your hypothesis. Whereas a small value means the data was unlikely to have occurred under your hypothesis. This means that p-values are better at rejecting bad hypotheses than confirming good ones. 

The above discussion is mostly concerned with how well the data fits the model, and not whether the hypotheses in question are true. Statistics alone cannot answer such questions. Because the data you collect can be consistent with many similar hypotheses, statistics can only tell you which hypothesis could best have generated the data. Not whether the hypotheses were true. Data analysis alone cannot tell you whether the government program was a success, whether the drug should be prescribed to patients, or whether the website really is better with this new design. Statistics can tell you how many people were saved from poverty, how effective the drug is, or whether more people buy your products after a website redesign.

This might be why p-values are the most misunderstood, misinterpreted, and occasionally miscalculated of statistical quantities. No matter how hard statisticians try, what the p-value really means hasn’t broken through to the wider scientific consciousness. In fact, p-values are so difficult to interpret that when the Journal of the American Medical Association surveyed its members in 2007 about how to interpret p-values, none of the available options were correct {% sidenote 1 'Windish DM, Huot SJ, Green ML: Medicine residents’ understanding of the biostatistics and results in the medical literature. JAMA 298:1010- 1022, 2007'%}.

Applied researchers are certainly not to blame for misinterpreting p-values: p-values are, after all,  the problem child of statistics—a quantity straightforward to calculate but philosophically and practically difficult to interpret. That being said, we still conduct experiments so we still have to interpret their outcomes—that means we have to interpret p-values.

## How to Interpret Experiment Results {% sidenote 2 'This example was supplied by: https://www.scribbr.com/statistics/p-value/'%}

Let’s say you want to know whether there’s a difference in longevity between two groups of mice fed on different diets, diet A and diet B. You can statistically test the difference between these two diets using a two-tailed t-test. We consider the following hypotheses:

* __Null hypothesis (H0)__: there is no difference in longevity between the two groups.
* __Alternative hypothesis (H1)__: there is a difference in longevity between the two groups.

If the mice live equally long on either diet, then your test statistic will closely match the test statistic from the null hypothesis (that there is no difference between groups). The resulting p-value could be anything between 0 and 1. However, if there is an average difference in longevity between the two groups, then your test statistic will move further away from the values predicted by the null hypothesis, and the p-value will get smaller. The p-value will never reach zero, because there’s always a possibility that the null hypothesis could generate the data you have seen.

You run the experiment: you randomise the mice into different groups and they receive either diet A or diet B. You find that the lifespan on diet A (M = 2.1 years; SD = 0.12) was shorter than the lifespan on diet B (M = 2.6 years; SD = 0.1), with an average difference of 6 months (t(80) = -12.75; p < 0.01). Your comparison of the two diets results in a p-value of less than 0.01 below your alpha value of 0.05; therefore, you determine that there’s a statistically significant difference between the two diets.

But we should be cautious here. The reported p-value means that there is a 1% chance that these data would occur under the null hypothesis. This is not the same thing as saying that the null hypothesis is not true. It just means that you can reject the hypothesis that both diets produce similar outcomes. Building up a fuller picture would require knowledge about the specific content of the diets. So we can’t rely on the p-value alone to tell us much about how diet affects lifespan. 

One of the biggest mistakes in statistics is to over-interpret your p-values. Deciding a policy based on the criteria that p < 0.05, thereby rejecting the null hypothesis , is tantamount to saying that the effect is irrelevant. It puts the cart before the horse. Remember you did the experiment in the first place to find out something else: to measure the magnitude and direction of the effect size. 

As an expert you will need to combine the formal results of the experiment with your domain knowledge and come to a conclusion. Sometimes the conclusion might be that the null hypothesis holds true, despite a significant result; and sometimes the treatment works despite the test statistic disagreeing. Beliefs and actions cannot come directly from statistical results, especially results from a single experiment. 

Hypothesis tests aren’t the only form of evidence out there, and you will have to gather many different strands to make a decision. Deciding upfront that if your p-value is  < 0.05 then your research program is a success, and a failure otherwise, is unhealthy. It leads researchers to focus their actions on the outcomes at the expense of the process. Only healthy processes will create good results.
