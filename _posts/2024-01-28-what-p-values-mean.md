
---
layout: post
title: “What _p_-values really mean“
categories: optimisation
---
## Introduction

P-values are the most misunderstood, misinterpreted, miscalculated, and misbegotten statistical quantities. The reason this is the case are a mixture of bad history, poor pedagogy, and lazy thinking. No matter how hard statisticians try, what the _p_-value really means hasn’t pierced the wider scientific and engineering consciousness. In fact, _p_-values are so difficult to interpret that when the Journal of the American Medical Association surveyed it’s members about how to interpret _p_-values, *none* of the options were correct.

This post outlines key misunderstandings of what *p*-values *don’t* mean, in the hope that once we understand what something isn’t then what it is becomes clearer. They are taken from my time as a statistician and data scientist in academia and industry. In this article I’ve chosen an $$ \alpha $$ threshold of 0.05, but it can be replaced with any value you desire.

## History lesson

Applied researchers are not at fault for misinterpreting the *p*-value. Its controversy goes back to founding of statistical inference. It’s okay, it’s not you. *P*-values are the ugly child of probability: a quantity that is straightforward to calculate, but which is philosophically and practically difficult to interpret. It doesn’t help that modern scientific practice overloads the meaning of significance testing.

### Fisher

R.A. Fisher was the geneticist and statistician who introduced the *p*-value. He was an unusual researcher: having theoretical knowledge of probability and extraordinary skills in formal calculation, as well as extensive practical field experience. He helped biology and genetics out of their infancy, but unfortunately for us, because of his background, he knew what he meant by a *p*-value, but he could never straightforwardly describe what it meant from an inferential point of view. Roughly, Fisher meant that a *p*-value was a numerical guide for how much the data you have collected are consistent with your null hypothesis. He never mentioned an error rate, or rejection of the null. All he meant by ‘significant’ was that something was worthy of notice. In his *Statistical Methods for Research Workers* he said:

> the writer prefers to set a low standard of significance at the 5 percent point…A scientific fact should be regarded experimentally established only if a properly designed experiment rarely fails to give this level of significance.

i.e. a *p*-value less than 0.05 just means that you should repeat your experiment, and if subsequent repetitions were also significant then you could consider it unlikely that your observed effects due to chance. 
### Neyman-Pearson

Most scientists, both in Fisher’s era and now, considered this a hugely unideal situation. Everybody knew when something was significant, they just couldn’t define it. So in 1928, Neyman and Pearson redefined the problem by formalising what they called a hypothesis test, and made the interpretation of *p*-values orders of magnitude more complicated. 

Their main contribution was to reframe the problem as one where the researcher chose between two hypotheses: the null and the alternative. They proposed degrees of freedom—called the false positive rate $$ \alpha $$ and the false negative rate $$ \beta $$—which should vary depending on the experimental situation with the consequences of error. These rates defined a critical region for a summary statistic, and if a result fell into the critical region then the null hypothesis could be definitively rejected. 

Notably, in this formulation, we do not care whether the null or the alternative hypotheses are true, or even useful. That is wholly up to the discretion of the researcher. Nor do we know where in the critical region the result fell, only that it did, or did not, fall into the critical region. There is no measure of evidence in the Neyman-Pearson land. 

> Without hoping to know whether each separate hypothesis is true or false, we may search for rules to govern our behaviour with regard to them, **in following which we insure that, in the long run of experience, we shall not often be wrong** [Emphasis mine]. Here, for example, would be such a "rule of behaviour"; to decide whether *H* of a given type be rejected or not, calculate a specified character, *x,* of the observed facts; if *x> Xo,* reject *H,* if *x* < *XQ,* accept *H.* Such a rule tells us nothing as to whether in a particular case *H* is true... But it may often be proved that if we behave according to such a rule, then in the long run we shall reject *H* when it is true not more, say, than once in a hundred times, and in addition we may have evidence that we shall reject *H* sufficiently often when it is false (32, pp. 290-1). 

When we try to interpret a *p*-value when comparing it to an error rate, we run into interpretive difficulties. It is generally defined as *the probability that the observed results, or more extreme results, if the null were true.*  In strict mathematical formulae:

$$ \mathrm{P}\left( X \geq x \mid H_{0}\right) $$ 

Where $$ X $$ is a random variable corresponding to some way of summarising the data, and $$ x $$ is the observed value of $$ X $$. 

What sticks out is that *p*-values are not part of any formal inferential calculus, their meaning and interpretation are elusive. 
## The mistakes/misconceptions

### Policies and decisions should be based on whether a _p_-value is < 0.05

This is gravest, and most common mistake. It’s made by policy makers in government, by product managers in industry, scientists in labs, and researchers in the field. There’s two issues: experimenters game their experiments as the stakes rise, and solely relying on *p*-values ignores other, more valuable, evidence for making a decision.

Randomised controlled trials are the gold standard of evidence for an intervention. We run experiments to measure the magnitude and direction of the effect size.  Deciding a policy (or a feature rollout) based on the criteria that *p* < 0.05 is tantamount to saying the magnitude of the effect isn’t relevant. It must the cart before the horse: *you did the experiment in the first place to find out something else!* Beliefs and actions cannot come directly from statistical results, and especially results from a single experiment. As a subject matter expert you will need to combine the formal results with your domain knowledge and come to a conclusion. 

Sometimes the conclusion could be that the null hypothesis is still true, despite a significant result; and sometimes the treatment works despite the test statistic disagreeing. Hypothesis tests aren’t the only form of evidence out there, and you will have to gather many different strands to make a decision.

It’s only half wrong to think that a policy should be made based on sound statistical evidence. But deciding upfront that if *p* < 0.05 then your research program is a success, and it’s a failure otherwise is unhealthy. It leads researchers to end-gain: the tendency to focus our actions on the outcomes, at the expense of the process. 

Researchers cut corners once they see the process as secondary. Look, I get it. Lots of good things happen to you if you can demonstrate that an effect with *p* < 0.05. If you’re a product manager, you can roll out a feature on a website and stick the $$$ you made your company on your CV. As a psychologist or economist you can write books and give after diner talks. If pure science is your muse, then a Nobel, or at least a Professorship, could await you. It’s totally human to want these things, but sacrificing the quality of your processes only undermines your goals. Just don’t it!

### *p* = 0.05 means that the null has a 5% chance of being true.

When we calculate the *p*-value, we assume the null is true. So it cannot also be the probability that the null hypothesis is false. 
###	_p_ > 0.05 means the alternative is true?

All this means is that the data you have collated are consistent with your null hypothesis, not that either hypothesis is true. The effect size you record in an experiment is the best estimate of the effect of your internet, regardless of the *p*-value. You would be better served concentrating on the plausibility of the effect size, than worrying about the test statistic.
#### If you can’t reject the null multiple times it’s true.

Not so: you can combine *p*-values with [Fisher’s method.](https://en.wikipedia.org/wiki/Fisher%27s_method) Roughly, for example with 2 *p*-values, you calculate $$ p = 2 p_1 p_2 $$ as the combined test statistic. This can make two null results significant!

#### If you can’t reject *this* null, you can’t *reject* any null.

Here’s another to melt your brain (courtesy of ~[Georgi Georgiev](https://blog.analytics-toolkit.com/2017/case-non-inferiority-designs-ab-testing/#authorStart)~):

> Say we have observed two groups of 4,000 users each in a randomized controlled experiment. The control converts at 10%, the variant at 11%, a 10% observed relative improvement. Is the result statistically significant at the 95% level, with a one-sided z-test?
> 
> No one can answer that question! What is missing from it is the null hypothesis. With a “classical” null hypothesis of “the difference between the variant and control is 0 or negative”, the answer is “no” (z-value corresponds to 92.77% significance). 
> 
> However, with a null hypothesis of “the variant is 2% worse than the control, or more”, the A/B test becomes significant at the 95% level (96.6%) with the lower bound of a 95% confidence interval for the difference in proportions at -1.3%.

Maybe you were trying to reject the wrong null all along.

### p < 0.05 is operationally important

*p*-values carry no information about the size of your effect, and that it what is operationally important. 


