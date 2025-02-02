---
layout: default
title:  Hypothesis Testing without assumptions
subtitle: What to do if you can't assume anything about your distributions.
date:   2025-01-15
categories: hypothesis-testing
toc: true
---

# Introduction

_The_ basic problem in statistics is comparing two samples of measurements. The basic issue is that you want to know whether these measurements came from the same source, or from different sources. In statistical language, we are comparing the null distribution—that the measurements came from the same source—against the alternative hypothesis —that the measurements come from distinct sources.

Very often statisticians assume some parametric form for the distributions. For example, normal distributions with known unknown means and the same variance. So we would be comparing:

$$
H_0: \mu_1 = \mu_2
$$

Against

$$
H_0: \mu_1 \neq \mu_2
$$

There are some downsides to this approach. Firstly, we are imposing an assumption on the data—that the measurements come from a distribution with a specific form—and we can never check the accuracy of that assumption. That is if we have gotten the correct parametric form the distributions. Secondly, the specific distributional form we assume is also unknowable. So we can never really know how wrong we are. 

The issue then is to find solutions to the following problem, without making assumptions about the data generating process of the measurements. 

*Problem 1* Let $x$ and $y$ be random variables defined on some topological space $\mathcal{X}$, with respective probability measures $p$ and $q$. Given two sets of observations $X = \left{x_1, x_2, \ldots x_m\right}$ and $Y = \left{y_1, y_2, \ldots y_n\right}$ drawn i.i.d from $p$ and $q$ respectively, can we decide whether $p \neq q$?

# Maximum Mean Discrepancy (MMD)

In general, MMD is defined by the idea of representing distances between distributions as distances between mean embeddings of features. We can define the notion of distance between probability distirbutions with the following lemma (from {% cite dudley2018real %}).

*Lemma 1* Let $\left(\mathcal{X}, d\right)$ be a metric space, and let $p$ and $q$ be two probability measures defined on $\mathcal{X}$. Then $p = q$ if and only if $\mathcal{E}_x\left(f\left(x\right)\right) = \mathcal{E}_y\left(f\left(y\right)\right)$ for all $f \in C\left(\mathcal{X}\right)$ where $C\left(\mathcal{X}\right)$ is the collection of bounded, continuous functions on $\mathcal{X}$.

The maximum mean discrepancy (MMD) is then defined based on a this class of functions:

*Definition 1* Let $mathcal{F}$ be a class of functions $f: \mathal{F} \rightarrow \mathcal{R}, and let $p, q, x, y, X, Y$ be defined as earlier. The MMD is then:

$$
MMD(p,q) := \mathrm{sup}_{f \in \mathcal{F}} \left( \mathcal{E}_x[f(x)] - \mathcal{E}_y[f(y)] \right)
$$

An empirical estimate of the MMD can be found by replacing the population expectations with their sample expectations based on the samples $X$ and $Y$ (although this is slightly biased):

$$
\hat{MMD(p,q)} := \mathrm{sup}_{f \in \mathcal{F}} \left( \frac{1}{m}\sum_{i=1}^m f(x_i) - \frac{1}{n}\sum_{i=1}^n f(y_i) \right)
$$

Although $C\left(\mathcal{X}\right)$ allows us to identify if $p = q$ the space has a 

$\phi:\mathcal{X} \rightarrow \mathcal{H}$, where $\mathcal{H}$ is some Hilbert space; this corresponds to a kernel by $k\left(x,y\right)=\left⟨\phi(x),\phi(y)\right⟩$. In general, the MMD is:

As one example, we might have 𝒳=ℝ^d and φ(x)=x, corresponding to a linear kernel. In that case:

MMD(P,Q) = ‖𝔼[X~P][φ(X)] - 𝔼[Y~Q][φ(Y)]‖ = ‖𝔼[X~P][X] - 𝔼[Y~Q][Y]‖_ℝd = ‖μ_P - μ_Q‖_ℝd

So this MMD is just the distance between the means of the two distributions. Matching distributions like this will match their means, though they might differ in their variance or in other ways.

## Special Case

Your case is slightly different: we have 𝒳=ℝ^d and ℋ=ℝ^p, with φ(x)=A'x, where A is a d×p matrix. So we have:

MMD(P,Q) = ‖𝔼[X~P][φ(X)] - 𝔼[Y~Q][φ(Y)]‖ = ‖𝔼[X~P][A'X] - 𝔼[Y~Q][A'Y]‖_ℝp = ‖A'𝔼[X~P][X] - A'𝔼[Y~Q][Y]‖_ℝp = ‖A'(μ_P - μ_Q)‖_ℝp

This MMD is the difference between two different projections of the mean. If p<d or the mapping A' otherwise isn't invertible, then this MMD is weaker than the previous one: it doesn't distinguish between some distributions that the previous one does.

## Stronger Distances

You can also construct stronger distances. For example, if 𝒳=ℝ and you use φ(x)=(x,x²) (giving a particular quadratic kernel), then the MMD becomes:

√[(𝔼X-𝔼Y)² + (𝔼X²-𝔼Y²)²]

And can distinguish not only distributions with different means but with different variances as well.

## Kernel Trick

And you can get much stronger than that: for general choices of kernel, you can use the kernel trick to compute the MMD:

MMD²(P,Q) = ‖𝔼[X~P]φ(X) - 𝔼[Y~Q]φ(Y)‖² = 
𝔼[X,X'~P]k(X,X') + 𝔼[Y,Y'~Q]k(Y,Y') - 2𝔼[X~P,Y~Q]k(X,Y)

It's then straightforward to estimate this with samples, for any kernel function k -- even ones where φ is infinite-dimensional, like the Gaussian kernel (also called "squared exponential" or "exponentiated quadratic"):

k(x,y) = exp(-1/(2σ²)‖x-y‖²)

If your choice of k is "characteristic," then the MMD becomes a proper metric on distributions: it's zero if and only if the two distributions are the same. (This is unlike when you use, say, a linear kernel, where two distributions with the same mean have zero linear-kernel MMD.) If you've heard of a "universal" kernel, those are characteristic, but there are a few kernels that are characteristic but not universal.

## Understanding the Name

Here's an explanation of the name, which is also useful for understanding the MMD.

For any kernel k:𝒳×𝒳→ℝ, there exists a feature map φ:𝒳→ℋ, where ℋ is a special Hilbert space called the reproducing kernel Hilbert space (RKHS) corresponding to k. This is a space of functions, f:𝒳→ℝ. These spaces satisfy a special key condition, called the reproducing property: ⟨f,φ(x)⟩=f(x) for any f∈ℋ.

The simplest example is the linear kernel k(x,y)=x⋅y. This can be "implemented" with ℋ=ℝ^d and φ(x)=x. But the RKHS is instead the space of linear functions f_x(t)=x⋅t, and φ(x)=f_x. The reproducing property is ⟨f_w,φ(x)⟩=⟨w,x⟩_ℝd.

In more complex settings, like a Gaussian kernel, f is a much more complicated function, but the reproducing property still holds.

## Alternative Characterization

Now, we can give an alternative characterization of the MMD:

MMD(P,Q) = ‖𝔼[X~P][φ(X)] - 𝔼[Y~Q][φ(Y)]‖ = 
sup{f∈ℋ:‖f‖≤1}⟨f,𝔼[X~P][φ(X)] - 𝔼[Y~Q][φ(Y)]⟩ = 
sup{f∈ℋ:‖f‖≤1}[𝔼[X~P][f(X)] - 𝔼[Y~Q][f(Y)]]

The second line is a general fact about norms in Hilbert spaces that follows immediately from Cauchy-Schwarz: sup{f:‖f‖≤1}⟨f,g⟩=‖g‖ is achieved by f=g/‖g‖.

The fourth line depends on a technical condition known as Bochner integrability, but is true e.g. for bounded kernels or distributions with bounded support.

This last line is why it's called the "maximum mean discrepancy" – it's the maximum, over test functions f in the unit ball of ℋ, of the mean difference between the two distributions. This is also a special case of an integral probability metric.