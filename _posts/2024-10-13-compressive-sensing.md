---
layout: default
title:  "An introduction to Compressive Sensing."
subtitle: Short, fat matricesa are useful, actually.
date:   2024-10-09
categories: Compressive sensing.
---

# Introduction \label{sec:csinto}

This post discusses Compressive Sensing \gls{cs}: an alternative signal acquisition method to Nyquist sampling, which is capable of accurate sensing at rates well below those predicted by the Nyquist theorem. This strategy hinges on using the structure of a signal, and the fact that many signals we are interested in can be compressed successfully. Thus, Compressive Sensing acquires the most informative parts of a signal directly.

The first section surveys the mathematical foundation of CS. It covers the Restricted Isometry Property and Stable Embeddings - necessary and sufficient conditions on which a signal can be successfully acquired. Informally, these conditions suggest that the sensing operator preserves pairwise distances between points when projected to a (relatively) low dimensional space from a high dimensional space. We then discuss which operators satisfy this condition, and why. In particular, random ensembles such as the Bernoulli/Gaussian ensembles satisfy this property - we discuss how this can be applied to the problem of wideband spectrum sensing. We also survey a small amount of the theory of Wishart matrices.

The section concludes with an overview of reconstruction algorithms for CS - methods for unpacking the original signal from its compressed representation. We give insight into the minimum number of samples for reconstruction. We survey convex, greedy, and Bayesian approaches; as well as algorithms which are blends of all three. The choice of algorithm is affected by the amount of undersampling required for system performance, the complexity of the algorithm itself, and the desired reconstruction accuracy. In general: the more complex the algorithm, the better the reconstruction accuracy. The allowable undersampling depends upon the signal itself, and the prior knowledge available to the algorithm. Greedy algorithms are the simplest class, making locally optimal updates within a predefined dictionary of signal atoms. Greedy algorithms have relatively poor performance, yet are the fastest algorithms. Convex algorithms are the next most complex, based upon minimising a global functional of the signal. This class is based upon generalised gradient descent, and has no fixed number of steps. Finally, Bayesian algorithms are the slowest and most complex, but offer the best performance - both in terms of undersampling (as these algorithms incorporate prior knowledge in an elegant way), and in terms of reconstruction accuracy.

We survey some distributed approaches to compressive sensing, in particular some models of joint sparsity, and joint sparsity with innovations.

Finally, we survey some of the approaches to wideband spectrum sensing based upon compressive sensing. In particular we survey the Random Demodulator and the Modulated Wideband converter. Both of these systems make use of low-frequency chipping sequences (also used in spread spectrum communication systems). These low-frequency sequences provide the basis for CS - several different sequences each convoluted with the signal are sufficient to accurately sense the signal.

# Preliminaries \label{sec:prelims}

Compressive sensing is a modern signal acquisition technique in which randomness is used as an effective sampling strategy. This is in contrast to traditional, or Nyquist, sampling, which requires that a signal is sampled at regular intervals. The motivation for this new method comes from two disparate sources: data compression and sampling hardware design.

The work of Shannon, Nyquist, and Whittaker \cite{unser2000sampling,} has been an extraordinary success - digital signal processing enables the creation of sensing systems which are cheaper, more flexible, and which offer superior performance to their analogue counterparts. For example, radio dongles, such as those which support the RTLSDR standard, which can process millions of samples per second, can now be bought for as little as £12. However, the sampling rates underpinning these advances have a doubling time of roughly 6 years - this is due to physical limitations in Analogue to Digital conversion hardware. Specifically, these devices will always be limited in bandwidth and dynamic range (number of bits), whilst applications are creating a deluge of data to be processed downstream.

Data compression means that in practice many signals encountered 'in the wild' can be fully specified by much fewer bits than required by the Nyquist sampling theorem. This is either a natural property of the signals, for example, images have large areas of similar pixels, or as a conscious design choice, as with training sequences in communication transmissions. These signals are not statistically white, and so these signals may be compressed (to save on storage). For example, lossy image compression algorithms can reduce the size of a stored image to about 1% of the size required by Nyquist sampling. In fact, the JPEG standard uses Wavelets to exploit the inter-pixel redundancy of images.

Whilst this vein of research has been extraordinarily successful, it poses the question: if the reconstruction algorithm is able to reconstruct the signal from this compressed representation, why collect all the data in the first place when most of the information can be thrown away?

Compressed Sensing answers these questions by way of providing an alternative signal acquisition method to the Nyquist theorem. Specifically, situations are considered where fewer samples are collected than traditional sensing schemes. That is, in contrast to Nyquist sampling, Compressive Sensing is a method of measuring the informative parts of a signal directly without acquiring unessential information at the same time.

These ideas have not come out of the ether recently however. Prony, in 1795, \cite{prony1795essai}, proposed a method for estimating the parameters of a number of exponentials corrupted by noise. This work was extended by Caratheodory in 1907 \cite{Caratheodory1907}, who proposed in the 1900s a method for estimating a linear combination of $$ k $$ sinusoids for the state at time $$ 0 $$ and any other $$ 2k $$ points. In the 1970s, Geophysicists proposed minimising the $$ \ell_1 $$-norm to reconstruct the structure of the earth from seismic measurements. Clarebuot and Muir proposed in 1973, \cite{claerbout1973robust}, using the $$ \ell_1 $$-norm as an alternative to Least squares. Whilst Taylor, Banks, and McCoy showed in \cite{taylor1979deconvolution} how to use the $$ \ell_1 $$-norm to deconvolve spike trains (used for reconstructing layers in the earth). Santosa and Symes in \cite{Santosa1986} introduced the constrained $$ \ell_1 $$-program to perform the inversion of band-limited reflection seismograms. The innovation of \gls{cs} is to tell us under which circumstances these problems are tractable.

The key insight in CS is that for signals which are sparse or compressible - signals which are non-zero at only a fraction of the indices over which they are supported, or signals which can be described by relatively fewer bits than the representation they are traditionally captured in - may be measured in a non-adaptive way through a measurement system which is orthogonal to the signal's domain.

Examples of sparse signals are:

1. A sine wave at frequency $$ \omega $$ is defined as a single spike in the frequency domain yet has an infinite support in the time domain.
2. An image will have values for every pixel, yet the wavelet decomposition of the image will typically only have a few non-zero coefficients.

Informally, CS posits that for $$ s $$-sparse signals $$ \alpha \in \mathbb{R}^{n} $$ (signals with $$ s $$ non-zero amplitudes at unknown locations) - $$ \mathcal{O}(s \log{n}) $$ measurements are sufficient to exactly reconstruct the signal.

In practice, this can be far fewer samples than conventional sampling schemes. For example, a megapixel image requires 1,000,000 Nyquist samples but can be perfectly recovered from 96,000 compressive samples in the wavelet domain \cite{candes2008introduction}.

The measurements are acquired linearly, by forming inner products between the signal and some arbitrary sensing vector:

$$
y_i = \langle \alpha, \psi_i \rangle
$$
\label{inner-product-repr}

or

$$
y = \Psi \alpha
$$
\label{vector-repr}

where $$ y_i $$ is the $$ i^{th} $$ measurement, $$ \alpha \in \mathbb{R}^n $$ is the signal, and $$ \psi_i $$ is the $$ i^{th} $$ sensing vector. We pass to \eqref{vector-repr} from \eqref{inner-product-repr}, by concatenating all the $$ y_i $$ into a single vector. Thus the matrix $$ \Psi $$ has the vectors $$ \psi_i $$ as columns.

If $$ \alpha $$ is not $$ s $$-sparse in the natural basis of $$ y $$, then we can always transform $$ \alpha $$ to make it sparse in some other basis:

$$
x = \Phi \alpha
$$
\label{vector-repr-2}

Note that the measurements may be corrupted by noise, in which case our model is:

$$
y = Ax + e
$$
\label{CSequation}

where $$ e \in \mathbb{R}^m $$, and each component is sampled from a $$ \mathcal{N}(0, 1/n) $$ distribution. Here we have $$ A = \Psi \Phi \alpha \in \mathbb{R}^{n \times m} $$, i.e., a basis in which the signal $$ x \in \mathbb{R}^n $$ will be sparse.

We require that sensing vectors satisfy two technical conditions (described in detail below): an Isotropy property, which means that components of the sensing vectors have unit variance and are uncorrelated, and an Incoherence property, which means that sensing vectors are almost orthogonal. Once the set of measurements have been taken, the signal may be reconstructed from a simple linear program. We describe these conditions in detail in the next section.

## RIP and Stable Embeddings

We begin with a formal definition of sparsity:

\begin{definition}[Sparsity]

A high-dimensional signal is said to be $$s$$-sparse, if at most $$s$$ coefficients $$x_i$$ in the linear expansion 

$$
\alpha = \sum_{i=1}^{n} \phi_i x_i 
$$
\label{sparse-basis-expansion}

are non-zero, where $$x \in \mathbb{R}$$, $$\alpha \in \mathbb{R}$$, and $$\phi_i$$ are a set of basis functions of $$\mathbb{R}^n$$.

We can write \eqref{sparse-basis-expansion} as:

$$
\alpha = \Phi x
$$
\label{def:alpha}

We can make the notion of sparsity precise by defining $$\Sigma_s$$ as the set of $$s$$-sparse signals in $$\mathbb{R}^n$$:

$$
\Sigma_s = \{ x \in \mathbb{R}^n : |\mathrm{supp}(x)| \leq s \}
$$

where $$\mathrm{supp}(x)$$ is the set of indices on which $$x$$ is non-zero.

\end{definition}

\begin{figure*}[h]
\centering
\includegraphics[height=7cm, width=\textwidth]{compressive_sensing_example.jpg}
\caption{A visualisation of the Compressive Sensing problem as an under-determined system. Image from \cite{cstutwatkin}}
\label{l1l2}
\end{figure*}

We may not be able to directly obtain these coefficients $$x$$, as we may not possess an appropriate measuring device, or one may not exist, or there is considerable uncertainty about where the non-zero coefficients are.

Given a signal $$\alpha \in \mathbb{R}^n$$, a matrix $$A \in \mathbb{R}^{m \times n}$$, with $$m \ll n$$, we can acquire the signal via the set of linear measurements:

$$
y = \Psi \alpha = \Psi\Phi x = Ax
$$
\label{cs-model}

where we have combined \eqref{vector-repr} and \eqref{def:alpha}. In this case, $$A$$ represents the sampling system (i.e., each column of $$A$$ is the product of $$\Phi$$ with the columns of $$\Psi$$). We can work with the abstract model \eqref{cs-model}, bearing in mind that $$x$$ may be the coefficient sequence of the object in the proper basis.

In contrast to classical sensing, which requires that $$m = n$$ for there to be no loss of information, it is possible to reconstruct $$x$$ from an under-determined set of measurements as long as $$x$$ is sparse in some basis.

There are two conditions the matrix $$A$$ needs to satisfy for recovery below Nyquist rates:

1. Restricted Isometry Property.
2. Incoherence between sensing and signal bases.

\begin{definition}[RIP]\label{def:RIP}
We say that a matrix $$A$$ satisfies the RIP of order $$s$$ if there exists a $$\delta \in (0, 1)$$ such that for all $$x \in \Sigma_s$$:

$$
(1 - \delta) \|x\|_2^2 \leq \|Ax\|_2^2 \leq (1 + \delta) \|x\|_2^2
$$

i.e., $$A$$ approximately preserves the lengths of all $$s$$-sparse vectors in $$\mathbb{R}^n$$.

\label{def:RIP}
\end{definition}

\begin{remark}
Although the matrix $$A$$ is not square, the RIP (\ref{def:RIP}) ensures that $$A^TA$$ is close to the identity, and so $$A$$ behaves approximately as if it were orthogonal. This is formalised in the following lemma from \cite{shalev2014understanding}:

\begin{lemma}[Identity Closeness \cite{shalev2014understanding}]
Let $$A$$ be a matrix that satisfies the RIP of order $$2s$$ with RIP constant $$\delta$$. Then for two disjoint subsets $$I, J \subset [n]$$, each of size at most $$s$$, and for any vector $$u \in \mathbb{R}^n$$:

$$
\langle Au_I, Au_J \rangle \leq \delta \|u_I\|_2 \|u_J\|_2
$$

where $$u_I$$ is the vector with component $$u_i$$ if $$i \in I$$ and zero elsewhere.

\end{lemma}

\end{remark}

\begin{remark} \label{rem:rip-delta-comment}
The restricted isometry property is equivalent to stating that all eigenvalues of the matrix $$A^TA$$ are in the interval $$[1 - \delta, 1 + \delta]$$. Thus, the meaning of the constant $$\delta$$ in (\ref{def:RIP}) is now apparent. $$\delta$$ is called the \textit{restricted isometry constant} in the literature.

The constant $$\delta$$ in \eqref{def:RIP} measures how close to an isometry the action of the matrix $$A$$ is on vectors with a few non-zero entries (as measured in $$\ell_2$$ norm). For random matrices $$A$$ where the components are drawn from a $$\mathcal{N}(0, 1/n)$$ distribution, $$\delta < \sqrt{2} - 1$$ \cite{candes2008restricted}.
\end{remark}

\begin{remark} [Information Preservation \cite{davenport2010signal}]
A necessary condition to recover all $$s$$-sparse vectors from the measurements $$Ax$$ is that $$Ax_1 \neq Ax_2$$ for any pair $$x_1 \neq x_2$$, $$x_1, x_2 \in \Sigma_s$$, which is equivalent to:

$$
\|A(x_1 - x_2)\|_2^2 > 0
$$

This is guaranteed as long as $$A$$ satisfies the RIP of order $$2s$$ with constant $$\delta$$. The vector $$x_1 - x_2$$ will have at most $$2s$$ non-zero entries, and so will be distinguishable after multiplication with $$A$$. To complete the argument, take $$x = x_1 - x_2$$ in definition \ref{def:RIP}, guaranteeing:

$$
\|A(x_1 - x_2)\|_2^2 > 0
$$

and requiring the RIP order of $$A$$ to be $$2s$$.
\end{remark}

\begin{remark} [Stability \cite{davenport2010signal}]
We also require that the dimensionality reduction of compressed sensing preserves relative distances: that is, if $$x_1$$ and $$x_2$$ are far apart in $$\mathbb{R}^n$$, then their projections $$Ax_1$$ and $$Ax_2$$ are far apart in $$\mathbb{R}^m$$. This will guarantee that the dimensionality reduction is robust to noise.
\end{remark}

A requirement on the matrix $$A$$ that satisfies both of these conditions is the following:

\begin{definition}[$$\delta$$-stable embedding \cite{davenport2010signal}]
We say that a mapping is a $$\delta$$-stable embedding of $$U,V \subset \mathbb{R}^n$$ if

$$
(1 - \delta) \|u - v\|_2^2 \leq \|Au - Av\|_2^2 \leq (1 + \delta) \|u - v\|_2^2
$$

for all $$u \in U$$ and $$v \in V$$.

\label{def:d-stable}
\end{definition} 

\begin{remark}[\cite{davenport2010signal}]
Note that a matrix $$A$$, satisfying the RIP of order $$2s$$, is a $$\delta$$-stable embedding of $$\Sigma_s, \Sigma_s$$. 
\end{remark}

\begin{remark}[\cite{davenport2010signal}]
Definition \ref{def:d-stable} has a simple interpretation: the matrix $$A$$ must approximately preserve Euclidean distances between all points in the signal model $$\Sigma_s$$.
\end{remark}

# Incoherence

Given that we know a basis in which our signal is sparse, $$\phi$$, how do we choose $$\psi$$ so that we can accomplish this sensing task? In classical sensing, we choose $$\psi_k$$ to be the set of $$T_s$$-spaced delta functions (or equivalently the set of $$1/T_s$$ spaced delta functions in the frequency domain). A simple set of $$\psi_k$$ would be to choose a (random) subset of the delta functions above.

In general, we seek waveforms in which the signals' representation would be dense.

\begin{definition}[Incoherence]
A pair of bases is said to be incoherent if the largest projection of two elements between the sensing ($$\psi$$) and representation ($$\phi$$) basis is in the set $$[1, \sqrt{n}]$$, where $$n$$ is the dimension of the signal. The coherence of a set of bases is denoted by $$\mu$$.
\end{definition}

Examples of pairs of incoherent bases are:

- Time and Fourier bases: Let $$\Phi = \mathbf{I}_n$$ be the canonical basis and $$\Psi = \mathbf{F}$$ with $$\psi_i = n^{-\frac{1}{2}} e^{i \omega k}$$ be the Fourier basis, then $$\mu(\phi, \psi) = 1$$. This corresponds to the classical sampling strategy in time or space.
- Consider the basis $$\Phi$$ to have only entries in a single row, then the coherence between $$\Phi$$ and any fixed basis $$\Psi$$ will be $$\sqrt{n}$$.
- Random matrices are incoherent with any fixed basis $$\Psi$$. We can choose $$\Phi$$ by creating $$n$$ orthonormal vectors from $$n$$ vectors sampled independently and uniformly on the unit sphere. With high probability $$\mu = \sqrt{n \log n}$$. This extends to matrices whose rows are created by sampling independent Gaussian or Bernoulli random vectors.

This implies that sensing with incoherent systems is good (in the sine wave example above, it would be better to sample randomly in the time domain as opposed to the frequency domain), and efficient mechanisms ought to acquire correlations with random waveforms (e.g., white noise).

\begin{theorem}[Reconstruction from Compressive measurements \cite{Candes2006}]
Fix a signal $$f \in \mathbb{R}^n$$ with a sparse coefficient basis, $$x_i$$ in $$\phi$$. Then a reconstruction from $$m$$ random measurements in $$\psi$$ is possible with probability $$1 - \delta$$ if:

$$
m \geq C \mu^2(\phi, \psi) S \log \left( \frac{n}{\delta} \right)
$$
\label{minsamples}

where $$\mu(\phi, \psi)$$ is the coherence of the two bases, and $$S$$ is the number of non-zero entries on the support of the signal.
\end{theorem}

## Random Matrix Constructions \label{sec:mtx-contruction}

To construct matrices satisfying definition \eqref{def:d-stable}, given $$m, n$$ we generate $$A$$ by $$A_{ij}$$ being i.i.d random variables from distributions with the following conditions \cite{davenport2010signal}:

\begin{condition}[Norm preservation]
$$
\mathbb{E} A_{ij}^2 = \frac{1}{m}
$$
\label{cond:norm-pres}
\end{condition}

\begin{condition}[sub-Gaussian]
There exists a $$C > 0$$ such that:
$$
\mathbb{E}\left( e^{A_{ij}t} \right) \leq e^{C^2 t^2 /2}
$$
\label{cond:sub-Gauss}
for all $$t \in \mathbb{R}$$.
\end{condition}

\begin{remark}
The term $$\mathbb{E}\left( e^{A_{ij}t} \right)$$ in \eqref{cond:sub-Gauss} is the *moment generating function* of the sensing matrix. Condition \eqref{cond:sub-Gauss} says that the moment-generating function of the distribution producing the sensing matrix is dominated by that of a Gaussian distribution, which is also equivalent to requiring that the tails of our distribution decay at least as fast as the tails of a Gaussian distribution. Examples of sub-Gaussian distributions include the Gaussian distribution, the Rademacher distribution, and the uniform distribution. In general, any distribution with bounded support is sub-Gaussian. The constant $$C$$ measures the rate of fall off of the tails of the sub-Gaussian distribution.
\end{remark}

Random variables $$A_{ij}$$ satisfying conditions \eqref{cond:norm-pres} and \eqref{cond:sub-Gauss} satisfy the following concentration inequality \cite{baraniuk2008simple}:

\begin{lemma}[sub-Gaussian \cite{baraniuk2008simple}]
$$
\mathbb{P}\Big( \biggl\lvert \|Ax\|_2^2 - \|x\|_2^2 \biggr\rvert \geq \varepsilon \|x\|_2^2 \Big) \leq 2e^{-cM\varepsilon^2}
$$
\label{cond:sub-Gauss concetration}
\end{lemma}

\begin{remark}
Lemma \ref{cond:sub-Gauss concetration} says that sub-Gaussian random variables are random variables such that for any $$x \in \mathbb{R}^n$$, the random variable $$\|Ax\|_2^2$$ is highly concentrated about $$\|x\|_2^2$$.
\end{remark}

Then in \cite{baraniuk2008simple} the following theorem is proved:

\begin{theorem}
Suppose that $$m$$, $$n$$ and $$0 < \delta < 1$$ are given. If the probability distribution generating $$A$$ satisfies condition \eqref{cond:sub-Gauss}, then there exist constants $$c_1, c_2$$ depending only on $$\delta$$ such that the RIP \eqref{def:RIP} holds for $$A$$ with the prescribed $$\delta$$ and any $$s \leq \frac{c_1 n}{\log{n/s}}$$ with probability $$\geq 1-2e^{-c_2n}$$.
\end{theorem}

For example, if we take $$A_{ij} \sim \mathcal{N}(0, 1/m)$$, then the matrix $$A$$ will satisfy the RIP, with probability \cite{baraniuk2008simple}:

$$
\geq 1 - 2\left(\frac{12}{\delta}\right)^k e^{-c_0\frac{\delta}{2}n}
$$

where $$\delta$$ is the RIP constant, $$c_0$$ is an arbitrary constant, and $$k$$ is the sparsity of the signal being sensed.

## Wishart Matrices \label{sec:wishart}

Let $$\{X_i\}_{i=1}^r$$ be a set of i.i.d. $$1 \times p$$ random vectors drawn from the multivariate normal distribution with mean 0 and covariance matrix $$H$$.

$$
X_i = \left(x_1^{(i)}, \ldots, x_p^{(i)}\right) \sim N\left(0, H\right)
$$

We form the matrix $$X$$ by concatenating the $$r$$ random vectors into an $$r \times p$$ matrix.

\begin{definition}[Wishart Matrix]
Let 

$$
W = \sum_{j=1}^r X_j X_j^T = X X^T
$$

Then $$W \in \mathbb{R}^{r \times r}$$ has the Wishart distribution with parameters:

$$
W_r(H, p)
$$

where $$p$$ is the number of degrees of freedom.
\end{definition}

\begin{remark}
This distribution is a generalization of the Chi-squared distribution if $$p = 1$$ and $$H = I$$.
\end{remark}

\begin{theorem}[Expected Value] \label{thm:wishart-mean}
$$
\mathbb{E}(W) = rH
$$
\end{theorem}

\begin{proof}
\begin{align*}
\mathbb{E}(W) &= \mathbb{E}\left(\sum_{j=1}^r X_j X_j^T\right) \\
&= \sum_{j=1}^r \mathbb{E}(X_j X_j^T) \\
&= \sum_{j=1}^r \left( \mathrm{Var}(X_j) + \mathbb{E}(X_j) \mathbb{E}(X_j^T) \right) \\
&= rH
\end{align*}
Where the last line follows as $$X_j$$ is drawn from a distribution with zero mean.
\end{proof}

\begin{remark}
The matrix $$M = A^T A$$, where $$A$$ is constructed by the methods from section \ref{sec:mtx-contruction}, will have a Wishart distribution. In particular, it will have:

$$
\mathbb{E}(M) = \frac{1}{m} I_n
$$
\label{remark: exp AtA}
\end{remark}

The joint distribution of the eigenvalues is given by \cite{levequeMatrices}:

$$
p\left(\lambda_1, \ldots, \lambda_r\right) = c_r \prod_{i=1}^r e^{-\lambda_i} \prod_{i<j} \left(\lambda_i - \lambda_j\right)^2
$$

The eigenvectors are uniform on the unit sphere in $$\mathbb{R}^r$$.

## Reconstruction Objectives

Compressive sensing places the computational load on reconstructing the coefficient sequence $$x$$, from the set of compressive samples $$y$$. This is in contrast to Nyquist sampling, where the bottleneck is in obtaining the samples themselves—reconstructing the signal is a relatively simple task.

Many recovery algorithms have been proposed, and all are based upon minimizing some functional of the data. This objective is based upon two terms: a data fidelity term, minimizing the discrepancy between the reconstructed and true signal, and a regularization term—biasing the reconstruction towards a class of solutions with desirable properties, for example sparsity. Typically the squared error $$\frac{1}{2} \|y - Ax\|_2^2$$ is chosen as the data fidelity term, while several regularization terms have been introduced in the literature.

A particularly important functional is:

$$
\arg \min_x \|x\|_1 \text{ s.t. } y = Ax
$$
\label{program:bp}

This is known as Basis Pursuit \cite{Chen1998a}, with the following program known as the LASSO \cite{tibshirani1996regression} as a noisy generalization:

$$
\arg \min_x \frac{1}{2} \|Ax - y\|_2^2 + \lambda \|x\|_1
$$
\label{program:lasso}

The statistical properties of LASSO have been well studied. The program performs both regularization and variable selection: the parameter $$\lambda$$ trades off data fidelity and sparsity, with higher values of $$\lambda$$ leading to sparser solutions.

The LASSO shares several features with Ridge regression \cite{hoerl1970ridge}:

$$
\arg \min_x \frac{1}{2} \|Ax - y\|_2^2 + \lambda \|x\|_2^2
$$
\label{program:Ridge-regression}

and the Non-negative Garrote \cite{breiman1995better}, used for best subset regression:

$$
\arg \min_x \frac{1}{2} \|Ax - y\|_2^2 + \lambda \|x\|_0
$$
\label{program:ell0}

The solutions to these programs can all be related to each other—it can be shown \cite{hastie2005elements} that the solution to \eqref{program:lasso} can be written as:

$$
\hat{x} = S_{\lambda}(x^{OLS}) = x^{OLS} \text{ sign}(x_i - \lambda)
$$
\label{soln:lasso}

where $$x^{OLS} = (A^T A)^{-1} A^T y$$ is the ordinary least squares solution, whereas the solution to Ridge regression can be written as:

$$
\hat{x} = \left( 1 + \lambda \right)^{-1} x^{OLS}
$$
\label{soln:ridge}

and the solution to the best subset regression \eqref{program:ell0} where $$\|x\|_0 = \{ |i| : x_i \neq 0 \}$$, can be written as:

$$
\hat{x} = H_{\lambda}(x^{OLS}) = x^{OLS} \mathbb{I}\left( |x^{OLS}| > \lambda \right)
$$
\label{soln:l0}

where $$\mathbb{I}$$ is the indicator function. From \eqref{soln:l0} and \eqref{soln:ridge}, we can see that the solution to \eqref{program:lasso}, \eqref{soln:lasso}, translates coefficients towards zero by a constant factor, and sets coefficients to zero if they are too small; thus, the LASSO is able to perform both model selection (choosing relevant covariates) and regularization (shrinking model coefficients).

![Solutions to the Compressive Sensing optimization problem intersect the $$l_1$$ norm at points where all components (but one) of the vector are zero (i.e., it is sparsity promoting) \cite{Tibshirani1996}.](l1l2.jpg)
\label{fig:l1l2}

Figure \ref{fig:l1l2} provides a graphical demonstration of why the LASSO promotes sparse solutions. \eqref{program:lasso} can also be thought of as the best convex approximation of the $$\ell_0$$ problem \eqref{program:ell0}, as the $$\ell_1$$-norm is the convex hull of the points defined by $$\|x\|_p$$ for $$p < 1$$ as $$p \rightarrow 0$$.

Other examples of regularizers are:

- **Elastic Net**: This estimator is a blend of both \eqref{program:lasso} and \eqref{program:Ridge-regression}, found by minimizing:

$$
\arg \min_x \frac{1}{2} \|Ax - y\|_2^2 + \lambda \|x\|_2^2 + \mu \|x\|_1
$$
\label{program:enat}

The estimate has the benefits of both Ridge and LASSO regression: feature selection from the LASSO, and regularization for numerical stability (useful in the under-determined case we consider here) from Ridge regression. The Elastic Net will outperform the LASSO \cite{zou2005regularization} when there is a high degree of collinearity between coefficients of the true solution.

- **TV regularization**:

$$
\arg \min_x \frac{1}{2} \|Ax - y\|_2^2 + \lambda \|\nabla x\|_1
$$
\label{program:tc}

This type of regularization is used when preserving edges while simultaneously denoising a signal is required. It is used extensively in image processing, where signals exhibit large flat patches alongside large discontinuities between groups of pixels.

- Candes and Tao in \cite{candes2007dantzig} propose an alternative functional:

$$
\min_{x \in \mathbb{R}^n} \|x\|_1 \text{ s.t. } \|A^T(Ax - y)\|_{\infty} \leq t\sigma
$$
\label{program:danzig}

with $$t = c \sqrt{2 \log{n}}$$. Similarly to the LASSO, this functional selects sparse vectors consistent with the data, in the sense that the residual $$r = y - Ax$$ is smaller than the maximum amount of noise present. In \cite{candes2007dantzig} it was shown that the $$l_2$$ error of the solution is within a factor of $$\log{n}$$ of the ideal $$l_2$$ error. More recent work by Bikel, Ritov, and Tsybakov \cite{bickel2009simultaneous} has shown that the LASSO enjoys similar properties.

## Reconstruction Algorithms
Broadly, reconstruction algorithms fall into three classes: convex-optimisation/linear programming, greedy algorithms, and Bayesian inference. Convex optimisation methods offer better performance, measured in terms of reconstruction accuracy, at the cost of greater computational complexity. Greedy methods are relatively simpler, but don't have the reconstruction guarantees of convex algorithms. Bayesian methods offer the best reconstruction guarantees, as well as uncertainty estimates about the quality of reconstruction, but come with considerable computational complexity.

| Algorithm Type  | Accuracy | Complexity | Speed |
|-----------------|----------|------------|-------|
| Greedy          | Low      | Low        | Fast  |
| Convex          | Medium   | Medium     | Medium|
| Bayesian        | High     | High       | Slow  |

### Convex Algorithms
Convex methods cast the optimisation objective either as a linear program with linear constraints or as a second-order cone program with quadratic constraints. Both of these types of programs can be solved with first-order interior point methods. However, their practical application to compressive sensing problems is limited due to their polynomial dependence upon the signal dimension and the number of constraints.

Compressive sensing poses a few difficulties for convex optimisation-based methods. In particular, many of the unconstrained objectives are non-smooth, meaning methods based on descent down a smooth gradient are inapplicable.

To overcome these difficulties, a series of algorithms originally proposed for wavelet-based image de-noising have been applied to CS, known as iterative shrinkage methods. These have the desirable property that they boil down to matrix-vector multiplications and component-wise thresholding.

Iterative shrinkage algorithms replace searching for a minimal facet of a complex polytope with an iteratively denoised gradient descent. The choice of the (component-wise) denoiser is dependent upon the regulariser used in $$ \eqref{program:lasso} $$. These algorithms have an interpretation as Expectation-Maximisation \cite{figueiredo2003algorithm} — where the E-step is performed as gradient descent, and the M-step is the application of the denoiser.

#### Iterative Soft Thresholding Algorithm (IST)
```python
\begin{algorithm}
\begin{algorithmic}[1]
\Procedure{IST}{$y,A, \mu, \tau, \varepsilon$}
\State $x^0 = 0$
\While{$\|x^{t} - x^{t-1}\|_2^2 \leq \varepsilon$}
\State $x^{t+1} \gets S_{\mu\tau}\left(x^t + \tau A^T z^t \right) $
\State $z^t \gets y - A x^t$
\EndWhile
\State \textbf{return} $x^{t+1}$
\EndProcedure
\end{algorithmic}
\caption{The Iterative Soft Thresholding Algorithm \cite{donoho1995noising}}\label{alg:IST}
\end{algorithm}
## Bayesian Algorithms
Bayesian methods reformulate the optimisation problem into an inference problem. These methods come with a unified theory and standard methods to produce solutions. The theory is able to handle hyper-parameters in an elegant way, provides a flexible modelling framework, and is able to provide desirable statistical quantities such as the uncertainty inherent in the prediction.

Previous sections have discussed how the weights $$x$$ may be found through optimisation methods such as basis pursuit or greedy algorithms. Here, an alternative Bayesian model is described.

Equation $$\eqref{CSequation}$$ implies that we have a Gaussian likelihood model: 

$$
p \left(y \mid z, \sigma^2 \right) = (2 \pi \sigma^2)^{-K/2} \exp{\left(- \frac{1}{2 \sigma^2} \|y - Ax|_{2}^{2} \right)}
$$

The above has converted the CS problem of inverting sparse weight $$x$$ into a linear regression problem with a constraint (prior) that $$x$$ is sparse.

To seek the full posterior distribution over $$x$$ and $$ \sigma^2 $$, we can choose a sparsity-promoting prior. A popular sparseness prior is the Laplace density function:

$$
p\left(x \mid \lambda\right) = \left(\frac{\lambda}{2}\right)^N \exp{-\lambda \sum_{i=1}^{N} |x_i|}
$$

Note that the solution to the convex optimisation problem $$\eqref{program:lasso}$$ corresponds to a maximum *a posteriori* estimate for $$x$$ using this prior. I.e., this prior is equivalent to using the $$l_1$$ norm as an optimisation function (see figure \ref{laplacenormal} \cite{Tibshirani1996}).

\begin{figure*}[h]
\centering
\includegraphics[height=7cm]{LaplaceandNormalDensity.png}
\caption{The Laplace ($$l_1$$-norm, bold line) and Normal ($$l_2$$-norm, dotted line) densities. Note that the Laplace density is sparsity-promoting as it penalises solutions away from zero more than the Gaussian density. \cite{Tibshirani1996}}
\label{laplacenormal}
\end{figure*}

The full posterior distribution on $$x$$ and $$\sigma^2$$ may be realised by using a hierarchical prior instead. To do this, define a zero-mean Gaussian prior on each element of $$e$$:

$$
p\left(e \mid a\right) = \prod_{i=1}^{N} \mathbb{N}\left(n_i \mid 0, \alpha_{i}^{-1}\right)
$$

where $$\alpha$$ is the precision of the distribution. A gamma prior is then imposed on $$\alpha$$:

$$
p\left(\alpha \mid a, b \right) = \prod_{i=1}^{N} \Gamma\left( \alpha_i \mid a, b \right)
$$

The overall prior is found by marginalising over the hyperparameters:

$$
p\left( x \mid a, b \right) = \prod_{i=1}^{N} \int_{0}^{\infty} \mathbb{N}\left(w_i \mid 0, \alpha_{i}^{-1}\right) \Gamma\left( \alpha_i \mid a, b \right)
$$

This integral can be done analytically and results in a Student-t distribution. Choosing the parameters $$a$$ and $$b$$ appropriately, we can make the Student-t distribution peak strongly around $$x_i = 0$$, i.e., sparsifying. This process can be repeated for the noise variance $$\sigma^2$$. The hierarchical model for this process is shown in figure \ref{fig:bayesiancs}. This model, and other CS models which do not necessarily have closed-form solutions, can be solved via belief-propagation \cite{Baron2010}, or via Monte-Carlo methods.

\begin{figure}[h]
\centering
\includegraphics[height=7cm]{bayesiancs.png}
\caption{The hierarchical model for the Bayesian CS formulation \cite{Ji2008}}
\label{fig:bayesiancs}
\end{figure}

However, as with all methodologies, Bayesian algorithms have their drawbacks. Most notable is the use of the most computationally complex recovery algorithms. In particular, MCMC methods suffer in high-dimensional settings, such as those considered in compressive sensing. There has been an active line of work to address this: most notably, Hamiltonian Monte Carlo (see \cite{neal2011mcmc}) — an MCMC sampling method designed to follow the typical set of the posterior density.

Belief propagation (BP) \cite{Yedidia2011} is a popular iterative algorithm, offering improved reconstruction quality and undersampling performance. However, it is a computationally complex algorithm. It is also difficult to implement. Approximate message passing (AMP) (figure \ref{alg:amp}) solves this issue by blending BP and iterative thresholding (figure \ref{alg:IST}). The algorithm proceeds like iterative thresholding but computes an adjusted residual at each stage. The final term in the update:

$$
z^{t+1} = y - Ax^t + \frac{\|x\|_0}{m} z^t
$$

comes from a first-order approximation to the messages passed by BP \cite{metzler2014denoising}. This is in contrast to the update from IST (figure \ref{alg:IST}):

$$
z^{t+1} = y - Ax^t
$$

The choice of prior is key in Bayesian inference, as it encodes all knowledge about the problem. Penalising the least-squares estimate with the $$\ell_1$$ norm,

## Compressive Estimation \label{sec:estimation}
In this section, we develop some intuition into constructing estimators for the signal $$s$$ directly from the compressive measurements:

\begin{theorem}

Given a set of measurements of the form:

$$
y = As + e
$$
\\
where $$A \in \re^{m \times n}$$, $$A_{ij} \sim \mathcal{N}\left(0,1/m\right)$$, and $$e \in \re^n$$ is AWGN, i.e., $$\sim N\left(0, \sigma^2 I\right)$$. We again assume that $$s$$ comes from a fixed set of models, parametrised by some set $$\Theta$$.

Then, the maximum likelihood estimator of $$s$$, for the case where $$s$$ can be expanded in an orthonormal basis $$s = \sum_{i=1}^n \alpha_i\phi_i$$:

$$
\hat{s} = \sum_{i=1}^n m \langle y, A\phi_i \rangle \phi_i
$$

\end{theorem}
\begin{proof}
The likelihood for this model is (as $$y$$ is a normal random variable):

$$
f\left(y \mid s\right) = \left(\frac{1}{\left(2\pi\right)^{n/2}}\right) \exp{\left(- \frac{\left(y-As\right)^T  \left(y-As\right)}{2} \right)}
$$

Taking the logarithm and expanding, we find

$$
\ln{f\left(y \mid s\right)} = -y^Ty - s^TA^TAs + 2\langle y, As \rangle + c
$$

which is equal to:

$$
\ln{f} = - \|y\|_2^2 - \|As\|_2^2 + 2\langle y, As \rangle
\label{log-like}
$$

(where the constant has been dropped). The first term of $$\eqref{log-like}$$ is constant, for the same reasons as in section $$\eqref{sec:estimation}$$. The term

$$
\|As\|_2^2 = \langle As, As \rangle
$$

can be written as

$$
\langle A^TAs, s\rangle
\label{ata}
$$

We will assume that $$A^TA$$ concentrates in the sense of \ref{cond:sub-Gauss concentration} and replace \ref{ata} with its expectation $$\ep{\left( \langle A^TAs, s\rangle \right)}$$

\begin{align*}
\ep{\left(\langle A^TAs, s\rangle\right)} &=  \ep{\sum_{i=1}^n (A^TAs)^T_i s_i} \\
&= \sum_{i=1}^n \ep{(A^TAs)_i s_i} \\
&= \sum_{i=1}^n \left(\frac{1}{m} e_i s_i\right)^T_i s_i \\
&= \frac{1}{m} \langle s, s \rangle
\end{align*}

because

$$
\ep{A^TA} = \frac{1}{m} I
$$

as it is a Wishart matrix (see section \ref{sec:prelims}). 
\\
So we can further approximate \eqref{log-like}:

$$
\ln{f\left(y \mid s\right)}  = - \|y\|_2^2 - \frac{1}{m} \|s\|_2^2 + 2 \langle y, As \rangle
\label{approx-log-like}
$$

The only non-constant part of \eqref{approx-log-like} is the third term, and so we define the estimator:

$$
\hat{s} = \argmax_{\Theta} \langle y , As\left(\Theta\right)\rangle
\label{eq: compressive-estimator}
$$
\end{proof}

\begin{corollary}
Consider the case where $$y = As$$ (no noise). Then

\begin{align*}
y^TA\phi_j &= \sum_i \alpha_i \phi_i^TA^TA\phi_j
\end{align*}

So 

\begin{align*}
y^TA\phi_j &= \sum_i \alpha_i \phi_i^TA^TA\phi_j \sim \frac{\alpha_i}{m} \delta_{ij}
\end{align*}

giving
 
$$
\widehat{\alpha_i} = m \left( y^T A \phi_j \right)
$$
\end{corollary}

\begin{remark}
The matrix $$M = A^TA$$ is the projection onto the row-space of $$A$$. It follows that $$\|Ms\|_2^2$$ is simply the norm of the component of $$s$$ which lies in the row-space of $$A$$. This quantity is at most $$\|s\|_2^2$$, but can also be $$0$$ if $$s$$ lies in the null space of $$A$$. However, because $$A$$ is random, we can expect that $$\|Ms\|_2^2$$ will concentrate around $$\sqrt{m/n} \|s\|_2^2$$ (this follows from the concentration property of sub-Gaussian random variables \eqref{cond:sub-Gauss concentration}).
\end{remark}

\begin{example}{Example: Single Spike}
We illustrate these ideas with a simple example: estimate which of $$n$$ frequencies $$s$$ is composed of.

A signal $$s \in \mathbb{R}^{300}$$ is composed of a single (random) delta function, with coefficients drawn from a Normal distribution (with mean 100, and variance 1), i.e., 

$$
s = \alpha_i \delta_i
$$
\\
with 

$$
a_i \sim \mathcal{N}\left(100, 1\right)
$$

and the index $$i$$ chosen uniformly at random from $$[1, n]$$.
\\
The signal was measured via a random Gaussian matrix $$A \in \mathbb{R}^{100 \times 300}$$, with variance $$\sigma^2 = 1/100$$, and the inner product between $$y = As$$ and all 300 delta functions projected onto $$\mathbb{R}^{100}$$ was calculated:

$$
\hat{\alpha}_j = m \langle (A \alpha_i \delta_i), A \delta_j \rangle
$$ 

We plot the $$\hat{\alpha_j}$$ below, figure \ref{fig:new_basis_25}, (red circles), with the original signal (in blue, continuous line). Note how the maximum of the $$\hat{\alpha_j}$$ coincides with the true signal.

\begin{figure}[h]
\centering
\includegraphics[height=7.3cm]{1spike_legend.jpg}
\label{fig:new_basis_25}
\caption{An example estimation of a single spike using Compressive Inference methods. Note how the maximum of the estimator $$\hat{\alpha}_j$$ corresponds to the true signal.}
\end{figure}
\end{example}

