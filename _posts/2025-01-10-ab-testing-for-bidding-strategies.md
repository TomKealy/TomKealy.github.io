---
layout: default
title:  "AB testing Google Ads Bidding strategies"
date:   2025-01-15
categories: hypothesis-testing
---

{:toc}

# Introduction

Google (and Meta, and Twitter) serves ads to users browsing the internet. For example, when you make a Google search, you will see sponsored content at the top of your search results. Similarly if you brose Instagram, some of the images you see will be products to buy. The ads you see when you are browsing the internet are paid for by companies who want you to buy their products. The companies bid, in automated auctions, for their ads to be served to you as you browse.

A **bid** is the amount of money you are willing to spend for a single click on a specific keyword in the Google Ads campaign. Those bids determine where your ads appear in search results. Bidders compete for the highest possible position in Google search rankings for the specific search query. Every time someone does a search in Google, the system runs all offers through Google’s algorithm and ranks the ads according to how much particular advertisers are willing to pay for that term as well as what is its quality score. In short, it is the auction that determines which ads will appear at that moment in that space and your bid puts you in the auction.

Google and Meta provide several ways to control the outcome of each auction: interested parties need to specify a target Return on Ad Spend (tROAS). This is the target ammount that you wish to make on each ad. For example, if a business spends 200 USD and this generates 1,000 USD in revenue (or conversion value), then the ROAS is 500%. Setting a tROAS of 500% says that the business needs to make 5 dollars for every dollar that’s spent on Google Ads.

Alternatively a company could control a target Cost per Acquisition (tCPA). Choosing a tCP Abidding strategy will generate the maximum number of conversions using a campaign’s budget. It will also account for the desired target cost per action, which is chosen by the advertiser. For example, if an advertiser has chosen a 20 USD Target CPA, Google will adjust bids to drive conversions and aim to achieve that tCPA.

So you can choose ROAS if you want the maximum conversion value per conversions for your ad budget. You can choose tCPA if you want the maximum number of conversion for a given ad budget.

Companies who bid for these ad spaces are interested in optimising their **ad spend.** That is would a 'set and forget' strategy—where you set a _fixed_ tROAS value and leave it throughout a campaign—be outperformed by an adaptive strategy, where the tROAS is adjusted (possibly by an ML model) during the campaign. This is useful, because maybe you set the tCPA to bid 50 USD per conversion, buy you could have gotten away with paying 40 USE instead. A 'set and forget it' approach would therefore pay 10 USD extra, and over 1000s of conversions this strategy can cost millions extra over the course of a campaign.

One way to decide whether either strategy is better, is to conduct an A/B test. The challenge is that we cannot directly control the amount of money each strategy spends. The target determines the total spend, which in turn affects the number of conversions and ultimately, the return on investment (ROI). The only aspect I can control is the duration of the experiment. That is, conditioned on the history of outcomes, do I continue for another day?

Our goal is to determine whether the adaptive strategy can outperform the fixed strategy by at least 10\% in terms of ROI.

Here are a few key questions I have:

1. What is the appropriate randomisation unit in this case? We can't be certain that the dollar ammounts or the number of conversions are at all comparible between experiment variations.
2. Given the uncertainty over the randomisation unit, is the simple total profit (revenue - spend) an appropriate metric to determine the outcome of the experiment? What is an appropriate method for statistical inference in this case.
3. Sample size & duration: How do I determine the appropriate sample size or experiment duration to detect a 10% uplift in ROI with statistical significance? Is it even correct to think of randomisation units being days?
4. Variance estimation: Given that spend is not directly controlled but influenced by target CPA, what metrics or techniques should I use to accurately estimate the variance in ROI across both groups?
5. Bias mitigation: How can I account for differences in spend or conversion patterns that might create bias, especially if one strategy adapts faster than the other?
6. Stopping rules: Since I can only control how long the experiment runs, what would be appropriate stopping criteria or statistical tests to apply during and after the experiment to ensure the results are reliable?

# Data

Google and Facebook collect data about each ad which is served to you. So, the data is at the level of the impression. I.e. each time somebody sees see an ad, that is recorded as a separate event. This means that a single person could see the same ad multiple times before clicking through (if they ever do so).

# Sequential Testing by Betting: A Martingale Approach

We are interested in testing which of two hypoteses are true:

1. The null hypothesis $H_0: P \in \mathcal{P}_{\text{null}}$.
2. The alternative $H_1: P \in \mathcal{P}_{\text{alt}}$.

We have observations of some procces denoted by $Z_1, Z_2, \ldots$ lying in some space $\mathcal{Z}$, and drawn i.i.d. according to $P$.

Testing by betting reframes hypothesis testing into a gambilng problem, instead of creating a function to produce test statistics. Totest the null $H_0$, a bettor may place repeated bets on the outcomes $\{Z_t : t \geq 1\}$ starting with an initial wealth $K_0 = 1$.

A single round of betting (say at time $t$) involves the following two steps:

Firstly, the bettor selects a payoff function $S_t: \mathcal{Z} \to [0, \infty)$. The payoff has to satisfy $\mathbb{E}_P[S_t(Z_t) \mid \mathcal{F}_{t-1}] = 1$. I.e. the bet is fair when the null is true. Then, the outcome $Z_t$ is revealed, and the bettor's wealth grows (or possibly shrinks) by a factor of $S_t(Z_t)$. Thus, the bettor's wealth after $t$ rounds of betting is $K_t = K_0 \prod_{i=1}^t S_i(Z_i)$.

The restriction on the conditional expectation of the payoff functions implies that under the null, $\{K_t : t \geq 0\}$ is a nonnegative martingale with an initial value 1. Due to this fact, $K_t$ is unlikely to take large values for any $t \geq 1$.

On the other hand, when $H_1$ is true, the bettor's choice of payoff functions, $\{S_t : t \geq 1\}$ should ensure that the wealth process grows exponentially. Such a wealth process naturally leads to the following sequential test: reject the null if $K_t \geq 1/\alpha$, where $\alpha \in (0, 1)$ is the desired confidence level. Ville's maximal inequality([^1]) ensures that this test controls the type-I error at level $\alpha$.

When testing simple hypotheses ($H_0: Z_t \sim P$ and $H_1: Z_t \sim Q$ with $P$ and $Q$ known), the payoff fucntion $S_t$ is just the likelihood ratio $Q/P$. With this choice of payoff functions, we have:

$$
\mathbb{E}_P\left[S_t \mid \mathcal{F}_{t-1}\right] = 1
$$

meaning it is a fair bet under the null. Under $H_1$, the wealth process with this payoff grows exponentially, with an optimal (expected) growth rate of $\text{KL}(Q, P)$: the KL-divergence between $Q$ and $P$.

When dealing with cases where either one or both of $H_0$ and $H_1$ are composite and nonparametric there is no obvious choice for the payoff functions. So there are a couple of design choices we need to make before we can use the testing by betting framework:

1. In which function class should $S_t$ lie?
2. How to ensure $\mathbb{E}\left[S_t \mid \mathcal{F}_{t-1}\right] = 1$ uniformly over $\mathcal{P}_{\text{null}}$?
3. How to ensure fast growth of $K_t$ under the alternative?

Before we continue with the exposition, we first define a sequential test:

> **Definition 1** (sequential-test). A level-$\alpha$ sequential test can be represented by a random stopping time $\tau$ taking values in ${1, 2, \ldots} \cup {\infty}$, and satisfying the condition $\mathbb{P}(\tau < \infty) \leq \alpha$, under the null $H_0$. Thus, $\tau$ denotes the random time at which the null hypothesis is rejected.

## A General Two Sample test

We begin by defining the two-sample testing problem.

**Definition:Two-sample testing.** Given a stream of paired observations $\{(X_t,Y_t) : t \geq 1\}$, drawn i.i.d. according to $P_X \times P_Y$ on the observation space $\mathcal{X} \times \mathcal{X}$, our goal is to test the null, $H_0 : P_X = P_Y$ against the alternative  $H_0 : P_X \neq P_Y$.

The distributions in the null class are invariant to the action of the operator $T : (\mathcal{X} \times \mathcal{X}) \to (\mathcal{X} \times \mathcal{X})$ that takes elements $(x,y) \in \mathcal{X} \times \mathcal{X}$ and flips their order; that is $T(x,y) = (y,x)$.

> **Remark** This definition assumes two streams of i.i.d. observations, $\{X_t : t \geq 1\}$ and $\{Y_t : t \geq 1\}$. However, {% cite shekhar2023nonparametric %} prove a theorem whose results are valid under a much weaker assumption that the stream $\{(X_t,Y_t) : t \geq 1\}$ consists of independent pairs of observations satisfying $(X_t,Y_t) \stackrel{d}{=} (Y_t, X_t)$ under the null, and $(X_t,Y_t) \stackrel{d}{\neq} (Y_t,X_t)$ under the alternative.

This means we only need:

* Each day's comparison to be a fair test under the null
* The betting strategy to only use past information

We don't need full independence between days, which makes this approach particularly suitable for our adaptive bidding scenario.

Now we need to do two things:

1. Construct a test function.
2. Choose a betting strategy.

### Constructing test functions

{cite % shekhar2023nonparametric %} begin by choosing a distance measure on the space of probability distributions which admits a variational representation. Specifically the function must be part of a class of functions $\mathbb{G}$ which maxmises the following distance:

$$
dG\left(P, Q\right) = sup_{g∈G} \midEP[g(X)] - EQ[g(Y)]\mid
$$

The function class $G$ should contain functions mapping to $\left[-1/2, 1/2\right]$. {cite % shekhar2023nonparametric %} choose a specific class of functions $G$ using the kernel maximum mean discrepancy (MMD) metric defined below:

**Definition (kernel MMD)** Let $\mathcal{X}$ denote the observation space, which for simplicity, we set to $\mathbb{R}^m$ for some $m \geq 1$, and let $K : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ be a positive definite kernel on $\mathcal{X}$. We assume that $K$ is uniformly bounded, that is, $\sup_{x,x'\in\mathcal{X}} K(x, x') \leq 1$, and let $\mathcal{H}_K$ denote the reproducing kernel Hilbert space (RKHS) associated with $K$. 

The associated IPM, called the kernel-MMD metric, is defined as follows:

$d_{MMD}(P, Q) = \sup_{\|g\|_K \leq 1} \mathbb{E}_P[g(X)] - \mathbb{E}_Q[g(Y)]$

where $\|g\|_K$ denotes the RKHS norm of the function $g$. The mean map of a distribution $P$ is a function in the RKHS given by $\mu_P := \mathbb{E}_P[K(X, \cdot)]$. When $P \neq Q$, the "witness" function $h^*$ that achieves the supremum in $d_{MMD}(P,Q)$ (i.e. witnesses the difference between $P,Q$) is simply given by $h^* := \mu_P - \mu_Q$, meaning that $d_{MMD}(P, Q) = \mathbb{E}_P[h^*(X)] - \mathbb{E}_Q[h^*(Y)]$.

and $g^* = \frac{1}{2}h^*$.

A predictor playing $\{g_t : t \geq 1\}$ is $\|\mu_P - \mu_Q\|_K$. 

The above discussion suggests the choice of $\mathcal{G} = \{g \in \mathcal{H}_K : \|g\| \leq 1/2\}$

$\tilde{g}(x,y) = g^*(x) - g^*(y) = \langle g^*, K(x,\cdot) - K(y,\cdot)\rangle$

Note that the scaling $h$ by $1/2$ in the definition of $g$ ensures that $\tilde{g}$ takes values in $[-1, 1]$.

This is implemented in the following code block (based upon the author's [own implementation](https://github.com/sshekhar17/nonparametric-testing-by-betting)):

```python
import numpy as np
from scipy.spatial.distance import cdist
from typing import Callable, Optional, Tuple

def compute_mmd(X: np.ndarray, 
                Y: np.ndarray, 
                kernel: Callable) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between X and Y samples.
    
    As described in paper Section 4: "Sequential Two-Sample Kernel-MMD Test",
    MMD is computed as:
    MMD^2(P,Q) = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
    where k is the kernel function.
    
    Args:
        X: First sample set 
        Y: Second sample set
        kernel: Kernel function k(x,y)
        
    Returns:
        float: MMD^2 value
    """
    # Compute kernel matrices
    Kxx = kernel(X)  # k(X,X)
    Kyy = kernel(Y)  # k(Y,Y)
    Kxy = kernel(X, Y)  # k(X,Y)
    
    # Compute expectations
    # Note: We exclude diagonal elements for Kxx and Kyy as they represent k(x,x)
    n = X.shape[0]
    m = Y.shape[0]
    
    xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n-1))  # E[k(X,X')] 
    yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m-1))  # E[k(Y,Y')]
    xy = Kxy.mean()  # E[k(X,Y)]
    
    # Compute MMD^2
    mmd2 = xx + yy - 2*xy
    
    return mmd2

def kernel_mmd_prediction(Xt: np.ndarray,
                         Yt: np.ndarray, 
                         kernel: Callable,
                         post_processing: str = 'clip') -> Tuple[float, dict]:
    """
    Compute MMD prediction for sequential testing using selected kernel.
    
    As discussed in paper Section 4, this implements the witness function
    associated with kernel-MMD for testing.
    
    Args:
        Xt: Current batch X samples
        Yt: Current batch Y samples
        kernel: Kernel function
        post_processing: How to constrain output ('clip' or 'tanh')
        
    Returns:
        tuple: (prediction, metadata)
    """
    # Compute centered kernel matrices
    Kxx = kernel(Xt)
    Kyy = kernel(Yt)
    Kxy = kernel(Xt, Yt)
    
    n = len(Xt)
    m = len(Yt)
    
    # Compute witness function values
    wx = (Kxx.sum(axis=1) - np.diag(Kxx)) / (n-1) 
    wy = Kxy.sum(axis=1) / m
    witness_x = (wx - wy) / n
    
    wy = (Kyy.sum(axis=1) - np.diag(Kyy)) / (m-1)
    wx = Kxy.sum(axis=0) / n  
    witness_y = (wy - wx) / m
    
    # Normalize predictions to [-1/2, 1/2] range as required by betting
    if post_processing == 'clip':
        prediction = np.clip(witness_x.mean() - witness_y.mean(), -0.5, 0.5)
    elif post_processing == 'tanh':
        # Use tanh for unbounded kernels as described in Section 2.3
        prediction = np.tanh(witness_x.mean() - witness_y.mean()) / 2
    else:
        raise ValueError(f"Unknown post_processing: {post_processing}")
    
    metadata = {
        'witness_x': witness_x,
        'witness_y': witness_y,
        'mmd2': compute_mmd(Xt, Yt, kernel)
    }
    
    return prediction, metadata

def get_optimal_kernel_bandwidth(X: np.ndarray, 
                               Y: np.ndarray) -> float:
    """
    Get optimal bandwidth for RBF kernel using median heuristic.
    
    As referenced in the paper's experiments section.
    
    Args:
        X: First sample set
        Y: Second sample set
        
    Returns:
        float: Optimal bandwidth
    """
    # Combine samples
    Z = np.vstack([X, Y])
    
    # Compute pairwise distances
    dists = cdist(Z, Z, 'euclidean')
    
    # Use median heuristic
    bandwidth = np.median(dists[np.triu_indices_from(dists, k=1)])
    
    return bandwidth
```

For our problem, we are interested in calculating the relative ROI of two bidding strategies on Google Smart Bidding. So we can use the following code:

```python
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ROIMetrics:
    revenue: float
    spend: float
    
    def roi(self) -> float:
        return self.revenue / self.spend if self.spend > 0 else 0.0

class ROIKernelMMDTest:
    """
    Kernel MMD test for ROI comparisons using betting martingales.
    Combines ROI witness function with kernel MMD framework from paper.
    """
    def __init__(self, 
                kernel_bandwidth: float = 1.0,
                alpha: float = 0.05,
                min_effect: float = 0.10):
        self.kernel_bandwidth = kernel_bandwidth
        self.alpha = alpha
        self.min_effect = min_effect
        self.wealth = 1.0
        self.wealth_history = [1.0]

    def compute_kernel(self, x: ROIMetrics, y: ROIMetrics) -> float:
        """
        Compute kernel between two ROI values using RBF kernel.
        """
        roi_diff = x.roi() - y.roi()
        return np.exp(-(roi_diff**2) / (2 * self.kernel_bandwidth**2))

    def compute_mmd_squared(self, 
                          adaptive_samples: List[ROIMetrics],
                          fixed_samples: List[ROIMetrics]) -> float:
        """
        Compute squared MMD between adaptive and fixed strategy samples.
        
        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)] 
        where k is the kernel function
        """
        n = len(adaptive_samples)
        m = len(fixed_samples)

        # Compute kernel matrices
        K_aa = np.zeros((n, n))
        K_ff = np.zeros((m, m))
        K_af = np.zeros((n, m))

        for i in range(n):
            for j in range(n):
                K_aa[i,j] = self.compute_kernel(adaptive_samples[i], adaptive_samples[j])
            for j in range(m):
                K_af[i,j] = self.compute_kernel(adaptive_samples[i], fixed_samples[j])

        for i in range(m):
            for j in range(m):
                K_ff[i,j] = self.compute_kernel(fixed_samples[i], fixed_samples[j])

        # Compute expectations (excluding diagonal elements where appropriate)
        E_aa = (K_aa.sum() - np.trace(K_aa)) / (n * (n-1))
        E_ff = (K_ff.sum() - np.trace(K_ff)) / (m * (m-1))
        E_af = K_af.mean()

        mmd_squared = E_aa + E_ff - 2*E_af
        return mmd_squared

    def compute_witness_value(self,
                            adaptive_metrics: ROIMetrics,
                            fixed_metrics: ROIMetrics,
                            adaptive_history: List[ROIMetrics],
                            fixed_history: List[ROIMetrics]) -> float:
        """
        Compute witness function value for sequential testing.
        Maps to [-1/2, 1/2] range as required by betting strategy.
        """
        # Compute ROI difference
        roi_diff = adaptive_metrics.roi() - fixed_metrics.roi()
        
        # Weight by kernel-based similarity to historical data
        kernel_weight = 0.0
        n_hist = len(adaptive_history)
        if n_hist > 0:
            for hist_a, hist_f in zip(adaptive_history, fixed_history):
                k_a = self.compute_kernel(adaptive_metrics, hist_a)
                k_f = self.compute_kernel(fixed_metrics, hist_f)
                kernel_weight += (k_a - k_f) / n_hist

        # Combine direct ROI difference with kernel-weighted historical influence
        witness = roi_diff + kernel_weight
        
        # Map to [-1/2, 1/2] using tanh
        return 0.5 * np.tanh(witness)

    def update(self,
              adaptive_metrics: ROIMetrics,
              fixed_metrics: ROIMetrics,
              adaptive_history: List[ROIMetrics],
              fixed_history: List[ROIMetrics]) -> Tuple[bool, str]:
        """
        Update test statistics and check stopping condition.
        
        Returns:
            Tuple of (should_stop, decision)
        """
        # Compute witness value
        witness = self.compute_witness_value(
            adaptive_metrics,
            fixed_metrics, 
            adaptive_history,
            fixed_history
        )
        
        # Compute optimal bet size (Kelly fraction)
        relative_improvement = (adaptive_metrics.roi() - fixed_metrics.roi()) / abs(fixed_metrics.roi()) if fixed_metrics.roi() != 0 else 0
        bet = (relative_improvement - self.min_effect) / 2
        bet = max(-0.5, min(0.5, bet))
        
        # Update wealth
        self.wealth *= (1 + bet * witness)
        self.wealth_history.append(self.wealth)

        # Check stopping conditions
        if self.wealth >= 1.0 / self.alpha:
            return True, "Reject null - Adaptive strategy shows significant improvement"
        elif self.wealth <= self.alpha:
            return True, "Accept null - Insufficient evidence for improvement"
        
        return False, "Continue testing"

def run_roi_mmd_experiment(
    adaptive_metrics: List[ROIMetrics],
    fixed_metrics: List[ROIMetrics],
    kernel_bandwidth: float = 1.0,
    alpha: float = 0.05,
    min_effect: float = 0.10
) -> Dict:
    """
    Run complete ROI MMD experiment.
    
    Args:
        adaptive_metrics: List of metrics from adaptive strategy
        fixed_metrics: List of metrics from fixed strategy
        kernel_bandwidth: Kernel sensitivity parameter
        alpha: Type I error rate
        min_effect: Minimum ROI improvement to detect
        
    Returns:
        Dict containing test results and metrics
    """
    test = ROIKernelMMDTest(
        kernel_bandwidth=kernel_bandwidth,
        alpha=alpha,
        min_effect=min_effect
    )
    
    n_days = len(adaptive_metrics)
    stopped = False
    stop_day = n_days
    decision = "Inconclusive"
    
    for day in range(n_days):
        # Get current day metrics
        adaptive = adaptive_metrics[day]
        fixed = fixed_metrics[day]
        
        # Get history up to current day
        adaptive_history = adaptive_metrics[:day]
        fixed_history = fixed_metrics[:day]
        
        # Update test
        should_stop, result = test.update(
            adaptive,
            fixed,
            adaptive_history,
            fixed_history
        )
        
        if should_stop:
            stopped = True
            stop_day = day + 1
            decision = result
            break
            
    # Compute final MMD
    final_mmd = test.compute_mmd_squared(adaptive_metrics, fixed_metrics)
    
    return {
        'stopped': stopped,
        'stop_day': stop_day,
        'decision': decision,
        'final_mmd': final_mmd,
        'wealth_history': test.wealth_history
    }
```

### Betting strategy

Having selected $\mathcal{G}$, the final step in instantiating the sequential test is choosing an appropriate prediction strategy. The regret of the prediction game after $n$ observations is:

$R_n \equiv R_n(A_{pred}, \mathcal{G}, X_1^n, Y_1^n) = \max_{g \in \mathcal{G}} \sum_{t=1}^n \langle g - g_t, K(X_t,\cdot) - K(Y_t,\cdot)\rangle.$

A natural choice is the plug-in or the empirical risk minimization (ERM) strategy, that simply selects $g_t = \arg\max_{g\in\mathcal{G}}\langle g,\mu_{\hat{P}_{X,t-1}} - \mu_{\hat{P}_{Y,t-1}}\rangle$. We can check that this choice results in a consistent sequential test. 

To get the exponent and bound on the expected stopping time under the alternative, however, we need to use an adaptive version of the online gradient ascent (OGA) strategy, that proceeds as follows, with $M_t := \sum_{i=1}^t \|g_i(X_i, \cdot) - g_i(Y_i, \cdot)\|^2_K$:

$g_1 = 0, \text{ and } g_{t+1} = \Pi_{\mathcal{G}}\left(g_t + \frac{1}{2\sqrt{M_t}}(K(X_t,\cdot) - K(Y_t,\cdot))\right) \text{ for } t \geq 1. \tag{17}$

Recall that $\Pi_{\mathcal{G}}$ denotes the projection operator (in terms of the RKHS norm $\|\cdot\|_K$) onto the function class $\mathcal{G}$, which acts as follows: $\Pi_{\mathcal{G}}(h) = \frac{h}{2\|h\|_K}$.

**Definition: Sequential Kernel MMD Test:**

Set $K_0 = 1$, $\lambda_1 = 0$, and $g_1 = 0 \in \mathcal{H}_K$. For $t = 1,2,...$:

- Observe $X_t$, $Y_t$
- Update the wealth: $K_t = K_{t-1} \times (1 + \lambda_t\langle g_t, K(X_t, \cdot) - K(Y_t, \cdot)\rangle)$
- Reject the null if $K_t \geq 1/\alpha$
- Update $g_{t+1}$ using the OGA prediction strategy described above
- Update $\lambda_{t+1}$ as follows: 

$\lambda_{t+1} = \min\{1,\max\{-1,\lambda_t - \frac{2z_t}{2-\log_3a_t^2}\}\}$

The full MMD Test is in the following code:

```python
import numpy as np
from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass

class SequentialKernelMMDTest:
    def __init__(self, 
                 kernel_func: Callable,
                 alpha: float = 0.05):
        """
        Sequential Kernel MMD Test as defined in paper Definition 8.
        
        Args:
            kernel_func: Kernel function k(x,y)
            alpha: Type I error rate
        """
        self.kernel = kernel_func
        self.alpha = alpha
        
        # Initialize test statistics
        self.K0 = 1.0
        self.Kt = 1.0  # Current wealth
        self.wealth_history = [1.0]
        
        # Initialize betting strategy
        self.lambda1 = 0.0
        self.lambda_t = 0.0
        self.a0 = 1.0  # For ONS strategy
        
        # Initialize prediction strategy
        self.g1 = None
        self.gt = None
        self.Mt = 0.0  # Sum of squared RKHS norms
        
    def _ons_betting_update(self, vt: float) -> float:
        """
        Online Newton Step betting strategy (Definition 5 in paper).
        """
        # Update at based on zt
        zt = vt / (1 - vt * self.lambda_t)
        self.a0 += zt * zt
        
        # Update lambda using ONS update rule
        lambda_next = self.lambda_t - (2 - np.log(3)) * zt / self.a0
        
        # Project to [-1/2, 1/2]
        lambda_next = np.clip(lambda_next, -0.5, 0.5)
        
        return lambda_next
    
    def _oga_prediction_update(self, 
                             Xt: np.ndarray, 
                             Yt: np.ndarray, 
                             kt_x: np.ndarray,
                             kt_y: np.ndarray) -> np.ndarray:
        """
        Online Gradient Ascent prediction strategy (Equation 17 in paper).
        """
        if self.gt is None:
            self.gt = np.zeros_like(kt_x)
            return self.gt
            
        # Compute RKHS norm of difference
        diff = kt_x - kt_y
        self.Mt += np.sum(diff * diff)
        
        # Update using OGA
        grad = diff / (2 * np.sqrt(self.Mt))
        gt_new = self.gt + grad
        
        # Project onto G using RKHS norm
        norm = np.sqrt(np.sum(gt_new * gt_new))
        if norm > 0:
            gt_new = gt_new / (2 * norm)  # ΠG operator
            
        return gt_new

    def update(self, 
               Xt: np.ndarray, 
               Yt: np.ndarray) -> Tuple[bool, str]:
        """
        Update test statistics for new observations.
        
        Args:
            Xt: New X observation
            Yt: New Y observation
            
        Returns:
            (should_stop, decision)
        """
        # Compute kernel evaluations
        kt_x = self.kernel(Xt)
        kt_y = self.kernel(Yt)
        
        # Update prediction gt+1 using OGA
        self.gt = self._oga_prediction_update(Xt, Yt, kt_x, kt_y)
        
        # Compute test statistic
        vt = np.dot(self.gt, (kt_x - kt_y))
        
        # Update wealth
        self.Kt *= (1 + self.lambda_t * vt)
        self.wealth_history.append(self.Kt)
        
        # Update betting fraction λt+1 using ONS
        self.lambda_t = self._ons_betting_update(vt)
        
        # Check stopping condition
        if self.Kt >= 1.0/self.alpha:
            return True, "Reject null hypothesis - Distributions differ significantly"
        
        return False, "Continue testing"
        
    def get_results(self) -> Dict:
        """Get test results."""
        return {
            'stopped': self.Kt >= 1.0/self.alpha,
            'wealth': self.Kt,
            'wealth_history': self.wealth_history,
            'lambda_final': self.lambda_t
        }

def run_sequential_mmd_test(X_stream: np.ndarray,
                           Y_stream: np.ndarray,
                           kernel_func: Callable,
                           alpha: float = 0.05,
                           max_samples: Optional[int] = None) -> Dict:
    """
    Run sequential kernel MMD test on two sample streams.
    
    Args:
        X_stream: First sample stream 
        Y_stream: Second sample stream
        kernel_func: Kernel function
        alpha: Type I error rate
        max_samples: Maximum samples to test (optional)
        
    Returns:
        Dict containing test results
    """
    # Initialize test
    test = SequentialKernelMMDTest(kernel_func, alpha)
    
    # Set max samples
    if max_samples is None:
        max_samples = len(X_stream)
    n = min(len(X_stream), len(Y_stream), max_samples)
    
    # Run sequential test
    for t in range(n):
        Xt = X_stream[t]
        Yt = Y_stream[t]
        
        should_stop, decision = test.update(Xt, Yt)
        if should_stop:
            break
            
    # Get results
    results = test.get_results()
    results.update({
        'samples_used': t + 1,
        'decision': decision
    })
    
    return results
```