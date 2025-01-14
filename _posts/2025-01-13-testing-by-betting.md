---
layout: default
title:  Hypothesis Testing by Betting
subtitle: A simpler alternative to $p$-values.
date:   2025-01-13
categories: hypothesis-testing
---

# Introduction

The $p$-value is the most misunderstood, misused, and maligned scientific quantity. The $p$-value is merely a summary of your data; it tells you how likely you are to see data at least as extreme as the data you have collected when your null hypothesis is correct. However it often does double duty in statistical analyses: it can tell us whether or not to reject the null hypothesis (e.g. if $ p \leq \alpha$) as well as how confident we should be in the alternative hypothesis. Altogether, this makes the $p$-value a poor tool for scientific communication.

A simpler alternative to $p$-values is to report the result of a bet against the null hypothesis. This is **testing by betting.** This method is closely related to the likelihood method, and leads to alternatives for statistical confidence and power.

This post is a longer exposition of _Testing by Betting_, Gleen Schafer (2021) {% cite shafer2021testing %}. The point was to include more exposition and some code. Any resemblence to the original is therefore natural (and in some cases I've jsut changed the language);

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

# Testing by betting

We are interested in phenomenon which we model with a probability distribution, $P$, with an associated set of random variables. $X$. For example:

1. The ammount of traffic per hour at an intersection. We could model this as a [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) distribution, and we want to know the mean parameter.
2. The height of pupils in a school. We could model this as a normal distribution, and we would like know the mean and standard deviation.

We will later see the actual value of the parameter $X$, which we denote $x$. How can we give more nuanced content to our claims (e.g. that traffic intensity is best modelled by a Poisson distribution), and how could we challenge them?

The way we choose to proceed is to interpret our claim as a collection of betting offers with Reality[^1]. Reality offers to sell me any payoff $S\left(x\right)$ for its expected value $\mathcal{E}_P\left(x\right)$. I can choose a (nonnegative) payoff $S$, so that $\mathcal{E}_P\left(x\right)$ is all that I risk.

My betting score is:

$$
\frac{S\left(x\right)}{\mathcal{E}_P\left(x\right)}
$$

The score doesn't change when I scale my bet, so I can take $\mathcal{E}_P\left(x\right) = 1$, and so the betting score is simply my payoff.

The standard way of testing a probability distribution $P$ is to select a significance level $α \in \left(0,1\right)$, usually small, and a set $E$ of possible values of $X$ such that $\mathcal{P}\left(X \in E\right) = \alpha$. The event $E$ is called the rejection region. The probability distribution $P$ is discredited (or rejected) if the actual value $x$ is in $E$.

For example, we could test the mean $X$ of a series of measurements $M_1, M_2, \ldots, M_k$ to see if the distribution $P$ has mean zero or not. We believe that $X$ comes from a standard normal distribution, and so we test (with a $z$-test for instance) that $X$ is in the region $X \in \left(-1.96, 1.96\right)$ (i.e. $X$ is within 2 standard deviations of the standard normal distribution). If we see a $x$ (the actual mean) that is bigger than 1.96 (i.e. $\mid x\mid > 1.96$) then we conclude that the $X$ does not have mean 0.

Textbooks seldom make the idea explicit, however a standard test is often thought of as a bet: I pay one dollar for the payoff $S_E$ defined by

$$
S_E = \begin{cases}
\frac{1}{\alpha} & \text{if } x \in E \\
0 & \text{otherwise}
\end{cases}
$$

If $E$ happens, I have multiplied the dollar I risked by $\frac{1}{\alpha}$. This makes standard testing a special case of testing by betting, where the bet is all-or-nothing. In return for a dollar, I get either $\frac{1}{\alpha}$ or nothing.

So a betting score $S\left(x\right)$ appraises the evidence against $P$. The larger $S\left(x\right)$, the stronger the evidence.

## Betting scores are likelihood ratios

Betting against a hypothesis is equivalent to proposing an alternative hypothesis $Q$ and comparing likelihoods. We show this this by first showing a betting score is a likelihood ratio, and then by showing that likelihood ratios are betting scores.

Earlier we defined our betting score as:

$$
\frac{S\left(x\right)}{\mathcal{E}_P\left(x\right)}
$$

Where our payoff in the bet is denoted by $S\left(x\right)$ and my stake is simplly the expected value $\mathcal{E}_P\left(x\right)$. The betting score is what you win divided by what you paid. I can choose a (nonnegative) payoff $S$, so that $\mathcal{E}_P\left(x\right)$ is all that I risk (i.e. I pay the expected value of this payoff $S$ under probability distribution $P$).
 
We also noted that the score doesn't change when I scale my bet (doubling both your payoff and stake gives the same score), so I can take $\mathcal{E}_P\left(y\right) = 1$. The assumption $\mathbb{E}_P(S) = 1$. Another way of saying this is that $\sum_x S(x)P(x) = 1$. Because $S(x)$ and $P(x)$ are:

1. nonnegative for all $x$ (and we are assuming that $P > 0$)
2. $\sum_x S(x)P(x) = 1$

$SP$ is a probability distribution. We _define_

$$
Q := SP
$$ 

and call $Q$ the alternative *implied* by the bet $S$. If $P$ is wrong and $Q$ is actually the true distribution, then your bet would be expected to make money. So $Q$ represents what you "think might actually be true" when you make your bet against $P$.

Because we assumed $P(x) > 0$ for all $x$, we can rearrange, and express our payoff as the ratio of these two probability distributions:

$$
S(x) = \frac{Q(x)}{P(x)}
$$

i.e. the betting score is a simply a likelihood ratio.

A likelihood ratio is a betting score.

If you have two probability distributions $Q$ and $P$ then the likelihood ration $S = Q/P$ satisfies the requirements for a bet:

1. $Q/P$ is nonnegative since probabilities are nonnegative.
2. $\sum_x \frac{Q(x)}{P(x)}P(x) = \sum_y Q(x) = 1$. I.e. The expected value of the bet under $P$ is 1.

expected value of this bet under $Q$ is nonnegative: $\mathbb{E}_Q(S) \geq 1$ ($\mathbb{E}_Q(S)$ is how much you expect to win if $Q$ is actually true).

In fact $\mathbb{E}_Q(S) = \mathbb{E}_P(S^2)$. Since $S = Q/P$, we can write $\mathbb{E}_Q(S) = \sum_x S(x)Q(x)$. We substitute $S = Q/P$:

$$
\begin{align*}
\mathbb{E}_Q(S) &= \sum_x \frac{Q(x)}{P(x)}Q(x) \\
&= \sum_x \frac{Q(x)^2}{P(x)}
\end{align*}
$$

Let's consider $\mathbb{E}_P(S^2)$:

$$
\mathbb{E}_P(S^2) = \sum_y S(x)^2P(x)
$$

Again substitute $S = Q/P$:

$$
\begin{align*}
\mathbb{E}_P(S^2) &= \sum_x \left(\frac{Q(x)}{P(x)}\right)^2P(x) \\
&= \sum_x \frac{Q(x)^2}{P(x)^2}P(x) \\
&= \sum_x \frac{Q(x)^2}{P(x)}
\end{align*}
$$

We can see that this is the same as what we got for $\mathbb{E}_Q(S)$, thus $\mathbb{E}_Q(S) = \mathbb{E}_P(S^2)$. When we square $S = Q/P$ and multiply by $P$ in $\mathbb{E}_P(S^2)$, we end up with the same expression as when we multiply $S = Q/P$ by $Q$ in $\mathbb{E}_Q(S)$.

A consequence of this insight is that because we know that $\mathbb{E}_P(S) = 1$ (this was one of our initial assumptions about the betting score), $\mathbb{E}_Q(S) = \mathbb{E}_P(S^2)$ implies:

$$
\begin{align*}
\mathbb{E}_Q(S) - 1
&= \mathbb{E}_P(S^2) - 1 \quad \text{(substituting }\mathbb{E}_Q(S) = \mathbb{E}_P(S^2)\text{)} \\
&= \mathbb{E}_P(S^2) - (\mathbb{E}_P(S))^2 \quad \text{(substituting }1 = \mathbb{E}_P(S)\text{)} \\
&= \text{Var}_P(S)
\end{align*}
$$

Here $\mathbb{E}_Q(S) - 1$ represents what I win if I am right, and not "Reality" ($P$). i.e the more variable your betting score is under the null hypothesis $P$, the more you stand to gain if your alternative $Q$ is actually true. If you make a more extreme bet (higher variance), you have more to gain if you're right.

## How much should I bet?

We began with your claiming that $P$ describes the phenomenon $X$ and my making a bet $S$ satisfying $S \geq 0$ and, for simplicity, $\mathbb{E}_P(S) = 1$. Suppose, however, that I do have an alternative $Q$ in mind. I have a hunch that $Q$ is a valid description of $X$. In this case, should I use $\frac{Q}/{P}$ as my bet?

There are two schools of thought: opportunistic and repeated testing, and testing in a single 'go.' The first justification is related to [Kelly Betting](https://en.wikipedia.org/wiki/Kelly_criterion), and the second is related to the [Neyman-Pearson Lemma](https://en.wikipedia.org/wiki/Neyman%E2%80%93Pearson_lemma).

### Kelly Betting
The thought that I should is supported by Gibbs's inequality which says that[^2]:

$$
\mathbb{E}_Q \ln\frac{Q}{P} \geq \mathbb{E}_Q \ln\frac{R}{P}
$$

for **any** probability distribution $R$ for $Y$. Because any bet $S$ is of the form $R/P$ for some such $R$, Gibb's inequality tells us that $\mathbb{E}_Q(\ln S)$ is maximized over $S$ by setting $S := Q/P$. This tells that if we have a hunch that $Q$ is more correct than $P$ (or any other probability distribution $R$) then we should be $\frac{Q}{P}$.

The quanitiy $\mathbb{E}_Q(\ln(Q/P))$ is known as the Kullback-Leibler divergence between $Q$ and $P$. It is the mean number of bits of information you gain when $Q$ is the true encoding distribution and not $P$.

Why should I choose $S$ to maximize $\mathbb{E}_Q(\ln S)$? Why not maximize $\mathbb{E}_Q(S)$? Or perhaps $Q(S \geq 20)$ or $Q(S \geq 1/\alpha)$ for some other significance level $\alpha$?

Maximizing $\mathbb{E}(\ln S)$ makes sense in a scientific context where we combine successive betting scores by multiplication. i.e. when we are testing opportunistically. When $S$ is the product of many successive factors, maximizing $\mathbb{E}(\ln S)$ maximizes $S$'s rate of growth. Logarithms are _additive_ in repeated bets.

### Neyman-Pearson
Conversely choosing $S$ to maximize $Q(S \geq 1/\alpha)$ is appropriate when the hypothesis being tested will not be tested again.

For a given significance level $\alpha$, we choose a rejection region $E$ such that $Q(x)/P(x)$ is at least as large for all $x \in E$ as for any $x \notin E$, where $Q$ is an alternative hypothesis.

Let us call the bet $S_E$ with this choice of $E$ the *level-$\alpha$ bet* against $P$ with respect to $Q$. The *Neyman-Pearson lemma* says that this choice of $E$ maximizes

$$Q(\text{test rejects }P) = Q(X \in E) = Q(S_E(X) \geq 1/\alpha)$$

which we call the *power* of the test with respect to $Q$. In fact, $S_E$ with this choice of $E$ maximizes

$$Q(S(Y) \geq 1/\alpha)$$

over all bets $S$, not merely over all-or-nothing bets. {% cite shafer2021testing %} offers a proof of this statement.

## Implied targets

When I get my betting score for a particular bet $S$ against $P$, how do I know if the score is good or not?

Choosing a payoff $S$ defines an alternative probability distribution, $Q : = SP$, and with $S$ being the bet against $P$ that maximizes $\mathbb{E}_Q(\ln S)$. We might hope for a betting score whose _logarithm_ is in the ballpark of $\mathbb{E}_Q(\ln S)$. I.e a betting score like:

$$
S_{*} : = \exp{\mathbb{E}_Q(\ln S)}
$$

We call $S_{∗}$ the **implied target** of the bet $S$. The implied target of the all-or-nothing bet is always $\frac{1}{\alpha}$.

The notion of an implied target is analogous to the notion of statistical power with respect to a particular alternative. But it has the advantage that we cannot avoid discussing it by refusing to specify a particular alternative. The implied alternative $Q$ and the implied target $S_{∗}$ are determined as soon as the distribution $P$ and the bet $S$ are specified. The implied target can be computed without even mentioning $Q$, because:

$$\mathbb{E}_Q(\ln S) = \sum_x Q(x)\ln S(x) = \sum_y P(x)S(x)\ln S(x) = \mathbb{E}_P(S\ln S)$$

(def expectation, then def of $Q$, collect like terms). i.e. this is expected log betting score under _either_ $Q$ (as $\mathbb{E}_Q(\ln S)$) _or_ $P$ (as $\mathbb{E}_P(S\ln S)$).

# Examples

**Example 1.** 

Suppose $P$ says that $X$ is normal with mean 0 and standard deviation 10, $Q$ says that $X$ is normal with mean 1 and standard deviation 10, and we observe $x = 30$.

1. Statistician A simply calculates a p-value: $P(X \geq 30) \approx 0.00135$. She concludes that $P$ is strongly discredited.

2. Statistician B uses the Neyman-Pearson test with significance level $\alpha = 0.05$, which rejects $P$ when $x > 16.5$. Its power is only about 6%[^3].

Seeing $x = 30$, it does reject $P$. Had she formulated her test as a bet, she would have multiplied the money she risked by 20.

3. Statistician C uses the bet $S$ given by:

$$
S(y) := \frac{q(x)}{p(x)} = \frac{\sqrt{10^2\pi}\exp(-(x-1)^2/200)}{\sqrt{10^2\pi}\exp(-x^2/200)} = \exp(\frac{2x-1}{200})
$$

So

$$
\mathbb{E}_Q(\ln(S)) = \frac{1}{200} = \frac{1}{200}
$$

I.e. the implied target is $\exp(1/200) \approx 1.005$. She does a little better than this very low target; she multiplies the money she risked by $\exp(59/200) \approx 1.34$.

The power and the implied target both told us in advance that the study was a waste of time. The betting score of 1.34 confirms that little was accomplished, while the low $p$-value and the Neyman-Pearson rejection of $P$ give a misleading verdict in favour of $Q$.

# Hypotheis Testing and Code

We now present three implementations of the ideas in {% cite shafer2021testing %}. Firstly we implement a version of Protocol 1, which formalises the above method of testing a probability distribution $P$ for a phenomenon X that takes values in a set $\mathcal{X}$. 

>Protocol 1. Testing a probability distribution
>    Sceptic announces $S: X → \left[0, \infty\right)$ such that $\mathbb{E}_P(S) = 1$. 
> Reality announces $X \in \mathcal{X}$.
> \mathcal{K}: = $S(x)$.

Where $\mathcal{K}$ is our final capital.

The ```BettingTest``` class below implements this Protocol.

```python
class BettingTest:
    def __init__(self, null_mean=0, null_std=1):
        """
        Initialize a betting-based hypothesis test.
        
        Parameters:
        - null_mean: Mean under the null hypothesis
        - null_std: Standard deviation under the null hypothesis
        """
        self.null_mean = null_mean
        self.null_std = null_std
        
    def compute_betting_score(self, data, alternative_mean):
        """
        Compute the betting score (likelihood ratio) comparing the null hypothesis
        to an alternative hypothesis.
        
        Parameters:
        - data: Observed data
        - alternative_mean: Mean under the alternative hypothesis
        
        Returns:
        - betting_score: The factor by which we multiply our money
        - implied_target: The expected betting score under the alternative
        """
        null_density = stats.norm.pdf(data, self.null_mean, self.null_std)
        alt_density = stats.norm.pdf(data, alternative_mean, self.null_std)
        
        betting_score = alt_density / null_density
        
        log_scores = np.log(betting_score)
        implied_target = np.exp(np.mean(log_scores))
        
        return betting_score, implied_target
```

We can use the code like this:

```python
if __name__ == "__main__":
    np.random.seed(42)
    
    true_mean = 0.5
    null_mean = 0
    null_std = 1
    n_samples = 100
    data = np.random.normal(true_mean, 1, n_samples)
    
    bt = BettingTest(null_mean=0=null_mean, null_std=null_std)
    betting_scores, implied_target = bt.compute_betting_score(data, alternative_mean=0.5)
    final_betting_score = np.exp(np.mean(np.log(betting_scores)))
    t_stat, p_value = stats.ttest_1samp(data, null_mean)
    is_significant = p_value < alpha

    print("\nResults:")
    print(f"Sample mean: {np.mean(data):.3f}")
    print(f"\nTraditional Test:")
    print(f"P-value: {p_value:.3f}")
    print(f"Significant at 0.05 level: {is_significant}")
    print(f"\nBetting Test:")
    print(f"Final betting score: {final_betting_score:.3f}")
    print(f"Implied target: {implied_target:.3f}")

    score_interp = "strong" if final_betting_score > 20 else \
                   "moderate" if final_betting_score > 5 else "weak"
    print(f"\nBetting score indicates {score_interp} evidence against null hypothesis")
    print(f"(We multiplied our money by a factor of {final_betting_score:.1f})")
```

Here is protocol 2:

```python
import numpy as np
from scipy import stats

class Protocol2Test:
    def __init__(self):
        """Initialize with K0 = 1 as specified in Protocol 2"""
        self.K = 1.0  # Current capital
        self.history = []  # Track history of outcomes
        
    def announce_bet(self, y_history):
        """
        Sceptic announces betting function Sn.
        
        In Protocol 2, this should satisfy EP(Sn(Yn)|y1,...,yn-1) = Kn-1
        We'll create a simple betting function based on the likelihood ratio
        between two normal distributions.
        """
        current_capital = self.K
        
        def S(y):
            null_density = stats.norm.pdf(y, loc=0, scale=1)
            if len(y_history) > 0:
                alt_mean = np.mean(y_history)
            else:
                alt_mean = 0.1
                
            alt_density = stats.norm.pdf(y, loc=alt_mean, scale=1)
            return current_capital * (alt_density / null_density)
        
        return S
    
    def update(self, yn):
        """
        Reality announces yn and capital is updated.
        
        Parameters:
        - yn: The realized value
        """
        S = self.announce_bet(self.history)
        self.K = S(yn)
        self.history.append(yn)
        return self.K
    
    def verify_martingale_property(self, n_samples=1000):
        """
        Verify that EP(Sn(Yn)|y1,...,yn-1) = Kn-1
        by Monte Carlo simulation.
        """
        S = self.announce_bet(self.history)
        samples = np.random.normal(0, 1, n_samples)
        expected_value = np.mean([S(y) for y in samples])
        
        print(f"Martingale property check:")
        print(f"Current capital (Kn-1): {self.K:.3f}")
        print(f"Expected value of Sn: {expected_value:.3f}")

if __name__ == "__main__":
    np.random.seed(42)
    data = np.random.normal(0.5, 1, 10)  # Data with non-zero mean
    test = Protocol2Test()
    print("Testing sequence:")
    
    for i, yn in enumerate(data):
        if i < 3:
            print(f"\nStep {i+1}:")
            test.verify_martingale_property()
        capital = test.update(yn)
        print(f"yn = {yn:.3f}, Capital = {capital:.3f}")
```

Here is a betting strategy based on differences between two stocastic processes:

```python
import numpy as np
from scipy import stats
from typing import Tuple, List, Optional

class SequentialBettingTest:
    def __init__(self, initial_capital: float = 1.0):
        """
        Initialize a sequential betting test for comparing two processes.
        
        Parameters:
        - initial_capital: Starting betting capital (default=1.0)
        """
        self.capital = initial_capital
        self.betting_scores = []
        self.cumulative_scores = []
        
    def compute_step_bet(self, 
                        x1_history: np.ndarray, 
                        x2_history: np.ndarray,
                        current_difference: float,
                        null_hypothesis_difference: float = 0.0,
                        volatility: float = 1.0) -> float:
        """
        Compute the optimal bet for the next step based on the history.
        Uses Kelly criterion for optimal bet sizing.
        
        Parameters:
        - x1_history: History of first process
        - x2_history: History of second process
        - current_difference: Current observation of difference
        - null_hypothesis_difference: Expected difference under null hypothesis
        - volatility: Expected volatility of the difference
        
        Returns:
        - bet_size: Amount to bet (as fraction of current capital)
        """
        historical_diffs = np.diff(x1_history - x2_history)
        if len(historical_diffs) > 0:
            hist_mean = np.mean(historical_diffs)
            hist_std = np.std(historical_diffs) if len(historical_diffs) > 1 else volatility
        else:
            hist_mean = 0
            hist_std = volatility
            
        advantage = (hist_mean - null_hypothesis_difference) / (hist_std ** 2)
        kelly_fraction = advantage / 2  # Classic Kelly for Gaussian
        return np.clip(kelly_fraction, -0.5, 0.5)
    
    def update(self, 
               x1: float, 
               x2: float, 
               null_hypothesis_difference: float = 0.0,
               x1_history: Optional[np.ndarray] = None,
               x2_history: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Update betting score based on new observations.
        
        Parameters:
        - x1: New observation from first process
        - x2: New observation from second process
        - null_hypothesis_difference: Expected difference under null hypothesis
        - x1_history: Optional history of first process
        - x2_history: Optional history of second process
        
        Returns:
        - current_capital: Updated capital after bet
        - betting_score: Score for this step
        """
        difference = x1 - x2
        
        if x1_history is None:
            x1_history = np.array([])
        if x2_history is None:
            x2_history = np.array([])
            
        bet_size = self.compute_step_bet(
            x1_history, x2_history, difference,
            null_hypothesis_difference
        )
        
        score = 1 + bet_size * (difference - null_hypothesis_difference)
        self.capital *= score
        
        self.betting_scores.append(score)
        self.cumulative_scores.append(self.capital)
        
        return self.capital, score
    
    def get_results(self) -> dict:
        """
        Get summary of betting test results.
        
        Returns:
        - Dictionary containing test results and statistics
        """
        return {
            'final_capital': self.capital,
            'betting_scores': self.betting_scores,
            'cumulative_scores': self.cumulative_scores,
            'average_score': np.mean(self.betting_scores),
            'total_steps': len(self.betting_scores)
        }

if __name__ == "__main__":
    np.random.seed(42)
    
    n_steps = 100
    process1 = np.random.normal(0.1, 1, n_steps) 
    process2 = np.random.normal(0.0, 1, n_steps)
    
    sbt = SequentialBettingTest()
    
    for t in range(n_steps):
        hist1 = process1[:t] if t > 0 else np.array([])
        hist2 = process2[:t] if t > 0 else np.array([])
    
        capital, score = sbt.update(
            process1[t], 
            process2[t],
            null_hypothesis_difference=0.0,
            x1_history=hist1,
            x2_history=hist2
        )
    results = sbt.get_results()
    
    print("\nSequential Betting Test Results:")
    print(f"Number of steps: {results['total_steps']}")
    print(f"Final capital: {results['final_capital']:.2f}")
    print(f"Average betting score: {results['average_score']:.2f}")
    t_stat, p_value = stats.ttest_ind(process1, process2)
    print(f"\nTraditional t-test p-value: {p_value:.4f}")
```

[^1]: You will have to be generous with notions here.

[^2]: The standard form of Gibbs's inequality (also known as the information inequality) states that: $-\sum_{i=1}^n p_i \log p_i \leq -\sum_{i=1}^n p_i \log q_i$ with equality if and only if $p_i = q_i$ for all $i$.

[^3]: To find the power, we need to calculate $Q(X > 16.5)$ where under $Q$, $X \sim N(1, 10^2)$. $P(X > 16.5) = P(\frac{X-1}{10} > \frac{16.5-1}{10}) = P(Z > 1.55) \text{ where } Z \text{ is standard normal} = 1 - \Phi(1.55) \approx 0.06 \text{ or } 6\%$