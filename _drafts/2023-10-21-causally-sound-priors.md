# What Your A/B Test Prior Is Actually Saying About Causality

When you run a Bayesian A/B test with a binary outcome, you need a joint prior on two probabilities: the event rate in control, $\theta_0$, and the event rate in treatment, $\theta_1$. This sounds like a minor technical detail — the kind of thing you decide in ten minutes and never revisit. It isn't. The prior you choose makes strong implicit statements about the causal structure of your experiment, and in most cases those statements are wrong.

There are two standard approaches. You can put independent Beta priors directly on $(\theta_0, \theta_1)$, which is conjugate and analytically tractable but encodes the assumption that knowing the control rate tells you nothing about the treatment rate. Or you can reparametrise into log-odds space and put Gaussians on the baseline log-odds and the log odds ratio, which does induce prior correlation between arms but leaves you trying to reason intuitively about unbounded parameters that nobody outside of a logistic regression textbook thinks in terms of.

Neither of these is obviously the right choice. A recent paper by Irons and Cinelli (2023) proposes a third option that's been sitting in the literature, largely unnoticed, for decades — and it's good enough that I think it should become the default for this class of problem.

## The Problem With Independent Priors

The independent Beta (IB) prior is the path of least resistance:

$$\theta_0 \sim \text{Beta}(a_0, b_0) \perp\!\!\!\perp \theta_1 \sim \text{Beta}(a_1, b_1)$$

Its main virtue is that it's conjugate to the binomial likelihood, so the posterior is closed-form and you never need to touch an MCMC sampler. That's a real advantage.

The problem is what it implies about the relationship between your two arms. By assuming prior independence, you're saying that observing a lot of adverse events in the control group tells you nothing about what to expect in the treatment group. But that's almost never true. If your control conversion rate comes in at 1%, your treatment conversion rate is almost certainly not uniformly distributed over $[0, 1]$ — it's going to be concentrated somewhere near 1%. The two rates share a common underlying reality: the population, the product, the measurement. Prior independence ignores all of that.

And because independence in the prior plus independence in the likelihood implies independence in the posterior, this failure persists after you see the data. The posterior for $\theta_0$ is informed only by control observations, and the posterior for $\theta_1$ only by treatment observations, even though both are estimating rates for the same people responding to related interventions.

The logit transformation (LT) approach fixes the dependence problem but creates a different one. Priors on log-odds are notoriously hard to specify meaningfully. What does a $\mathcal{N}(0, 1)$ prior on the log odds ratio actually mean in terms of effect sizes you'd consider plausible? You can work it out, but it requires a translation step that most practitioners don't perform and that creates a gap between the prior you specify and the beliefs you actually hold.

## Thinking Causally About Binary Experiments

The key insight in Irons and Cinelli's paper is that you can think about the prior problem from a causal inference perspective rather than a statistical one, and doing so leads naturally to a better parametrisation.

In a randomised controlled trial, each participant $i$ has two potential outcomes: $Y_i(0)$ (what would happen under control) and $Y_i(1)$ (what would happen under treatment). The observed outcome is whichever one they actually received. The joint distribution of $(Y_i(0), Y_i(1))$ — across the whole population — falls into exactly four types:

- **Doomed**: $Y_i(0) = 1, Y_i(1) = 1$ — adverse outcome regardless.
- **Immune**: $Y_i(0) = 0, Y_i(1) = 0$ — no adverse outcome regardless.
- **Preventive**: $Y_i(0) = 1, Y_i(1) = 0$ — treatment prevents the adverse outcome.
- **Causal**: $Y_i(0) = 0, Y_i(1) = 1$ — treatment causes the adverse outcome.

The observed binomial proportions $\theta_0$ and $\theta_1$ are just the margins of this $2 \times 2$ table of potential outcomes. The cell probabilities $p_{jk} = P(Y_i(0) = j, Y_i(1) = k)$ determine both margins:

$$\theta_0 = p_{10} + p_{11}, \quad \theta_1 = p_{01} + p_{11}$$

This is what makes the independence assumption look so strange from a causal perspective. $\theta_0$ and $\theta_1$ share $p_{11}$ — the doomed fraction — in both their definitions. If the doomed fraction is large, both rates will be high. Assuming prior independence between $\theta_0$ and $\theta_1$ is equivalent to ignoring this shared dependence entirely.

## The BREASE Parametrisation

Irons and Cinelli propose thinking about the problem in terms of three quantities that clinicians — and frankly, any competent experimenter — already reason about when they design a study:

**Baseline risk** $\theta_0$: the probability of an adverse outcome in the absence of treatment.

**Efficacy** $\eta_e$: the probability that treatment prevents an adverse outcome for someone who would otherwise have experienced one. Formally, $\eta_e = P(Y_i(1) = 0 \mid Y_i(0) = 1)$.

**Risk of adverse side effects** $\eta_s$: the probability that treatment causes an adverse outcome for someone who would otherwise have been fine. Formally, $\eta_s = P(Y_i(1) = 1 \mid Y_i(0) = 0)$.

These are conditional probabilities of sufficient causation — $\eta_e$ is the probability that treatment is sufficient to cure or save, $\eta_s$ the probability it's sufficient to harm. They're also the quantities you naturally think about when you're deciding whether to run a test in the first place.

The treatment risk then decomposes cleanly via the law of total probability:

$$\theta_1 = (1 - \eta_e)\theta_0 + \eta_s(1 - \theta_0)$$

This is the BREASE decomposition (Baseline Risk, Efficacy, Adverse Side Effects). It makes the dependence between $\theta_0$ and $\theta_1$ explicit: $\theta_1$ is a function of $\theta_0$, modified by how effective the treatment is and how much harm it causes. Setting $\eta_e = \eta_s = 0$ gives $\theta_1 = \theta_0$; the null hypothesis of no treatment effect falls out naturally.

An aside worth noting: the quantity $1 - \theta_1/\theta_0$, which is commonly reported as "vaccine efficacy" in clinical trials, is only a valid estimate of $\eta_e$ under the monotonicity assumption that the treatment never causes harm ($\eta_s = 0$). The BREASE framework makes this assumption explicit and testable rather than leaving it implicit.

## The Prior

The natural prior for this parametrisation is a set of independent Beta distributions on the three variation-independent parameters:

$$\theta_0 \sim \text{Beta}^*(\mu_0, n_0) \perp\!\!\!\perp \eta_e \sim \text{Beta}^*(\mu_e, n_e) \perp\!\!\!\perp \eta_s \sim \text{Beta}^*(\mu_s, n_s)$$

where $\text{Beta}^*(\mu, n)$ is a Beta distribution with mean $\mu$ and prior "sample size" $n = a + b$. The hyperparameters are directly interpretable: $\mu_0$ is your prior belief about the control rate, $\mu_e$ is your prior belief about how often the treatment will work for people it's meant to help, and $\mu_s$ is your prior belief about how often it will cause harm.

This is considerably easier to elicit than hyperparameters for log-odds. If you're running a test on a feature that affects 5% of users, you'd set $\mu_0 \approx 0.05$. If you have good evidence from similar past experiments that your treatment should help about 30% of people who would otherwise have a bad outcome, you'd set $\mu_e \approx 0.3$. If side effects are expected to be small, you'd set $\mu_s$ close to zero with a tight prior.

Although the prior on $(\theta_0, \eta_e, \eta_s)$ is independent, the induced prior on $(\theta_0, \theta_1)$ is not — and has a controllable direction:

$$\text{Cor}(\theta_0, \theta_1) \begin{cases} < 0 & \text{if } \mu_e + \mu_s > 1 \\ = 0 & \text{if } \mu_e + \mu_s = 1 \\ > 0 & \text{if } \mu_e + \mu_s < 1 \end{cases}$$

For most reasonable experiments where you expect modest effects and small side effects, $\mu_e + \mu_s < 1$ and the two arms are positively correlated a priori, as they should be.

## Connection to the Dirichlet Prior

The BREASE prior is a generalisation of the Dirichlet distribution on the response type probabilities $p = (p_{00}, p_{01}, p_{10}, p_{11})$. In fact, the Dirichlet prior — which appears in the literature on partially identified problems and non-compliance analysis — corresponds to the special case where the prior sample sizes satisfy:

$$n_e = \mu_0 n_0, \quad n_s = (1-\mu_0) n_0$$

In other words, the Dirichlet implicitly forces you to have the same "amount of prior information" about $\theta_0$, $\eta_e$, and $\eta_s$, scaled by the baseline rate. The BREASE framework relaxes this constraint. You might have a well-calibrated estimate of the baseline rate from years of historical data ($n_0$ large) while having very little idea how effective the treatment will be ($n_e$ small). The Dirichlet can't represent this distinction; BREASE can.

The generalisation is a grouped Dirichlet distribution on $p$, which has been studied since the 1970s but was rarely connected to the problem of prior specification in binary experiments.

## Tractability

The reason independent Beta priors are popular is that they make computation trivial. You'd expect that moving to a more expressive model would cost you that tractability. Remarkably, it doesn't.

Because the likelihood (3.4 in the paper) is polynomial in $(\theta_0, \eta_e, \eta_s)$ after expanding via the binomial theorem twice, the marginal likelihood has an analytical form — a weighted sum of beta functions:

$$L_1(D) = \binom{N_0}{y_0}\binom{N_1}{y_1} \sum_{j=0}^{y_1} \sum_{k=0}^{N_1-y_1} \binom{y_1}{j}\binom{N_1-y_1}{k} \frac{B(k+\mu_e n_e,\, j+(1-\mu_e)n_e)}{B(\mu_e n_e,\, (1-\mu_e)n_e)} \cdot \frac{B(\cdot, \cdot)}{B(\cdot,\cdot)} \cdot \frac{B(\cdot,\cdot)}{B(\cdot,\cdot)}$$

This means you get analytical Bayes factors for free — critical if you want to do hypothesis testing rather than just estimation. The posterior is a finite mixture of independent Beta distributions, and you can draw exact samples via data augmentation (Algorithm 1 in the paper) rather than MCMC.

The data augmentation step is elegant. Rather than sampling $(\theta_0, \eta_e, \eta_s)$ directly, you first impute the unobserved counterfactual counts — how many people in the treatment arm would have had the adverse outcome if untreated ($y_1(0)$) and how many control subjects wouldn't have had the adverse outcome if treated ($x_1(1)$). Conditional on these imputed counts, the parameters have an independent Beta posterior and can be sampled exactly. This avoids the multimodality issues that cause JAGS and Stan to fail on this model in certain regimes.

## Why Bayes Factors Here Are Sensitive to the Prior

The aspirin example in the paper is instructive about where prior sensitivity actually matters. The frequentist analysis of the Physicians' Health Study gives a clear result: 26 fatal myocardial infarctions out of 11,034 in placebo versus 10 out of 11,037 in the aspirin group, p = 0.008.

The Bayesian picture is less clear. The Bayes factor under the IB default prior strongly favours the null ($\text{BF}_{01} = 20.27$). The LT approach gives moderate evidence for an effect ($\text{BF}_{10} = 5.24$). The BREASE default yields essentially no evidence either way ($\text{BF}_{10} = 1.2$). Three approaches, wildly different conclusions.

What's happening? The Bayes factor penalises the alternative hypothesis for probability mass placed in regions inconsistent with the data. The IB prior, by placing independent uniforms on $\theta_0$ and $\theta_1$, implicitly spreads prior mass over a large region of parameter space including scenarios where aspirin causes more heart attacks than placebo. This diffuse alternative is heavily penalised relative to the sharp null.

The BREASE sensitivity analysis makes this explicit: as you lower the prior expectation of side effects $\mu_s$ below 1%, strong evidence for the alternative hypothesis emerges ($\text{BF}_{10} > 10$). For aspirin — a well-understood over-the-counter medication — it's extremely implausible that aspirin would cause fatal myocardial infarction in a substantial fraction of otherwise healthy patients. Encoding this prior belief ($\mu_s$ near zero) yields the conclusion you'd expect from domain knowledge.

The COVID vaccine trial, by contrast, is robust to essentially any choice of prior. The Bayes factor in favour of an effect is on the order of $10^{33}$ across the entire prior hyperparameter space. No reasonable prior elicitation changes the conclusion.

## A Default Prior for Industry Use

For practical A/B testing where you want a sensible default before incorporating domain knowledge, Irons and Cinelli suggest BREASE$(1/2, \mu, \mu; 2, 1, 1)$ with $\mu = 0.3$. This specification:

- Puts flat marginal priors on both $\theta_0$ and $\theta_1$ (matching the IB flat prior on marginals).
- Induces positive prior correlation between arms.
- Assumes no treatment effect on average.
- Concentrates prior mass on the diagonal $\theta_0 = \theta_1$, which is desirable for hypothesis testing because it gives the null a fair hearing.
- Has a single tunable parameter $\mu$ with a direct interpretation: the expected efficacy (and expected side effect rate) of the treatment.

This is a considerably more principled default than uniform independent Betas, with essentially no added computational cost.

## Practical Takeaways

The BREASE framework matters for a few concrete reasons.

**Prior elicitation becomes conversations with domain experts.** Instead of asking a product manager to specify a log-odds ratio prior, you can ask: "What fraction of users who would normally churn do you expect this feature to retain?" and "Do we have any reason to think this could make things worse for users who would otherwise be fine?" These are questions that get real answers.

**Sensitivity analysis becomes meaningful.** Showing a stakeholder that your conclusion is robust across all plausible values of $\mu_s$ — or that it isn't — is a comprehensible statement. Showing them a sensitivity plot over log-odds hyperparameters is not.

**Exact posterior sampling is available even when MCMC fails.** The data augmentation algorithm is exact and fast, and handles prior-data conflict cases where gradient-based samplers get stuck.

**The framework separates identified from partially-identified parameters explicitly.** $(\theta_0, \theta_1)$ are identified; $(\eta_e, \eta_s)$ are not. Even if you don't care about the causal interpretation of $\eta_e$ and $\eta_s$, they provide a principled route to a joint prior on the quantities you do care about.

The main limitation is that the framework applies to binary outcomes and binary treatment. Extensions to continuous outcomes, survival data, or multi-arm tests are not handled, though the authors note these as directions for future work.

The paper is by Nicholas Irons and Carlos Cinelli, both at the University of Washington Department of Statistics. The preprint is [available on arXiv](https://arxiv.org/abs/2310.XXXXX) and the R implementation is straightforward given the closed-form expressions derived in the appendix.

---

*The ideas here connect closely to the broader project of making Bayesian experimentation more causally grounded. If you're interested in how potential outcomes thinking intersects with Bayesian inference in practice, the references to Richardson, Evans, and Robins (2011) and the Tian and Pearl causality literature are worth following up.*