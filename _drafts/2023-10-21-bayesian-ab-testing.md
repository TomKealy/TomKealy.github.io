# Priors for Bayesian Inference in Controlled Experiments

## Introduction

Suppose that you had a new diet that you wanted to test, say you wanted to know if eating nothing but potatoes would reduce body mass.

One way to test this diet would be to recruit a group of plucky volunteers and, by hook or by crook, make them eat nothing but potatoes for a period of time, say a month. For each volunteer you would measure their body mass at the start of the procedure and at the end. You declare the effect of the intervention as the mean difference in body mass of your volunteers. So at the end of the study you measure the ending weights of your volunteers, calculate their individual differences, and find that on average eating nothing but potatoes led to a loss of, say, 4kg. Would this be enough to conclude that eating nothing but potatoes causes the 4kg weight loss?

Upon closer inspection, this study is not really enough to conclude that the diet caused the weight loss. The most we can really say is that the diet was associated with weight loss, but the study design limits the conclusions we can reach[^1]. We thought we were doing science, but sadly we were merely performing a ritual.

[^1]: A poorly intentioned interlocutor could always say that our study design limits the conclusions we can reach. Most of the time they would be wrong, but since we have created such a poor procedure in this case they are right.

We can improve our chances of saying something better if we do a little thinking before we can do any calculations. There are two major obstacles we need to overcome or circumvent before we can go from the results of our study to the conclusion we want (potatoes are good for weight loss).

Firstly, just because we made a lot of people eat nothing but potatoes and they lost weight, this doesn't mean that the potato diet caused the weight loss. We don't know what would have happened to these volunteers if they hadn't eaten only potatoes. Maybe the volunteers would have lost weight anyway due to seasonal changes in activity. The participants might have had natural weight fluctuations. Or the study itself caused the weight loss due to an increased awareness amongst the participants of their eating habits or the fact that they knew they were being observed.

Eating only potatoes likely means eating fewer calories overall (potatoes are filling but relatively low-calorie). The diet eliminates processed foods, sugar, and alcohol. The restrictiveness itself might reduce overall food intake. Volunteers might exercise more because they're in a "health study." In other words, the potato diet might inadvertently change other behaviors. Finally, we left body weight measurements up to the volunteers themselves. Body mass can fluctuate, depending on how much they drank, when they weighed themselves, and whatever they ate/drank immediately before measurement.

These are all problems of internal validity, i.e. whether our intervention alone caused the changes we see. Internal validity is a question of whether we can explain the past, i.e. it's a retrospective quality of the study. As well as questions of whether we can conclude if our intervention worked we also want to know if it will continue to work. That's just another way of saying will our study generalise to other populations.

There are a few reasons we might be doubtful. The first is that our 'Plucky volunteers' who agree to eat only potatoes for a month are not representative of the general population. They might be more motivated to lose weight, more disciplined, or even healthier to begin with. Anybody who willingly eats nothing but potatoes is already very odd. "By hook or by crook" suggests that we forced compliance with the diet in unusual ways that wouldn't necessarily apply in real life. Real people trying this diet at home would fail in ways which the study didn't capture. And one month isn't really a long time; the initial weight loss could just be water weight, our bodies adapt metabolically over time, and real people would struggle to adhere to this very boring diet once the novelty wears off. Even if the diet "worked" for these specific volunteers under these specific conditions, we have little reason to believe it would work for other people in normal life circumstances. This is all another way of saying that our study suffers from issues of external validity.

There are two ways to improve this study. We need to add a control group (randomized to either potato diet or their normal diet) to establish causation, and we must recruit a more representative sample and conduct the study in more naturalistic conditions, or replicate across different populations and settings.

What we really want to do is to run a randomised controlled trial. Let's break that term of art down a little. By controlled trial we mean that we systematically manipulate the treatments and control other variables. This allows us to determine whether the treatment causes changes in the outcomes. But why do we want this control?

It's because something other than our treatment could have caused the outcome. We call these 'something others' extraneous variables, or confounders. They are problematic because they provide alternative explanations for changes in the outcomes. For example in our diet study the volunteers could be especially sporty with a high level of physical training. This could go both ways: the high initial degree of physical conditioning could mean that the volunteers had little weight to lose, and so the study underestimates the true potential of the potato diet. It could also mean that our volunteers are disciplined enough to stick to the diet, and that discipline is the determining factor in weight loss. All this is a fancy way of saying that confounders might cause the changes rather than the treatment. A controlled experiment means that we, as scientists, need to find ways to control these extraneous variables to minimise any systematic differences between treated and untreated participants other than the treatment itself. This control isolates the effects of the treatment.

The second piece of the term of art was randomised. The point of randomization is to turn bias into a source of (random) error. Consider our running example: the potato diet. If I, as the researcher, intervene to ensure that athletes are in the treatment arm of my study, then the internal validity of the study has been compromised. We cannot get an unbiased estimate of the treatment effect of the intervention.

However, if I flip a coin for each athlete, and by chance I get a sequence of heads such that all athletes end up in the treatment arm, the resulting estimate of the treatment effect will be unbiased but just very imprecise (far from the true treatment effect due to sampling variability). Randomization promises us two things: unbiasedness as well as a framework for bounding the uncertainty of our estimates.

So we have addressed the objections to our study protocol. We need to randomly sample members of the representative population, split them into two groups, and apply the treatment (eat nothing but potatoes for a month) to only one group. Now we can establish something much stronger than our initial claim, that eating nothing but potatoes is associated with a 4kg weight loss. If we did our study correctly we can conclude both that it was the diet which caused the weight loss in our volunteers and that a potatoes only diet would cause weight loss in other members of the target population.

We can go even farther. We could make it so that the researcher does the randomisation, but doesn't know which arm receives the treatment. This practice has become a gold standard in applied research because it eliminates any potential bias on the part of the practitioners. For example by giving preferential treatment to volunteers in the treatment arm of the study[^2]. We can't double blind the potato diet, as everyone involved in the study would know who's eating potatoes, but double blinding is still a good thing to try to do.

[^2]: Double blinding is also an ethical practice. In drug trials people who receive a placebo are patients too, and they deserve the same standard of care as those patients receiving the drug which we hope is effective. We want to do good by the people who have, often literally, put their life on the line for science.

Another practice we should consider is pre-registration. Pre-registration is the practice of researchers publicly posting critical decision points in a study prior to conducting it. This makes their plans transparent, making deviations from those plans discoverable, and improving the validity of tests of significance. Preregistration helps address specific threats to internal validity of a study related to reporting results. In plain English, once you have your data it's very easy to come to any conclusion that you want. Eating potatoes didn't lead to weight loss on average? Perhaps you happened to measure your patients cholesterol, or their blood pressure, before you began the study? Measure the same things afterwards, reanalyse the data, and then report any interesting finding after the fact. Stay silent about the analyses which yield nothing. This is called 'p-hacking' and is very easy to do, just out of your desperation that your efforts amount to something. Pre-registration keeps you honest.

Let's take a moment to understand what we've done. We started simply: we were going to feed volunteers nothing but potatoes for a month and measure what happened to their weight. On reflection there were a lot of ways that our initial study design could not achieve the conclusion we sought. We refined our design to deal with two kinds of validity; internal validity, which caused doubts about whether the study could explain its finding; and external validity, which caused doubts about whether our diet would cause weight loss in people who weren't part of our study. We addressed all of these doubts with a simple idea: sample representative people from the population at large, and randomise which treatment they receive. Now we can make strong causal claims about our treatment.

This simple idea has had profound consequences. Controlled experiments are the foundation of basically everything we know about:

- Drugs and drug safety
- Digital product development and optimization
- Public policy implementation
- Educational interventions
- Agricultural practices
- Industrial processes

They are hugely important across virtually every field of applied research. Finding a way to squeeze more out of them is, despite it looking like a solved problem, something worth spending our collective time on. 

The simplest version of this problem — and the one we'll focus on — is what's called a binary experiment. Each participant receives either the treatment or the control, and the outcome is binary: they either experience the adverse event or they don't. A drug trial measuring mortality, a vaccine trial measuring infection, an A/B test measuring whether a user churns. The same structure throughout.
When participants are independent draws from a common population, the statistical problem reduces to comparing two binomial proportions — the event rate in each arm. This is about as simple as inference gets, and Bayesian approaches to it go back to Bayes and Laplace themselves. You specify a joint prior over both rates, observe the counts, update. The posterior tells you about the risk difference, the risk ratio, whatever contrast you care about.

Simple problem, you'd think simple solution. But the prior is trickier than it looks. There are two mainstream approaches, and both have meaningful flaws.

The first assigns independent Beta priors to each rate. This is conjugate and computationally painless, but it encodes the assumption that knowing the control rate tells you nothing about the treatment rate — which is almost never true. The second reparametrises into log-odds and puts Gaussians on the baseline log-odds and the log odds ratio. This does induce dependence between the arms, but odds ratios are notoriously hard to reason about intuitively, which makes prior elicitation and sensitivity analysis painful in practice.

There's a better way, and it comes from thinking causally about what the experiment is actually doing. Randomisation solves the identification problem — it ensures our estimate of the treatment effect is unbiased. But it doesn't tell us how to think about what's happening underneath. When we specify a prior over the two event rates, we're implicitly making assumptions about the causal structure of the experiment, whether we intend to or not. Independent Beta priors say the two rates have nothing to do with each other before we see the data. That's a causal claim, and it's usually wrong.

The better starting point is to ask: what can the treatment actually do to a person? It can help someone who would otherwise have had a bad outcome. It can harm someone who would otherwise have been fine. Or it does nothing. These three possibilities — and their relative probabilities — are what should be driving the prior.


## Potential Outcomes

The potential outcomes framework of causal inference (Rubin, 1974; Neyman, 1990). Let $N$ denote the total number of participants in the study, $Z_i$ a binary treatment indicator and $Y_i$ a binary outcome indicator for subject $i \in \{1,...,N\}$. We denote by $Y_i(z)$ the potential outcome of subject $i$ under the experimental condition $Z_i = z$, where $z = 0$ indicates the control and $z = 1$ the treatment condition. Under the standard consistency assumption, we have that the observed outcome of subject $i$ equals the potential outcome associated to the experimental condition that subject $i$ has actually received, i.e., $Y_i = Y_i(Z_i)$. Throughout the paper, we adopt the convention that $Y_i = 1$ denotes an adverse outcome, such as death or the contraction of a disease. We take a super-population perspective, and assume that subjects are independent and identically distributed (i.i.d.) draws from a common population. We assume complete randomization, which implies ignorability of the treatment assignment, $\{Y_i(1),Y_i(0)\} \perp\!\!\!\perp Z_i$.

## Data Setup

When subjects are independently drawn from a common super-population and the treatment is assigned at random, it follows that the observed counts of adverse outcomes in each treatment arm,

$$y_0 = \sum_{i:Z_i=0} Y_i, \quad y_1 = \sum_{i:Z_i=1} Y_i,$$

follow independent binomial distributions:

$$y_0 \sim \text{Binomial}(N_0, \theta_0) \perp\!\!\!\perp y_1 \sim \text{Binomial}(N_1, \theta_1),$$

where here, $\theta_1 = P(Y_i(1) = 1)$, $N_1 = \sum_i Z_i$ denote the probability of an adverse outcome and the sample size of the treatment group, and $\theta_0 = P(Y_i(0) = 1)$, $N_0 = N - N_1$ are the analogous quantities for the control group. We refer to the probabilities $\theta_0$ and $\theta_1$ as the baseline risk and risk of treatment, respectively.

This defines the likelihood under the marginal parameterization of a binary experiment—so called because the parameters $(\theta_0, \theta_1)$ are defined in terms of the marginal distribution of the potential outcomes $Y_i(0)$ and $Y_i(1)$:

$$L(D|\theta_0, \theta_1) = {N_0 \choose y_0} \theta_0^{y_0} (1-\theta_0)^{N_0-y_0} \times {N_1 \choose y_1} \theta_1^{y_1} (1-\theta_1)^{N_1-y_1},$$

where hereafter we denote the observed data by $D = (y_0, y_1, N_0, N_1)$. To determine the effect of treatment, if any, Bayesian inference is carried out using the posterior distribution of the parameters $(\theta_0, \theta_1)$, which requires specification of a prior distribution for $(\theta_0, \theta_1)$.

There are two main parameterizations with accompanying priors currently in use, discussed extensively in Agresti and Min (2005) and Dablander et al. (2022)—these are the independent beta (IB) and logit transformation (LT) approaches, which we now discuss.

## Priors

Let's first deal with a simple two-variant experiment test, where the metric of interest is the proportion of subjects which recover from a disease. We call our variants Control and Test. Confusingly, these are labeled as A and B. The Control (A) variant either receives a placebo, or the state of the art treatment. The Test (B) variant receives the new treatment. We wish to infer whether the Test treatment is superior, in some sense to be defined later. Let us first simply infer the underlying rates for the different arms of a fictional experiment.

Let $p_A, p_B$ be the true recovery rate for variants A and B respectively. Then the outcome of whether a subject recovers in variant A is the random variable $\text{Bernoulli}(p_A)$, and similarly for variant B.

We need to make an assumption here, which is that any particular patient's chance of recovery is independent of any other patient's chance of recovery. On the face of it, this is a straightforward assumption to make: if I take a pill, then the pill only affects me. However there are plenty of situations where this assumption doesn't hold. For example, consider a government-run vaccination drive. Researchers want to measure how receiving the vaccine affects an individual's immunity. However, vaccination doesn't exist in isolation—it protects not only the vaccinated person but also those around them through herd immunity.

In a real-world campaign in Indonesia, researchers tried to estimate the vaccine's individual effectiveness. But as more people were vaccinated, community transmission dropped, benefiting even those who remained unvaccinated. Here, SUTVA broke down—the treatment (vaccination) of one person directly influenced others' outcomes (their infection status).

The assumption we made is called the Stable Unit Treatment Value Assumption (SUTVA) and is the spine of all causal inference methods (of which randomised controlled experiments are the gold standard).

Finally we can write the total number of recovered subjects in each variant as:

$$y \sim \sum^N \text{Bernoulli}(\theta) = \text{Binomial}(N, p)$$

Where

$$\text{Binomial}(N, p) = \Pr(X=k) = {n \choose k} p^{k}(1-p)^{N-k}$$

The simplest thing is to generate some fake data with count outcomes. So we do that, say by creating a function which accepts an $N$ and a $p$ as parameters and produces binomially distributed data.

### Independent Priors

The next step is to choose a prior for this likelihood. The simplest one to choose is the Beta distribution.

$$p_A, p_B \sim \text{Beta}(a, b)$$

We say that a continuous random variable has a Beta distribution with shape parameters $a$ and $b$ if its probability density satisfies:

$$p(u) \sim u^{a-1} (1-u)^{b-1}$$

One useful fact about the Beta distribution is that a Beta times a beta is beta. We can see this from the example above: if the prior is a $\text{Beta}(a, b)$ distribution and the likelihood is $\text{Binomial}(N, p)$ then the posterior is $\text{Beta}(a + k, b + N - k)$. This is the very useful property called conjugacy. What this means is that the outcome is just moved and scaled along the relevant axis, and that the distributional form of the posterior remains the same. It saves us, the statisticians, from either doing some complicated integrals, or using sophisticated algorithms to estimate the posterior distribution.

Anyway, here is a function which can estimate the posterior if we assume a Beta prior on the likelihood.

Here is a Python block of code to estimate the posterior using numerical methods.

However, this approach has the flavour of convenience about it. We assume away a few of the difficult parts of the problem to make progress. This is a fine way to start, and it gives us a baseline which we can compare future improvements, but it's no real solution.

In particular, we have assumed two things about the data:

- That there is total independence across the arms of the experiment, i.e. there is no common component of the problem.
- That all subjects respond equally to the treatment.

A significant drawback of the simple prior approach is the restrictive assumption of independence between $p_A$ and $p_B$. In most experimental settings, we would expect our knowledge about the risks in the control and treatment groups to be dependent. For example, if we know that the population prevalence of an infectious disease is approximately 1%, we would expect the prevalence of the disease among those receiving a vaccine to be concentrated around 1% or below, reflecting the common prior belief that it is unlikely that the vaccine would cause the disease. This simple prior fails to accommodate this natural dependence between risks in each arm of the experiment. Since independence in the prior and the likelihood implies independence a posteriori, this failure also extends to the posterior.

### Logit Transformation

We can address the first assumption (that there is no common effect across both arms of the experiment) with a different model. We can reparameterise the model in terms of the logit-transformed risks, by defining the parameters $\beta, \psi$ satisfying

$$\log\left(\frac{p_A}{1-p_A}\right) = \beta - \frac{\psi}{2}$$

$$\log\left(\frac{p_B}{1-p_B}\right) = \beta + \frac{\psi}{2}$$

We can then assign the following independent priors to $\beta, \psi$:

$$\beta \sim \mathcal{N}(\mu_{\beta}, \sigma_{\beta}^2)$$

$$\psi \sim \mathcal{N}(\mu_{\psi}, \sigma_{\psi}^2)$$

where $\mu = (\mu_{\beta}, \mu_{\psi})$ and $\sigma = (\sigma_{\beta}, \sigma_{\psi})$ are hyperparameters.

This prior creates correlation between $p_A$ and $p_B$ through their shared dependence on $\beta$ and $\psi$. The tradeoff is that we lose some interpretability. Here $\beta$ represents the "grand log odds"—basically the average log odds across treatment arms—while $\psi$ is the log odds ratio. Odds ratios are famously hard to think about intuitively, which makes it tricky to set reasonable prior means and variances for these unbounded hyperparameters. You can make this easier by thinking about marginal effects and then translating those back into log-odds, but it's still not as straightforward.

There are also computational downsides to this logit approach compared to the independent beta prior. Unlike before, we can't calculate marginal likelihoods and Bayes factors in closed form, so we need to use approximate posterior sampling methods instead.

## Response Type Parameterisation

The IB and LT approaches focus on the margins of the joint distribution of the potential outcomes $Y_i(0)$ and $Y_i(1)$. This focus is natural, because the observed data depends only upon the parameters $p_0$ and $p_1$. However, thinking in terms of their joint distribution reveals alternative ways of inducing prior dependence between these parameters. Specifically, the joint distribution of potential outcomes is fully characterized by four probabilities

$$p_{jk} = P(Y_i(0) = j, Y_i(1) = k), \quad j,k \in \{0,1\}$$

The probabilities $p = \{p_{jk}\}_{j,k \in \{0,1\}}$ describe the frequencies of the four possible response types in the population (Copas, 1973; Greenland and Robins, 1986). These include:

(i) the "doomed" $\{Y_i(0) = 1, Y_i(1) = 1\}$, for whom the adverse outcome occurs regardless of treatment; 

(ii) the "immune" $\{Y_i(0) = 0, Y_i(1) = 0\}$, for whom the adverse outcome does not occur regardless of treatment; 

(iii) the "preventive" $\{Y_i(0) = 1, Y_i(1) = 0\}$, for whom treatment prevents the adverse outcome; and, 

(iv) the "causal" $\{Y_i(0) = 0, Y_i(1) = 1\}$, for whom treatment causes the adverse outcome. 

Here $\theta_0$ and $\theta_1$, which satisfy $\theta_0 = p_{10} + p_{11}$ and $\theta_1 = p_{01} + p_{11}$, define the margins of Table 1.

Whereas in the marginal parameterization, independence of the likelihood and prior imply that estimation of $\theta_0$ is only informed by data in the control group (and similarly for $\theta_1$), the response type (RT) parameterization intertwines the data from each arm of the study. The shared dependence of $\theta_0$ and $\theta_1$ on the response type proportions reveals the link between outcomes in the control and treated groups.

A Bayesian approach to modeling the response type probabilities $p$ requires specification of a prior density supported on the probability simplex, making the Dirichlet distribution a natural candidate

$$p = (p_{00}, p_{10}, p_{01}, p_{11}) \sim \text{Dirichlet}(a_{00}, a_{10}, a_{01}, a_{11}), \quad a_{00}, a_{10}, a_{01}, a_{11} > 0$$

Indeed, priors of this type have been used in the analysis of partially identified quantities in randomized trials with non-compliance, such as in Chickering and Pearl (1996).

As we show next, the Dirichlet prior is a special case of our proposal, and our analysis not only extends it, but also clarifies its advantages and limitations as a means to induce the desired joint prior distribution on the two binomial proportions $(\theta_0, \theta_1)$.

The BREASE Parametrisation
Irons and Cinelli propose thinking about the problem in terms of three quantities that clinicians — and frankly, any competent experimenter — already reason about when they design a study:
Baseline risk θ0\theta_0
θ0​: the probability of an adverse outcome in the absence of treatment.

Efficacy ηe\eta_e
ηe​: the probability that treatment prevents an adverse outcome for someone who would otherwise have experienced one. Formally, ηe=P(Yi(1)=0∣Yi(0)=1)\eta_e = P(Y_i(1) = 0 \mid Y_i(0) = 1)
ηe​=P(Yi​(1)=0∣Yi​(0)=1).

Risk of adverse side effects ηs\eta_s
ηs​: the probability that treatment causes an adverse outcome for someone who would otherwise have been fine. Formally, ηs=P(Yi(1)=1∣Yi(0)=0)\eta_s = P(Y_i(1) = 1 \mid Y_i(0) = 0)
ηs​=P(Yi​(1)=1∣Yi​(0)=0).

These are conditional probabilities of sufficient causation — ηe\eta_e
ηe​ is the probability that treatment is sufficient to cure or save, ηs\eta_s
ηs​ the probability it's sufficient to harm. They're also the quantities you naturally think about when you're deciding whether to run a test in the first place.

The treatment risk then decomposes cleanly via the law of total probability:
θ1=(1−ηe)θ0+ηs(1−θ0)\theta_1 = (1 - \eta_e)\theta_0 + \eta_s(1 - \theta_0)θ1​=(1−ηe​)θ0​+ηs​(1−θ0​)
This is the BREASE decomposition (Baseline Risk, Efficacy, Adverse Side Effects). It makes the dependence between θ0\theta_0
θ0​ and θ1\theta_1
θ1​ explicit: θ1\theta_1
θ1​ is a function of θ0\theta_0
θ0​, modified by how effective the treatment is and how much harm it causes. Setting ηe=ηs=0\eta_e = \eta_s = 0
ηe​=ηs​=0 gives θ1=θ0\theta_1 = \theta_0
θ1​=θ0​; the null hypothesis of no treatment effect falls out naturally.

An aside worth noting: the quantity 1−θ1/θ01 - \theta_1/\theta_0
1−θ1​/θ0​, which is commonly reported as "vaccine efficacy" in clinical trials, is only a valid estimate of ηe\eta_e
ηe​ under the monotonicity assumption that the treatment never causes harm (ηs=0\eta_s = 0
ηs​=0). The BREASE framework makes this assumption explicit and testable rather than leaving it implicit.

The Prior
The natural prior for this parametrisation is a set of independent Beta distributions on the three variation-independent parameters:
θ0∼Beta∗(μ0,n0)⊥ ⁣ ⁣ ⁣⊥ηe∼Beta∗(μe,ne)⊥ ⁣ ⁣ ⁣⊥ηs∼Beta∗(μs,ns)\theta_0 \sim \text{Beta}^*(\mu_0, n_0) \perp\!\!\!\perp \eta_e \sim \text{Beta}^*(\mu_e, n_e) \perp\!\!\!\perp \eta_s \sim \text{Beta}^*(\mu_s, n_s)θ0​∼Beta∗(μ0​,n0​)⊥⊥ηe​∼Beta∗(μe​,ne​)⊥⊥ηs​∼Beta∗(μs​,ns​)
where Beta∗(μ,n)\text{Beta}^*(\mu, n)
Beta∗(μ,n) is a Beta distribution with mean μ\mu
μ and prior "sample size" n=a+bn = a + b
n=a+b. The hyperparameters are directly interpretable: μ0\mu_0
μ0​ is your prior belief about the control rate, μe\mu_e
μe​ is your prior belief about how often the treatment will work for people it's meant to help, and μs\mu_s
μs​ is your prior belief about how often it will cause harm.

This is considerably easier to elicit than hyperparameters for log-odds. If you're running a test on a feature that affects 5% of users, you'd set μ0≈0.05\mu_0 \approx 0.05
μ0​≈0.05. If you have good evidence from similar past experiments that your treatment should help about 30% of people who would otherwise have a bad outcome, you'd set μe≈0.3\mu_e \approx 0.3
μe​≈0.3. If side effects are expected to be small, you'd set μs\mu_s
μs​ close to zero with a tight prior.

Although the prior on (θ0,ηe,ηs)(\theta_0, \eta_e, \eta_s)
(θ0​,ηe​,ηs​) is independent, the induced prior on (θ0,θ1)(\theta_0, \theta_1)
(θ0​,θ1​) is not — and has a controllable direction:

Cor(θ0,θ1){<0if μe+μs>1=0if μe+μs=1>0if μe+μs<1\text{Cor}(\theta_0, \theta_1) \begin{cases} < 0 & \text{if } \mu_e + \mu_s > 1 \\ = 0 & \text{if } \mu_e + \mu_s = 1 \\ > 0 & \text{if } \mu_e + \mu_s < 1 \end{cases}Cor(θ0​,θ1​)⎩⎨⎧​<0=0>0​if μe​+μs​>1if μe​+μs​=1if μe​+μs​<1​
For most reasonable experiments where you expect modest effects and small side effects, μe+μs<1\mu_e + \mu_s < 1
μe​+μs​<1 and the two arms are positively correlated a priori, as they should be.

Connection to the Dirichlet Prior
The BREASE prior is a generalisation of the Dirichlet distribution on the response type probabilities p=(p00,p01,p10,p11)p = (p_{00}, p_{01}, p_{10}, p_{11})
p=(p00​,p01​,p10​,p11​). In fact, the Dirichlet prior — which appears in the literature on partially identified problems and non-compliance analysis — corresponds to the special case where the prior sample sizes satisfy:

ne=μ0n0,ns=(1−μ0)n0n_e = \mu_0 n_0, \quad n_s = (1-\mu_0) n_0ne​=μ0​n0​,ns​=(1−μ0​)n0​
In other words, the Dirichlet implicitly forces you to have the same "amount of prior information" about θ0\theta_0
θ0​, ηe\eta_e
ηe​, and ηs\eta_s
ηs​, scaled by the baseline rate. The BREASE framework relaxes this constraint. You might have a well-calibrated estimate of the baseline rate from years of historical data (n0n_0
n0​ large) while having very little idea how effective the treatment will be (nen_e
ne​ small). The Dirichlet can't represent this distinction; BREASE can.

The generalisation is a grouped Dirichlet distribution on pp
p, which has been studied since the 1970s but was rarely connected to the problem of prior specification in binary experiments.

Tractability
The reason independent Beta priors are popular is that they make computation trivial. You'd expect that moving to a more expressive model would cost you that tr actability. Remarkably, it doesn't.
Because the likelihood (3.4 in the paper) is polynomial in (θ0,ηe,ηs)(\theta_0, \eta_e, \eta_s)
(θ0​,ηe​,ηs​) after expanding via the binomial theorem twice, the marginal likelihood has an analytical form — a weighted sum of beta functions:

L1(D)=(N0y0)(N1y1)∑j=0y1∑k=0N1−y1(y1j)(N1−y1k)B(k+μene, j+(1−μe)ne)B(μene, (1−μe)ne)⋅B(⋅,⋅)B(⋅,⋅)⋅B(⋅,⋅)B(⋅,⋅)L_1(D) = \binom{N_0}{y_0}\binom{N_1}{y_1} \sum_{j=0}^{y_1} \sum_{k=0}^{N_1-y_1} \binom{y_1}{j}\binom{N_1-y_1}{k} \frac{B(k+\mu_e n_e,\, j+(1-\mu_e)n_e)}{B(\mu_e n_e,\, (1-\mu_e)n_e)} \cdot \frac{B(\cdot, \cdot)}{B(\cdot,\cdot)} \cdot \frac{B(\cdot,\cdot)}{B(\cdot,\cdot)}L1​(D)=(y0​N0​​)(y1​N1​​)j=0∑y1​​k=0∑N1​−y1​​(jy1​​)(kN1​−y1​​)B(μe​ne​,(1−μe​)ne​)B(k+μe​ne​,j+(1−μe​)ne​)​⋅B(⋅,⋅)B(⋅,⋅)​⋅B(⋅,⋅)B(⋅,⋅)​
This means you get analytical Bayes factors for free — critical if you want to do hypothesis testing rather than just estimation. The posterior is a finite mixture of independent Beta distributions, and you can draw exact samples via data augmentation (Algorithm 1 in the paper) rather than MCMC.
The data augmentation step is elegant. Rather than sampling (θ0,ηe,ηs)(\theta_0, \eta_e, \eta_s)
(θ0​,ηe​,ηs​) directly, you first impute the unobserved counterfactual counts — how many people in the treatment arm would have had the adverse outcome if untreated (y1(0)y_1(0)
y1​(0)) and how many control subjects wouldn't have had the adverse outcome if treated (x1(1)x_1(1)
x1​(1)). Conditional on these imputed counts, the parameters have an independent Beta posterior and can be sampled exactly. This avoids the multimodality issues that cause JAGS and Stan to fail on this model in certain regimes.

Why Bayes Factors Here Are Sensitive to the Prior
The aspirin example in the paper is instructive about where prior sensitivity actually matters. The frequentist analysis of the Physicians' Health Study gives a clear result: 26 fatal myocardial infarctions out of 11,034 in placebo versus 10 out of 11,037 in the aspirin group, p = 0.008.
The Bayesian picture is less clear. The Bayes factor under the IB default prior strongly favours the null (BF01=20.27\text{BF}_{01} = 20.27
BF01​=20.27). The LT approach gives moderate evidence for an effect (BF10=5.24\text{BF}_{10} = 5.24
BF10​=5.24). The BREASE default yields essentially no evidence either way (BF10=1.2\text{BF}_{10} = 1.2
BF10​=1.2). Three approaches, wildly different conclusions.

What's happening? The Bayes factor penalises the alternative hypothesis for probability mass placed in regions inconsistent with the data. The IB prior, by placing independent uniforms on θ0\theta_0
θ0​ and θ1\theta_1
θ1​, implicitly spreads prior mass over a large region of parameter space including scenarios where aspirin causes more heart attacks than placebo. This diffuse alternative is heavily penalised relative to the sharp null.

The BREASE sensitivity analysis makes this explicit: as you lower the prior expectation of side effects μs\mu_s
μs​ below 1%, strong evidence for the alternative hypothesis emerges (BF10>10\text{BF}_{10} > 10
BF10​>10). For aspirin — a well-understood over-the-counter medication — it's extremely implausible that aspirin would cause fatal myocardial infarction in a substantial fraction of otherwise healthy patients. Encoding this prior belief (μs\mu_s
μs​ near zero) yields the conclusion you'd expect from domain knowledge.

The COVID vaccine trial, by contrast, is robust to essentially any choice of prior. The Bayes factor in favour of an effect is on the order of 103310^{33}
1033 across the entire prior hyperparameter space. No reasonable prior elicitation changes the conclusion.

A Default Prior for Industry Use
For practical A/B testing where you want a sensible default before incorporating domain knowledge, Irons and Cinelli suggest BREASE(1/2,μ,μ;2,1,1)(1/2, \mu, \mu; 2, 1, 1)
(1/2,μ,μ;2,1,1) with μ=0.3\mu = 0.3
μ=0.3. This specification:


Puts flat marginal priors on both θ0\theta_0
θ0​ and θ1\theta_1
θ1​ (matching the IB flat prior on marginals).

Induces positive prior correlation between arms.
Assumes no treatment effect on average.
Concentrates prior mass on the diagonal θ0=θ1\theta_0 = \theta_1
θ0​=θ1​, which is desirable for hypothesis testing because it gives the null a fair hearing.

Has a single tunable parameter μ\mu
μ with a direct interpretation: the expected efficacy (and expected side effect rate) of the treatment.

This is a considerably more principled default than uniform independent Betas, with essentially no added computational cost.
Practical Takeaways

The BREASE framework matters for a few concrete reasons.
Prior elicitation becomes conversations with domain experts. Instead of asking a product manager to specify a log-odds ratio prior, you can ask: "What fraction of users who would normally churn do you expect this feature to retain?" and "Do we have any reason to think this could make things worse for users who would otherwise be fine?" These are questions that get real answers.
Sensitivity analysis becomes meaningful. Showing a stakeholder that your conclusion is robust across all plausible values of μs\mu_s
μs​ — or that it isn't — is a comprehensible statement. Showing them a sensitivity plot over log-odds hyperparameters is not.

Exact posterior sampling is available even when MCMC fails. The data augmentation algorithm is exact and fast, and handles prior-data conflict cases where gradient-based samplers get stuck.
The framework separates identified from partially-identified parameters explicitly. (θ0,θ1)(\theta_0, \theta_1)
(θ0​,θ1​) are identified; (ηe,ηs)(\eta_e, \eta_s)
(ηe​,ηs​) are not. Even if you don't care about the causal interpretation of ηe\eta_e
ηe​ and ηs\eta_s
ηs​, they provide a principled route to a joint prior on the quantities you do care about.

The main limitation is that the framework applies to binary outcomes and binary treatment. Extensions to continuous outcomes, survival data, or multi-arm tests are not handled, though the authors note these as directions for future work.
The paper is by Nicholas Irons and Carlos Cinelli, both at the University of Washington Department of Statistics. The preprint is available on arXiv and the R implementation is straightforward given the closed-form expressions derived in the appendix.