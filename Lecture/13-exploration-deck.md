---
subtitle:    Exploration
chapter:     13
feedback:
  deck-id:  'deeprl-exploration'
...


------------------------------------------------------------------------------

# Content

------------------------------------------------------------------------------

# Content

- Recap
  - Exploration-exploitation dilemma
  - Multi-armed bandits
- Advanced exploration methods for tabular approaches
  - Upper confidence bounds
  - Thompson sampling
  - Information gain
- Exploration in continuous spaces
  - The counting problem
  - Intrinsic motivation
  - Curiosity-driven exploration
  - Random network distillation

# Where are we?

::: small
| Chapter | Topic                                                  |                            Content  |
| :--: | :-------------------------------------------------------- | :---------------------------------- |
|      | **Basics \& tabular methods**                             |                                     |
|  1-5 | Bandits, MDPs, Dynamic Programming, Monte Carlo, TD Learning |   RL basics in finite dimensions  |
|      | **Deep-learning-based methods**                           |        |
|   6  | Brief introduction to deep learning                       |    The basics for what comes next    |
|   7  | Value function approximation                              |    Value estimation with function approximation    | 
|   8  | Deep Q-learning                                           |   Q-learning with neural networks     | 
|   9  | Policy gradients                                          | Direct optimization of the policy      | 
|  10  | Actor-critic algorithms| Improved policy gradients via value functions | 
|  11  | Advanced algorithms (Part I): From policy gradient to PPO | The PG route to modern RL algorithms | 
|  12  | Advanced algorithms (Part II): From $Q$-learning to Soft Actor-Critic | The AC route to modern RL algorithms| 
| [13]{style="color: red;"}  | [Exploration]{style="color: red;"}     | [How to explore in complex scenarios]{style="color: red;"} | 
|      | **Model-Based Control**                                   |        |
|      | **Advanced Topics**                                       |        |

Table: Lecture contents
:::
------------------------------------------------------------------------------

# Recap

------------------------------------------------------------------------------

# Why is exploration a problem?

*Sparse rewards and temporally extended tasks*

::: small

::: columns-5-4
::: platzhalter
In many reinforcement learning problems, useful behavior requires a long sequence of actions before any reward is observed.

$$ s_0 \rightarrow s_1 \rightarrow \cdots \rightarrow s_T, $$
$$ r_t = 0 \text{ for most } t, \qquad r_T > 0. $$

The agent receives little or no local signal telling it whether an intermediate action was useful.
:::

::: platzhalter
::: definition
### Canonical examples
- key--door navigation;
- Montezuma-style games;
- robotic manipulation from pixels;
- long-horizon planning with delayed success.
:::

[TODO: tikz diagram of the state chain $s_0 \xrightarrow{r=0} s_1 \xrightarrow{r=0} \cdots \xrightarrow{r>0} s_T$.]{style="color: red;"}
:::
:::
:::

# Exploration versus Exploitation

*The fundamental dilemma in online decision making*

::: small

At each decision point, the agent must choose between using what it currently knows and collecting information that may improve future decisions.

::: columns-5-5
::: definition
### Exploitation
Choose the action that appears best under the current value estimate:
$$ a_t = \arg\max_{a \in \Ac} \widehat Q_t(s_t,a). $$
Short-term objective: maximize known return.
:::

::: definition
### Exploration
Choose actions that may improve the agent's knowledge about the environment:\
$a_t$ chosen to reduce uncertainty or discover better policies.\
Long-term objective: learn enough to act better later.
:::
:::

[**Good exploration is not merely random action noise; it is purposeful information gathering.**]{style="color: red;"}
:::

# Examples of the same dilemma

*From everyday choices to reinforcement learning*

::: small

| Setting | Exploit | Explore |
| :-- | :-- | :-- |
| Restaurant selection | Go to your favourite restaurant | Try a new restaurant |
| Online ads | Show the best-known advert | Test a less-certain advert |
| Mineral mining | Drill at the best-known location | Drill in a new region |
| Reinforcement learning | Repeat known high-return behaviour | Try a policy that may reach a new high-reward region |

::: definition
### Core intuition
The best long-term strategy may require short-term sacrifices: actions that look suboptimal now may reveal information that enables much better behavior later.
:::
:::

# What does "optimal exploration" mean?

*Different evaluation criteria lead to different algorithms*

::: small

Exploration is hard partly because there is no single universal definition of optimality.

::: columns-5-5
::: definition
### Regret
Compare the learner to an optimal policy while learning:
$$ \mathrm{Reg}(K) = \sum_{k=1}^{K} \Bigl[ V^{\star}_1(s_{k,1}) - V^{\pi_k}_1(s_{k,1}) \Bigr]. $$
How much cumulative reward do we lose during learning?
:::

::: definition
### Sample complexity (PAC)
How many samples or episodes are needed before the agent can output a near-optimal policy?
$$ V^{\star}(s)-V^{\pi}(s) \leq \varepsilon $$
$$ \text{with probability at least }1-\delta . $$
This is about learning a good policy eventually, not necessarily maximizing reward during learning.
:::
:::

::: footer
Regret/PAC foundations: [@lai1985asymptotically; @auer2002finitetime; @jaksch2010near; @azar2017minimax].
:::
:::

# Multi-armed bandits

*A "simple" exploration problem: decision-making under partial feedback*

::: small

Each round the agent chooses an arm $a \in \Ac$, observes a reward $r(a)$, and updates its estimates. The feedback is *partial*: it sees only the reward of the arm it pulled, never the rewards of the arms it skipped. Each arm $i$ has an unknown mean reward
$$ \mu_i = \mathbb E[r_t \mid a_t=i], \qquad i\in\{1,\ldots,m\}. $$
In a Bayesian version, we place a prior on the unknown means; for Bernoulli rewards one common choice is
$$ \mu_i \sim \mathrm{Beta}(\alpha_i,\beta_i), \qquad r_t \mid a_t=i \sim \mathrm{Bernoulli}(\mu_i). $$

::: definition
### Pseudo-regret: loss relative to the best arm
Let $\mu^\star = \max_i \mu_i$. The expected pseudo-regret over $T$ rounds is
$$ \mathbb E[\mathrm{Reg}(T)] = \sum_{t=1}^{T} \mathbb E\bigl[\mu^\star-\mu_{a_t}\bigr]. $$
:::

::: footer
Bandit foundations: [@thompson1933likelihood; @lai1985asymptotically; @auer2002finitetime].
:::
:::

# Why naive exploration is not enough

*Greedy and random exploration both fail in different ways*

::: small

::: columns-5-5
::: definition
### Greedy selection
$$ a_t = \arg\max_{a \in \Ac} Q_t(a). $$

- efficient if the estimates are already correct;
- can lock onto a suboptimal action forever.
:::

::: definition
### $\varepsilon$-greedy selection
$$ a_t = \begin{cases} \arg\max_a Q_t(a), & 1-\varepsilon,\\ \text{random action}, & \varepsilon. \end{cases} $$

- simple and often useful;
- a *fixed* $\varepsilon$ never stops wasting pulls.
:::
:::

::: definition
### GLIE: Greedy in the Limit with Infinite Exploration
A policy is GLIE if (i) every action is tried infinitely often, and (ii) it becomes greedy in the limit. For example, use $\varepsilon$-greedy with $\varepsilon_t \downarrow 0$ but $\sum_t \varepsilon_t = \infty$. In tabular settings, GLIE is a standard condition for asymptotic convergence, but it is not a finite-time exploration strategy.
:::

::: footer
GLIE and convergence of on-policy RL: [@singh2000convergence].
:::
:::

------------------------------------------------------------------------------

# Advanced exploration methods for tabular approaches

------------------------------------------------------------------------------

# Three principles for exploration

::: small

Each principle needs two ingredients: an estimate of *uncertainty*, and an implicit *value placed on new information*. They differ in how they use them:

::: columns-1-1-1
::: definition
### Optimism (UCB)
Unknown may be good:
$$ a_t = \arg\max_a Q_t(a) + b_t(a). $$
Act on an optimistic bound until uncertainty shrinks.
:::

::: definition
### Posterior/Thompson sampling
Sample a plausible MDP:
$$ M_t \sim p(M \mid \Dc_t), \quad \pi_t \approx \pi^{\star}_{M_t}. $$
Act optimally in the sampled MDP.
:::

::: definition
### Information gain
Prefer informative actions:
$$ a_t \approx \arg\max_a \, \mathrm{IG}(z, y \mid a). $$
Reduce uncertainty about what matters.
:::
:::

[**Most modern exploration methods are scalable approximations of one of these.**]{style="color: red;"}

In deep RL: intrinsic bonuses $\approx$ optimism; ensembles / bootstrap / noisy nets $\approx$ posterior sampling; variational objectives $\approx$ information gain.

::: footer
Optimism: [@auer2002finitetime; @jaksch2010near]. Posterior sampling: [@osband2013more]. Information-directed sampling: [@russo2018ids].
:::
:::

# Upper confidence bounds

*Optimism in the face of uncertainty: act as if the world is as good as it plausibly could be*

::: small

Build a high-probability *upper confidence bound* on each arm's mean and play the largest. Maintain an estimate of the average reward $\widehat{\mu}_i$ for every action
$$ a_t = \arg\max_i \;\underbrace{\widehat{\mu}_i}_{\text{exploit}} + \underbrace{\sqrt{\tfrac{2\ln t}{N_i(t)}}}_{\text{explore}} . $$

::: columns-5-5
::: definition
### Why it works
An arm is chosen because its mean looks high *or* because it is under-sampled (large bonus). The bonus shrinks as $N_i(t)$ grows (the number of times action $a_i$ has been chosen), so uncertainty is resolved before it is trusted.
:::

::: definition
### Guarantee
For the bandit problem, UCB has logarithmic regret:
$$ \mathbb{E}[\mathrm{Reg}(T)] = \mathcal{O}(\log T), $$
where the hidden constants depend on the reward gaps. Thus average regret vanishes.
:::
:::

::: footer
UCB1 and its analysis: [@auer2002finitetime]. Optimism in MDPs: [@jaksch2010near].
:::
:::

# Thompson sampling

*Probability matching: explore in proportion to the probability of being optimal*

::: small

Maintain a posterior (belief) over each arm's mean. Each round, draw a sampled model from the belief, treat it as if it were the true model, and act optimally in it:
$$ \underbrace{\tilde{\mu}_i \sim \mathrm{Beta}(\alpha_i,\beta_i)}_{\text{1. sample a model}} \qquad \underbrace{a_t = \arg\max_i \tilde{\mu}_i}_{\text{2. act optimally in it}} \qquad \underbrace{(\alpha_{a_t},\beta_{a_t}) \mathrel{+}= (r_t,\,1-r_t)}_{\text{3. update the belief}} $$

Here a "model" is one plausible value $\tilde{\mu}=(\tilde{\mu}_1,\dots)$ for all the arm means; we commit to it for a single action, then revise the belief with the observed reward. (Bernoulli rewards with a Beta prior make the Bayesian update exact and cheap.)

::: columns-5-5
::: definition
### Why it works
Exploration comes from posterior *width*: uncertain arms occasionally draw high samples. As data accrues, posteriors concentrate and the policy turns exploitative on its own.
:::

::: definition
### Theory and practice
Under standard stochastic-bandit assumptions, Thompson sampling has logarithmic regret, and Chapelle \& Li showed it is highly competitive empirically --- a default baseline.
:::
:::

::: footer
Thompson sampling: [@thompson1933likelihood; @osband2013more]. Empirical study: [@chapelle2011empirical].
:::
:::

# Information gain

*Exploration as Bayesian experimental design*

::: small

Suppose we want to learn a latent quantity $z$, e.g. the optimal action $a^\star$, or its value. We maintain a prior belief $p(z)$ with entropy $H(z)$. Taking action $a$ yields an observation $y$ (for a bandit, $y = r(a)$), and the posterior $p(z \mid y)$ has entropy $H(z \mid y)$.

::: definition
### Information gain $=$ expected reduction in entropy
$$ \mathrm{IG}(z, y \mid a) = H(z) - \mathbb{E}_{y \mid a}\big[H(z \mid y)\big]. $$
Choosing $a$ to maximize $\mathrm{IG}(z, y \mid a)$ resolves uncertainty about $z$ as fast as possible.
:::

[**But pure information-seeking ignores reward --- it explores to *learn*, not to *earn*.**]{style="color: red;"}\
We need to trade information against regret (next slide).

::: footer
Expected information gain / Bayesian experimental design: [@lindley1956measure].
:::
:::

# Information-directed sampling (IDS)

*Accept regret only in proportion to the information it buys*

::: small

Greedy ignores information; pure information gain ignores reward. IDS balances them through the *information ratio*. For each action, define its expected one-step regret and its information gain about the optimal arm $a^\star$:
$$ \Delta_t(a) = \mathbb{E}\big[r(a^\star) - r(a) \mid \Dc_t\big], \qquad g_t(a) = \mathrm{IG}\big(a^\star,\, r(a) \mid a, \, \Dc_t\big). $$

::: definition
### Pick the action with the smallest information ratio
$$ a_t = \arg\min_{a}\; \Psi_t(a), \qquad \Psi_t(a) = \frac{\Delta_t(a)^{2}}{g_t(a)} . $$
Take an action only when its regret is small *or* it is highly informative.
:::

By valuing information explicitly, IDS can outperform UCB and Thompson when the most informative arm is not the most promising one.

::: footer
Information ratio: [@russo2018ids]. Deterministic IDS: [@kirschner2018information].
:::
:::

------------------------------------------------------------------------------

# Exploration in continuous spaces

------------------------------------------------------------------------------

# What changes beyond bandits?

*Actions affect both reward and future information*

::: small

In an MDP, actions do not only produce rewards; they also determine which future states and future information become accessible.
$$ s_{t+1} \sim p(\cdot \mid s_t,a_t), \qquad r_t = r(s_t,a_t). $$

::: columns-5-5
::: definition
### New difficulties
- rewards may be delayed;
- credit assignment is long-horizon;
- exploration is over trajectories, not isolated actions (e.g. Montezuma's Revenge, NetHack).
:::

::: definition
### Deep RL difficulties
- uncertainty may be over rewards, dynamics, values, and representations;
- in continuous spaces, state counts are not obvious;
:::
:::

::: footer
Contextual / procedurally-generated exploration: [@raileanu2020ride; @zhang2021noveld; @henaff2022exploration].
:::
:::

# Exploration bonuses in reinforcement learning

*A practical bridge from theory to scalable algorithms*

::: small

A common approach is to modify the reward with an intrinsic exploration bonus:
$$ r^+_t = r_t + \beta\, b_t(s_t,a_t,s_{t+1}), \qquad \beta > 0. $$

::: columns-5-5
::: definition
### Count or density bonuses
Reward rarely visited states:
$$ b_t(s) \approx \frac{1}{\sqrt{N_t(s)}}. $$
In large spaces, replace counts with density models or pseudo-counts.
:::

::: definition
### Curiosity bonuses
Reward surprising or informative transitions:\
$b_t(s,a,s') \approx$ prediction error / information gain.\
The bonus guides the policy toward useful unknowns.
:::
:::

::: footer
Counts/pseudo-counts: [@bellemare2016unifying; @tang2017exploration]. Curiosity / information gain: [@houthooft2016vime; @pathak2017curiosity; @burda2019exploration].
:::
:::

# The counting problem

*Optimistic exploration in RL: generalizing counts beyond tabular states*

::: small

In tabular RL, optimism can be implemented with a count-based bonus:
$$ r_t^{+} = r_t + \beta\, b_t(s_t), \qquad b_t(s) \approx \frac{1}{\sqrt{N_t(s)}} . $$

::: definition
### Problem
In high-dimensional or continuous spaces, exact counts are not useful:
$$ N_t(s) \approx 1 \quad \text{for almost every observed state.} $$
:::

::: definition
### Idea
Replace exact counts by a learned notion of familiarity:
$$ N_t(s) \quad \leadsto \quad \widehat N_t(s) \quad \text{from a density or novelty model.} $$
Nearby or similar states should share statistical strength.
:::

::: footer
Pseudo-count exploration: [@bellemare2016unifying].
:::
:::

# What can replace counts?

*Density, uncertainty, and novelty models*

::: small

The model does not need to generate perfect samples. For exploration, it mainly needs to answer:
$$ \text{``How familiar is this state or state-action pair?''} $$

::: columns-1-1-1
::: definition
### Density models
Estimate how likely a state is under past experience.

- CTS models
- PixelCNN
- hash-based counts
- compression length
:::

::: definition
### Uncertainty models
Estimate uncertainty about rewards, values, or dynamics.

- Gaussian processes
- Bayesian neural nets
- ensembles
- bootstrapped value functions
:::

::: definition
### Implicit novelty models
Detect whether a state is easy to distinguish from past data.

- exemplar models
- discriminators
- GAN-style density ratios
- random network distillation
:::
:::

::: footer
Examples: [@bellemare2016unifying; @tang2017exploration; @ostrovski2017count; @houthooft2016vime; @fu2017ex2; @burda2019exploration].
:::
:::

# UCB with Gaussian process regression

*Optimism from posterior uncertainty*

::: small

Suppose the unknown quantity is a smooth reward or value function
$$ f(x), \qquad x=(s,a) \quad \text{or simply an arm in a continuous bandit.} $$
After observing data $\Dc_t$, Gaussian process regression gives
$$ f(x)\mid \Dc_t \approx \mathcal N\!\left( m_t(x), \sigma_t^2(x) \right). $$

::: definition
### GP-UCB rule
Choose the point with the largest optimistic value:
$$ x_t = \arg\max_x \left[ m_t(x) + {\beta}\,\sigma_t(x) \right]. $$
:::

- $m_t(x)$: exploitation term.
- $\sigma_t(x)$: exploration bonus $\sim$ epistemic uncertainty.
- The kernel quantifies the similarity between state-action pairs.

::: footer
GP-UCB: [@srinivas2010gpucb]; RKHS bandits: [@chowdhury2017kernelized].
:::
:::

# Density modelling with exemplar models

*Novel states are easy to distinguish from past states*

::: small

Instead of explicitly modelling a density $p_\theta(s)$, ask a discriminative question:
$$ \text{Can a classifier distinguish this state from the replay buffer?} $$
For an observed exemplar $s_i$, train a classifier that separates
$$ s_i \quad \text{from} \quad \Dc_t. $$

::: definition
### Intuition
If a new state is very different from everything seen before, then it is easy to classify as its own exemplar.
$$ \text{easy to distinguish} \quad \Rightarrow \quad \text{novel} \quad \Rightarrow \quad \text{large exploration bonus}. $$
:::

This gives an implicit density estimate without requiring a high-quality generative model.

::: footer
EX2: Exploration with Exemplar Models: [@fu2017ex2].
:::
:::

# Exemplar models in practice

*Avoiding the "one classifier per state" problem*

::: small

A literal exemplar method sounds too expensive: one classifier for every visited state.

It also sounds trivial: in continuous spaces, every state is unique.

::: definition
### Why it still works
The classifier is regularized and trained on features, so it cannot simply test exact equality.
$$ s = s_i \quad \text{is not the point.} $$
The point is whether $s$ is distinguishable from past experience in a meaningful representation.
:::

::: definition
### Amortized version
Train one neural network that takes both inputs:
$$ D_\theta(s, s_i) = \text{probability that } s \text{ matches exemplar } s_i. $$
This single model amortizes the cost across many exemplars.
:::

::: footer
Discriminative novelty as implicit density estimation: [@fu2017ex2].
:::
:::

# Prediction gain and pseudo-counts

*If a state changes the density model a lot, it was novel*

::: small

Let $p_t(s)$ be the density model over visited states evaluated at state $s$ before training on it, and let $p_{t}'(s)$ be the density after updating the model with $s$.

::: definition
### Prediction gain
$$ \mathrm{PG}_t(s) = \log p_{t}'(s) - \log p_t(s). $$
:::

- If the density model changes a lot after seeing $s$, then $s$ was unfamiliar.
- Prediction gain can be converted into a pseudo-count $\widehat N_t(s)$.
- Then use a count-like exploration bonus:
$$ r_t^+ = r_t + \beta \frac{1}{\sqrt{\widehat N_t(s_t)}}, \qquad \text{where} \qquad \widehat N_t(s) = \frac{ p_t(s)\bigl(1-p_{t}'(s)\bigr) }{ p_{t}'(s)-p_{t}(s) }. $$

::: footer
Prediction gain and pseudo-counts: [@bellemare2016unifying].
:::
:::

# Intrinsic motivation

*Posterior sampling in RL: sample a plausible value function and commit to it*

::: small

Classically, posterior sampling is done over a Bayesian object, which is usually an MDP. In bandits, Thompson sampling means sampling plausible mean rewards. In value-based deep RL, the practical analogue is to sample a value function.

::: definition
### Simple idea
At the start of an episode:
$$ Q_t \sim p(Q \mid \Dc_t), \qquad \pi_t(s) = \arg\max_a Q_t(s,a). $$
Follow this policy for the whole episode, collect data, then update the posterior.
:::

- Randomness is in the value function, not in every action.
- The sampled policy is internally consistent across time.
- This produces temporally extended exploration.

::: footer
Posterior sampling in RL: [@osband2013more; @osband2017randomized].
:::
:::

# Posterior sampling with Gaussian processes

*A clean function-space version*

::: small

A Gaussian process gives a distribution over functions:
$$ Q(x) \sim \mathcal{GP}(m(x), k(x,x')), \qquad x=(s,a). $$
After observing data $\Dc_t$, GP regression gives a posterior:
$$ Q \mid \Dc_t \sim \mathcal{GP}(m_t,k_t). $$

::: definition
### GP Thompson sampling
Sample one function from the posterior:
$$ Q_t \sim \mathcal{GP}(m_t,k_t), $$
then act greedily with respect to that sampled function:
$$ a_t = \arg\max_a Q_t(s_t,a). $$
:::

::: footer
Gaussian processes: [@rasmussen2006gaussian]. GP Thompson sampling: [@bijl2016smcgpThompson].
:::
:::

# Bootstrapped DQN

*A practical approximation to posterior sampling*

::: small

Exact Bayesian inference over deep Q-functions is usually intractable. Bootstrapped DQN approximates this with an ensemble of Q-functions:
$$ Q_{\theta_1}, Q_{\theta_2}, \ldots, Q_{\theta_M}. $$

::: definition
### Episode-level exploration
At the start of each episode, sample one head:
$$ m \sim \mathrm{Uniform}\{1,\ldots,M\}, \qquad a_t = \arg\max_a Q_{\theta_m}(s_t,a). $$
Use the same head for the whole episode.
:::

- Instead of many separate networks, use one shared network with $M$ output heads.
- Different heads represent different plausible Q-functions.
- Random Q-functions give coherent exploration; random actions only dither locally.

::: footer
Bootstrapped DQN: [@osband2016bootstrapped]. Randomized prior functions: [@osband2018randomized].
:::
:::

# Information gain in reinforcement learning

*What should the agent try to learn?*

::: small

In principle, exploration can be driven by information gain:
$$ \mathrm{IG} = H(\text{belief before}) - \mathbb E[ H(\text{belief after}) ]. $$

- **Reward information:** useful if rewards are informative, but weak when rewards are sparse.
- **State-density information:** useful for novelty and state coverage.
- **Dynamics information:** useful for learning the MDP model.

::: definition
### Main difficulty
Exact information gain is usually intractable in large MDPs. So deep RL methods use approximations: density change, model error, model uncertainty, or parameter change.
:::
:::

# VIME: information gain as a KL divergence

*From entropy reduction to a variational objective*

::: small

The information gain about the dynamics parameters $\theta$ from a transition is the expected reduction in posterior entropy, which rewrites as an expected KL between the new and old belief:
$$ \mathrm{IG} = H(\theta\mid\Dc_t) - \mathbb{E}_{s_{t+1}}\!\big[ H(\theta\mid\Dc_t, a_t, s_{t+1}) \big] = \mathbb{E}_{s_{t+1}}\!\Big[ D_{\mathrm{KL}}\big( p(\theta\mid\Dc_t, a_t, s_{t+1}) \,\|\, p(\theta\mid\Dc_t) \big) \Big]. $$

The posterior is intractable, so approximate it by a variational $q(\theta;\phi)$ that maximizes the variational lower bound (ELBO):
$$ \max_{\phi}\; \mathbb{E}_{q(\theta;\phi)}\!\big[\log p(\Dc_t\mid\theta)\big] - D_{\mathrm{KL}}\big( q(\theta;\phi)\,\|\,p(\theta) \big). $$

::: definition
### Weight uncertainty: a fully factorized Gaussian
Represent $q$ as independent Gaussians, one per network weight (a mean and a variance) --- Bayes by Backprop:
$$ q(\theta;\phi) = \prod_i \mathcal N\!\big(\theta_i;\, \mu_i, \sigma_i^2 \big), \qquad \phi = (\mu,\sigma). $$
:::

::: footer
VIME: [@houthooft2016vime]. Weight uncertainty / Bayes by Backprop: [@blundell2015weight].
:::
:::

# VIME: a KL bonus from updated weight uncertainty

*An approximate information-gain reward*

::: small

Given a new transition $(s_t,a_t,s_{t+1})$, update the variational parameters by a step of the variational objective on that datapoint --- i.e. update the weight means and variances,
$$ \phi = (\mu,\sigma) \;\longrightarrow\; \phi' = (\mu',\sigma'). $$

::: definition
### Intrinsic reward $=$ change in belief
Use the KL between the updated and previous belief as an approximate information-gain bonus (closed-form for factorized Gaussians):
$$ r_t^{+} = r_t + \beta\, D_{\mathrm{KL}}\big( q(\theta;\phi') \,\|\, q(\theta;\phi) \big). $$
:::

[**Trade-off:**]{style="color: red;"} as an approximate information-gain method, VIME has an appealing mathematical formalism, but the models (Bayesian neural networks) are more complex and generally harder to use effectively than simpler proxies.

::: footer
VIME: [@houthooft2016vime].
:::
:::

# Curiosity-driven exploration

*Simpler proxy: prediction error --- reward the agent for visiting states the model cannot yet predict*

::: small

Instead of estimating information gain exactly, learn a forward model and reward states it predicts poorly.

Encode observations with an autoencoder, and predict the next encoding from the current one and the action:
$$ z_t = e(o_t), \qquad \widehat z_{t+1} = f_\theta(z_t,a_t). $$
The realized next encoding is $z_{t+1} = e(o_{t+1})$.

::: definition
### Exploration bonus: how unfamiliar was the transition?
$$ b_t = \bigl\| z_{t+1} - \widehat z_{t+1} \bigr\|^2 . $$
:::

Intuition: a large error means the model has not yet learned this part of the environment.

[**Caution:** pure prediction error is attracted to inherently noisy states.]{style="color: red;"}

::: footer
Prediction-error exploration: [@stadie2015incentivizing; @pathak2017curiosity].
:::
:::

# Simpler proxy: learning progress

*Reward improvement, not just change*

::: small

Information gain measures how much the *belief* changes. A first proxy is how much the learned *model* changes after a transition:
$$ b_t \propto \|\theta_{t+1}-\theta_t\|^2 . $$
But raw change can still be driven by noise --- the model keeps moving without ever getting better.

::: definition
### Learning progress: reward the model getting *better*
Measure the drop in prediction error from updating on the transition:
$$ b_t \propto \underbrace{\|z_{t+1}-\widehat z_{t+1}^{\,\text{old}}\|^2}_{\text{error before update}} - \underbrace{\|z_{t+1}-\widehat z_{t+1}^{\,\text{new}}\|^2}_{\text{error after update}} . $$
:::

You cannot make progress on unlearnable noise, so learning progress avoids the noisy-state trap of pure prediction error (previous slide).

::: footer
Artificial curiosity and learning progress: [@schmidhuber1991curiosity; @schmidhuber2010formal].
:::
:::

# Random network distillation

*Heuristic novelty: prediction errors as counts --- we only need to tell novel from familiar*

::: small

A model $p_\theta(s)$ used for exploration need not generate good samples --- or even output accurate densities. It only needs to tell whether a state is *novel*.

::: definition
### General recipe: fit a target, reward the error
Pick a target function $f^\star(s,a)$; fit a model $f_{\theta_t}(s,a)$ on the buffer $\Dc_t$; reward large error:
$$ b_t(s,a) = \big\| f_{\theta_t}(s,a) - f^\star(s,a) \big\|^2 . $$
Inputs unlike the training data are predicted poorly $\Rightarrow$ treated as novel.
:::

A natural choice is $f^\star = s_{t+1}$, the next state: then this is forward-model prediction error, a proxy for information gain about the dynamics. This inherits the [**noisy-TV problem**]{style="color: red;"} --- a stochastic next state keeps the error high forever. Can we choose $f^\star$ so that novelty is decoupled from stochasticity?

::: footer
Exploration by random network distillation: [@burda2019exploration].
:::
:::

# Random network distillation (RND)

*Prediction error as a novelty (optimism / count) bonus*

::: small

Fix a *randomly initialized* vector-valued target network $f^\star$ that maps a state to a feature vector, and train a predictor $f_{\theta_t}$ to match it on visited states:
$$ f_{\theta_t} \approx f^\star, \qquad b_t(s) = \big\| f_{\theta_t}(s) - f^\star(s) \big\|^2 . $$

::: columns-5-5
::: definition
### What the error measures
The predictor fits states seen often, so error is small there and large on novel states. It behaves like a novelty bonus --- roughly, familiar states have small error and novel states have large error.
:::

::: definition
### Why a random target
The target is a fixed deterministic function of the current state, so the error reflects whether the predictor has already learned that state.
:::
:::

With a simple setup, RND was the first method to exceed average human performance on Montezuma's Revenge without demonstrations.

::: footer
Exploration by random network distillation: [@burda2019exploration].
:::
:::

------------------------------------------------------------------------------

# Outlook: task-driven and task-agnostic exploration

------------------------------------------------------------------------------

# Task-driven exploration

*Exploration for a known reward*

::: small

::: definition
### Intuition
The reward function is already specified. The agent explores because it may help obtain more reward later.
:::

- The agent has a concrete task to solve.
- Exploration is judged by improvement on that task (regret).
- The core trade-off is:
$$ \text{exploit known reward} \quad \text{vs.} \quad \text{explore for better future reward}. $$
- Once the task is solved, exploration should usually decrease.

**Examples:** UCB, posterior sampling, information gain, intrinsic bonuses.
:::

# Task-agnostic exploration - Part 2

*Exploration before the reward is known*

::: small

::: definition
### Intuition
No downstream reward is given yet. The agent explores to learn what is possible in the environment.
:::

- The goal is broad experience, not one specific task.
- The agent may learn useful states, goals, skills, or representations.
- Good exploration means covering meaningful parts of the environment.
- Later, this experience should help solve many different tasks.

**Examples:** state coverage, state marginal matching, goal-conditioned practice, skill discovery.
:::

# Suggested reading

::: small
- **Curiosity and learning progress:** [@schmidhuber1991curiosity; @schmidhuber2010formal]
- **Prediction-error bonuses in deep RL:** [@stadie2015incentivizing]
- **Count-based exploration and pseudo-counts:** [@bellemare2016unifying]
- **Information gain about dynamics:** [@houthooft2016vime]
- **Posterior sampling / deep exploration:** [@osband2016bootstrapped]
- **Gaussian-process dynamic programming with UCB:** [@deisenroth2009gpdp]
:::

# References

::: small
::: { #refs }
:::
:::
