---
subtitle:    Offline Reinforcement Learning
chapter:     16
feedback:
  deck-id:  'deeprl-offline-RL'
...


------------------------------------------------------------------------------

# Content

------------------------------------------------------------------------------

# Content

- Learning policies from datasets
- Offline reinforcement learning
  - Distributional shifts
  - Overestimation bias
  - Relation to imitation learning
  - Algorithmic evolution of methods
- Policy constraint methods
  - Forward and backward KL divergence
  - RL + BC (behavioral cloning)
  - advantage-weighted regression
  - Batch-constrained $Q$-learning
- 
  - Conservative $Q$-learning
- Implicit $Q$-learning
- Offline-to-online RL

# Where are we?

::: small
| Chapter | Topic                                                        |                                  Content |
| :-----: | :----------------------------------------------------------- | :--------------------------------------- |
|         | **Basics \& tabular methods**                                |                                          |
| 1-5     | Bandits, MDPs, Dynamic Programming, Monte Carlo, TD Learning | RL basics in finite dimensions           |
|         | **Deep-learning-based methods**                              |                                          |
| 6-13    | DQN, policy gradients, actor-critic, PPO/SAC, exploration    | Deep RL from basics to modern algorithms |
|         | **Model-Based Control**                                      |                                          |
|  14-15  | Optimal \& feedback control, MPC, planning, model-based RL   | Planning \& control when we know the model | 
|         | **Advanced Topics**                                          |                                          |
|   [16]{style="color: red;"}    | [Offline reinforcement learning]{style="color: red;"}                       | [RL with a static, given dataset]{style="color: red;"} |
|         | Imitation learning                                           |                                          |
|         | Transfer learning                                            |                                          |

Table: Lecture contents
:::


------------------------------------------------------------------------------

# Learning policies from datasets

------------------------------------------------------------------------------

# Imitation learning

::: small
::: incremental
- Two of the biggest challenges in RL: 
  1. Long waiting times until we find a policy that does not fail completely.
  1. Designing good reward functions that actually make the agent do what we want it to do.
- What can we do to circumvent this?\
$\Rightarrow$ "Let's just show the agent how it's done?": learn from an expert! 
- The concept behind this is **imitation learning**.
  - Given examples from a human expert, try to derive a policy that mimicks this behavior as closely as possible ...
  - ... while generalizing to previously unseen scenarios.
:::

::: fragment
![Source: [Medium.com](https://medium.com/@sebuzdugan/day-86-100-imitation-learning-teaching-agents-by-mimicking-experts-353ea29dea76).](images/16-offline-RL/Imitation-learning.png){height=300px}
![Source: [Innovation campus future mobility](https://www.icm-bw.de/en/news-and-events/news/newsdetail/machine-learning-by-human-hand).](images/16-offline-RL/imitation-learning-hand.jpg){height=300px}
:::

:::

# Some developments in imitation learning (1)

::: small
::: columns-5-5

::: platzhalter
::: definition
### Behavioral cloning (BC)

::: incremental
- Treats imitation as **supervised learning**. Inputs: expert states; Outputs/Labels: expert actions as labels
- Ignores the underlying MDP.
:::

[$\pluspoint$ [Simple to implement.]{style="color: green;"}]{.fragment}

[$\pluspoint$ [No environment interaction during training.]{style="color: green;"}]{.fragment}

[$\pluspoint$ [Very fast.]{style="color: green;"}]{.fragment}

[$\minuspoint$ [Compounding errors / covariate shift: A small error at step one puts the agent in a state it has never seen before, causing increasingly poor decisions.]{style="color: red;"}]{.fragment}

[$\minuspoint$ [Assumes i.i.d. data distribution, which is false in sequential environments.]{style="color: red;"}]{.fragment}

:::
:::

::: fragment
::: definition
### Inverse reinforcement learning (IRL)

::: incremental
- Instead of copying the actions, IRL tries to **figure out the intent** by assuming that the expert is optimizing an unobserved reward function.
- Extracts hidden reward function from the demonstrations and then runs standard RL on top.
:::

[$\pluspoint$ [Robust to compounding errors.]{style="color: green;"}]{.fragment}

[$\pluspoint$ [Reward function can transfer well even if the environment changes slightly.]{style="color: green;"}]{.fragment}

[$\minuspoint$ [Expensive inner-loop problem: RL training in each iteration.]{style="color: red;"}]{.fragment}

[$\minuspoint$ [Mathematically underdetermined (many different reward functions can explain the same expert behavior).]{style="color: red;"}]{.fragment}

:::
:::
:::
:::

# Some developments in imitation learning (2)

::: small
::: columns-4-6

::: platzhalter
::: definition
### Interactive / Human-in-the-Loop IL

::: incremental
- To fix BC's compounding errors without the cost of IRL, methods like *Dataset Aggregation* (DAgger, [@Ross2011dagger]) were introduced.
  - Runs current policy in the environment. 
  - When it makes a mistake and goes off-track, an expert intercepts and provides the correct action labels for those erroneous states. 
  - This data is added back into the training set.
:::

[$\pluspoint$ [Directly fixes covariate shift.]{style="color: green;"}]{.fragment}

[$\pluspoint$ [forces the agent to learn how to recover from its own mistakes.]{style="color: green;"}]{.fragment}

[$\minuspoint$ [Requires an expert to be constantly available and online during training to label data.]{style="color: red;"}]{.fragment}

[$\minuspoint$ [Exhausting for humans and frequently impossible for complex setups.]{style="color: red;"}]{.fragment}

:::
:::

::: fragment
::: definition
### Generative advarsarial imitation learning (GAIL) [@Ho2016gail]

::: columns-5-5

::: incremental
- Two-player game: 
  - Generator: the agent trying to mimic the expert policy.
  - Discriminator: classifier trying to distinguish between agent and expert trajectories. 
:::

::: platzhalter
![Inspired by Generative Adversarial Networks (GANs). ](images/16-offline-RL/GAN.png){width=430px}
:::

:::


::: incremental
- The discriminator penalizes the agent when it does not create expert-like trajectories.
:::

[$\pluspoint$ [Mimicks entire trajectories, not just single transitions as BC.]{style="color: green;"}]{.fragment}

[$\pluspoint$ [No expensive inner loop as in IRL.]{style="color: green;"}]{.fragment}

[$\pluspoint$ [Does not require an interactive expert like DAgger.]{style="color: green;"}]{.fragment}

[$\minuspoint$ [Inherits GAN instabilities (mode collapse, highly sensitive hyperparameters).]{style="color: red;"}]{.fragment}

[$\minuspoint$ [Very data hungry.]{style="color: red;"}]{.fragment}

:::
:::
:::
:::

# What if we want to do better than the data?

::: small

| Algorithm (class) | Key concept |
| :- | :- |
| Behavioral cloning (BC) | Copy data (i.e., "expert" as closely as possible).
| Inverse RL (IRL) | Identify reward from data $\Rightarrow$ then standard RL.
| Dataset aggregation (DAgger) | Expert corrects faulty behavior (meaning there is an online component).
| Generative advarsarial imitation learning (GAIL) | The generative version of BC mimicking trajectories.

Table: The key ideas in imitation learning

\

::: fragment
**BUT: What if we cannot get new data, but we still want to be better than the performance of the dataset?**
:::

[Maybe the data came from]{.fragment}

::: incremental
- Experts,
- semi-experts,
- one or more past RL policies,
- a mixture of the above?
:::

::: fragment
**$\Rightarrow$ Offline reinforcement learning!**
:::
:::


------------------------------------------------------------------------------

# Offline Reinforcement Learning

------------------------------------------------------------------------------

# What is offline RL?

::: small
![Inspired by [@Levine2020offlinerltutorial].](images/16-offline-RL/Concept.svg){width=1200px .embed}

::: fragment
### Offline RL is closely related to supervised learning / data-driven AI!
:::

::: columns-5-5

::: incremental
- Data-driven AI is about learning from large datasets.\
[$\textcolor{green}{\mathbf{+}\text{ Learns about the real world from data.}}$]{.fragment}\
[$\textcolor{red}{\mathbf{-}\text{ Doesn’t try to do better than the data.}}$]{.fragment}
:::

::: incremental
- Reinforcement learning is about optimization.\
[$\textcolor{green}{\mathbf{+}\text{ Optimizes a goal with emergent behavior.}}$]{.fragment}\
[$\textcolor{red}{\mathbf{-}\text{ Doesn’t make use of real-world data.}}$]{.fragment}
:::

:::

[$\Rightarrow$ Data without optimization doesn’t allow us to solve new problems in new ways.]{.fragment}\
[$\Rightarrow$ Optimization without data is hard to apply to the real world outside of simulators.]{.fragment}\

::: fragment
::: center
**Offline reinforcement learning**: Use *prior data* for RL training, *without any interactions* with the environment!\
[**Goal**: Given a dataset $\Dc$, learn the best possible policy $\piphi$ that is *supported by the dataset*.]{.fragment}
:::
:::
:::

# Can offline RL work?

::: small

::: incremental
1. **Find the "good stuff"** in a dataset full of good and bad behaviors.\
[**Example**, consider a dataset of various drivers:]{.fragment}\
[$\circ$ with imitation learning, we would find the average performance :thumbsdown:]{.fragment}\
[$\circ$ with offline RL, we hope to extract a policy mimicking the best driver's performance :thumbsup:]{.fragment}
:::


::: columns-3-1-1
::: incremental
2. **Generalization**: good behavior in one place may suggest good behavior in another place.\
[**Driving example**: we improve upon the best driver by generalizing to unseen situations.]{.fragment}
:::

::: fragment
![Source: [D4RL](https://sites.google.com/view/d4rl-anonymous/).](images/16-offline-RL/maze2D.gif){width=150px}
:::

::: fragment
![](images/16-offline-RL/maze2D-route.svg){width=150px}
:::
:::


::: columns-3-2
::: incremental
3. **“Stitching”**: parts of good behaviors can be recombined.
:::

::: fragment
![Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse/).](images/16-offline-RL/stitching.svg){width=400px .embed}
:::
:::

[**Takeaway**: It's not just imitation learning!]{.fragment}\
[**Intuition**: Get order from chaos.]{.fragment}

::: incremental
- On micro scales (e.g., a "straight path" from a bunch of wiggly trajectories).
- On macro scales (e.g., $A\to B \to C$ $\Rightarrow$ $A\to C$).
:::

:::

# Why is offline RL so hard?

::: small
**Fundamental problem**: counterfactual queries

::: columns-5-5-2

::: fragment
![**Training data.**](images/16-offline-RL/car-straight.png){width=500px}
:::

::: fragment
![**What the policy wants to explore.**](images/16-offline-RL/car-turn.png){width=500px}
:::

::: incremental
- Is this good or bad?
- How can we know we have never seen it?
:::

[Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse/).]{.footer}

:::

::: columns-5-5

::: incremental
- **Online RL** algorithms don’t have to handle this.
  - They can simply try this action and see what happens.
- **Offline RL** methods have to account for these unseen (*OOD*: "*out of distribution*") actions, 
  - ideally in a safe way ...
  - while still making use of generalization to come up with behaviors that are better than the best thing seen in the data!
:::

::: fragment
::: definition
### Example: Q-learning / Bellman update

$$ Q_\theta \gets r + \gamma \E_{s'\sim \psprimesa}\Big[\underbrace{\max_{a'\in\Ac} Q_{\bar{\theta}}(s',a')}_{\textbf{can go OOD}}\Big]  $$
:::
:::

:::

::: fragment
::: definition
### :bulb: *out of distribution* vs. *out of sample*.

::: incremental
- Out of sample $\Rightarrow$ generalization: Evaluation outside the training dataset, but on data **from the same distribution**.
- Out of distribution $\Rightarrow$ Evaluation on unseen **data from a different distribution**. [This is much harder!]{.fragment}
:::
:::
:::
:::



# Distributional shifts (1)

::: small
::: columns-5-1-5

![Inspired by [@Levine2020offlinerltutorial].](images/16-offline-RL/offline-RL.svg){width=500px}

::: platzhalter
 
:::

::: fragment
**Formally**:
[$$\begin{align*} 
\Dc &= \set{(s_i,a_i,s'_i,r_i)}_{i=1}^N \\
s &\sim \rho_\beta(s) \\
a &\sim \pi_\beta\agivenb{a}{s} \quad \fragment{ \text{(generally unknown)} } \\
s' &\sim \psprimesa
\end{align*}$$]{.math-incremental}
:::

:::

::: fragment
::: columns-5-5
![Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse/).](images/16-offline-RL/Concept-wo-sampling.svg){width=320px .embed}


::: fragment
$$\text{RL objective:}\quad \max_\phi \sum_{t=0}^T \Expsub{\gamma^t r_t}{s\sim\rho_\phi, a\sim \piphi\agivenb{\cdot}{s}}$$

::: incremental
- "Model" is valid under $\pi_\beta$...
- **not** under $\piphi$!
  - "Model" could be the $Q$-function $Q_{\piphi}$,
  - or the dynamics $p_\phi\agivenb{s'}{s,a}$.
:::
:::

:::
:::

:::

# Distributional shifts (2)

::: small
::: columns-6-4

::: platzhalter
### Example: Empirical risk minimization (maximum likelihood)

$$\theta = \arg\min_{\hat\theta} \Expsub{\cbracket{f_{\hat{\theta}}(x) - y}^2}{x\sim p(x), y\sim p\agivenb{y}{x}}$$

[**Question**: Given some $x^*$, is $f(x^*)$ correct?]{.fragment}


::: incremental
- $$\Expsub{\cbracket{f_{\hat{\theta}}(x) - y}^2}{x\sim p(x), y\sim p\agivenb{y}{x}} \quad \text{is low}$$
[$\Rightarrow$ probably yes.]{.fragment}
- $$\Expsub{\cbracket{f_{\hat{\theta}}(x) - y}^2}{x\sim \bar{p}(x), y\sim p\agivenb{y}{x}} \quad \text{is generally not, if} ~ \bar{p}(x) \neq p(x)$$
[$\Rightarrow$ probably no.]{.fragment}
- What if $x^* \sim p(x)$?
  - Not necessarily... 
  - but neural networks tend to generalize well!
:::

:::

::: platzhalter
::: fragment
::: definition

### What if we pick the maximizing $x^*$?

[$$ x^* = \arg\max_x f_\theta(x) $$]{.fragment}

::: fragment
![Inspired by [@Levine2020offlinerltutorial].](images/16-offline-RL/argmax-distribution.svg){width=400px .embed}
:::
:::
:::

::: fragment
:bulb: This is exactly what we're exploiting in adversarial learning!

![Adversarial example [@Goodfellow2015adversarial].](images/16-offline-RL/adversarial-panda.png){width=400px}

:::
:::
:::
:::

# Overestimation bias in offline RL

::: small
::: columns-6-4

::: platzhalter
### $Q$-learning and $Q$-function actor-critic

[$$\begin{align*} 
\phi &= \arg\min_{\hat{\theta}} \Expsub{\norm{Q_\theta(s,a)-y}^2}{s\sim\rho_\beta,a\sim \textcolor{red}{\pi_\beta\agivenb{\cdot}{s}}}, \\
y_i &= r_i + \gamma \Expsub{Q_\theta(s'_i,a')}{s'_i\sim\rho_\beta,a'\sim \textcolor{blue}{\piphi\agivenb{\cdot}{s'_i}}}
\end{align*}$$]{.math-incremental}

::: incremental
- Trying to find the best policy: **we want** $\textcolor{blue}{\piphi\agivenb{a}{s}} \neq \textcolor{red}{\pi_\beta\agivenb{a}{s}}$! 
- But we only have $s'_i$ for $s_i, a_i$ from our offline dataset.
- **Even worse**: We're actually picking an *adversarial example*, $$\phi = \arg\max_{\hat{\phi}} \Expsub{Q_\theta(s,a)}{a\sim \pi_{\hat\phi}\agivenb{\cdot}{s}}.$$
:::
:::

::: fragment
::: platzhalter
::: definition
### Example: SAC offline RL [@Kumar2019stabilizing]

::: columns-1-1

![How well the agent performs.](images/16-offline-RL/offline-RL-return.png){width=250px}

::: fragment
![How well it *thinks* it does.](images/16-offline-RL/offline-RL-Q.png){width=250px}
:::

:::

:::
:::
:::
:::


::: columns-6-4

::: platzhalter
::: fragment
### Model-based RL
:::

::: fragment
$$f = \arg\min_{\hat f} \Expsub{\norm{f(s,a)-s'}_2^2}{s\sim\rho_\beta,a\sim\textcolor{red}{\pi_\beta\agivenb{\cdot}{s}}, s'\sim \psprimesa} $$
:::

::: incremental
- Probably large: $\Expsub{\norm{f(s,a)-s'}_2^2}{s\sim\rho_\beta,a\sim\textcolor{blue}{\pi_\phi\agivenb{\cdot}{s}}, s'\sim \psprimesa}.$
- **Even worse**: Pick $\piphi$ to *maximize the reward* under $f$!
:::
:::

![Adversarial example [@Goodfellow2015adversarial].](images/16-offline-RL/adversarial-panda.png){width=400px}

:::
:::


# Algorithmic evolution of methods

::: small
The following strategies (or classes of approaches) are the most common ones today:

1. **Policy constraint methods**: Stay close to the data, e.g.,
$$ \KLdiv{\piphi\agivenb{\cdot}{s}}{\pi_\beta\agivenb{\cdot}{s}}\leq \epsilon. $$

::: fragment
2. **Value-regularization \& pessimism**: Go anywhere, but assume what you haven't seen is dangerous $\Rightarrow$ avoid overestimation.

![Inspired by [@Levine2020offlinerltutorial].](images/16-offline-RL/argmax-distribution.svg){width=400px}
:::

::: fragment
3. **Implicit $Q$-learning**: Avoid out-of-distribution actions in updates. 
:::
\

::: fragment
::: center
**Common theme: "Maximize return while staying close to the dataset.**
:::
:::
:::

------------------------------------------------------------------------------

# Policy constraint methods

------------------------------------------------------------------------------

------------------------------------------------------------------------------

# Value-regularization \& pessimism

------------------------------------------------------------------------------

------------------------------------------------------------------------------

# Implicit $Q$-learning {menu-title="Implicit Q-learning"}

------------------------------------------------------------------------------


------------------------------------------------------------------------------

# Offline-to-online RL

------------------------------------------------------------------------------

# A straightforward engineering approach

::: small
[@Ball2023offlineRL]
:::

------------------------------------------------------------------------------

# Benchmarking

------------------------------------------------------------------------------

# The D4RL benchmark

::: small
[https://sites.google.com/view/d4rl-anonymous/](https://sites.google.com/view/d4rl-anonymous/).
:::


# References

::: { #refs }
:::
