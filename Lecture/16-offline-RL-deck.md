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

- On-policy vs. off-policy vs. offline RL
  - Distributional shifts
  - Overestimation bias
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



# Distributional shifts

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
![Inspired by [@Levine2020offlinerltutorial].](images/16-offline-RL/Concept-wo-sampling.svg){width=320px .embed}


::: fragment
$$\text{RL objective:}\quad \max_\phi \sum_{t=0}^T \Expsub{\gamma^t r_t}{s\sim\rho_\phi, a\sim \pi_\phi\agivenb{\cdot}{s}}$$

::: incremental
- "Model" is valid under $\pi_\beta$...
- **not** under $\pi_\phi$!
  - "Model" could be the $Q$-function $Q_{\pi_\phi}$,
  - or the dynamics $p_\phi\agivenb{s'}{s,a}$.
:::
:::

:::
:::

:::

# Overestimation bias

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
![Inspired by [@Levine2020offlinerltutorial].](images/16-offline-RL/argmax-distribution.svg){width=300px .embed}
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



# Overestimation bias in RL

::: small

:::

# Algorithmic evolution of methods

::: small
Algorithmic Evolution: 
- Constraint (Don't go over there!) 
- Pessimism (Go anywhere, but assume what you haven't seen is dangerous) 
- Implicit/Sequence (Don't even look at things you haven't seen).
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
