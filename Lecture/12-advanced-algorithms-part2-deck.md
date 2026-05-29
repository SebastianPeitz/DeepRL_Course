---
subtitle:    Advanced Algorithms
chapter:     11
feedback:
  deck-id:  'deeprl-advanced-algorithms'
...


------------------------------------------------------------------------------

# Content

------------------------------------------------------------------------------

# Content

- The evolution of modern RL algorithms
- Part (II): Improving $Q$-learning
  - Deterministic policy gradient 
  - Deep deterministic policy gradient (DDPG)
  - TD3 
  - Soft Actor-Critic


# Where are we?

::: small
| Chapter | Topic                                                  |                            Content  |
| :--: | :-------------------------------------------------------- | :---------------------------------- |
|      | **Basics \& tabular methods**                             |                                     |
|   1-5  | Bandits, MDPs, Dynamic Programming, Monte Carlo, TD Learning |   RL basics in finite dimensions  |
|      | **Deep-learning-based methods**                           |        |
|   6  | Brief introduction to deep learning                       |    The basics for what comes next    |
|   7  | Value function approximation                              |    Value estimation with function approximation    | 
|   8  | Deep Q-learning                                           |   Q-learning with neural networks     | 
| 9    | Policy gradients                                          | Direct optimization of the policy      | 
|  10  | Actor-critic algorithms| Improved policy gradients via value functions | 
|  11  | Advanced algorithms (Part I): From policy gradient to PPO | The evolution of moderl RL algorithms | 
|  [12]{style="color: red;"}  | [Advanced algorithms (Part II): From $Q$-learning to Soft Actor-Critic]{style="color: red;"} | [The evolution of moderl RL algorithms]{style="color: red;"} | 
|      | **Advanced Topics**                                       |        |

Table: Lecture contents
:::

------------------------------------------------------------------------------

# The evolution of modern RL algorithms

------------------------------------------------------------------------------

# The evolution of modern RL algorithms

::: small
<!-- We have covered most of the foundations of RL methods. [Let's look at the **two main branches** we have seen so far.]{.fragment} -->

::: fragment
::: columns-5-5

::: platzhalter
## (I) Policy gradient methods

::: incremental
- Policy is explicitly parameterized and optimized directly: $$L(\phi)=\Expsub{r(\tau)}{\tau\sim p_\phi(\tau)}.$$
<!-- - $$\nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} A^\pi(s_t, a_t)}{\tau\sim p_\phi(\tau)}.$$ -->
- Gradient descent: $\phi \gets \phi  + \alpha \nabla_\phi L(\phi)$.
- Methods are typically **on-policy**.
- **Algorithms**: REINFORCE $\rightarrow$ Actor-Critic $\rightarrow$ Natural Policy Gradient $\rightarrow$ Trust-region policy optimization (TRPO) $\rightarrow$ Proximal policy optimization (PPO).
:::
:::

::: fragment
## (II) Value-based methods

::: incremental
- Approximation of the $Q$-function:\
$$Q^*(s,a) = r + \max_{a'\in\Ac}Q^*(s',a').$$
- Extract implicit policy: $\pi(s) = \arg\max_{a\in\Ac} Q^*(s,a)$.
- Typically **off-policy**.
- **Algorithms**: Q-learning $\rightarrow$ DQN $\rightarrow$ Deep deterministic policy gradient (DDPG) $\rightarrow$ TD3 $\rightarrow$ Soft actor-critic (SAC).
:::
:::

:::
:::


::: columns-5-5

::: fragment
### [Shortcomings]{style="color: red;"}

::: incremental
- [High variance gradients.]{style="color: red;"}
- [Unstable policy updates.]{style="color: red;"}
- [Catastrophic performance collapse.]{style="color: red;"}
:::
:::

::: fragment
### [Shortcomings]{style="color: red;"}

::: incremental
- [Instability from bootstrapping.]{style="color: red;"}
- [Overestimation bias.]{style="color: red;"}
- [Maximization over actions / continuous action spaces.]{style="color: red;"}
:::
:::
:::


::: columns-5-5
::: fragment
### [Improvement strategy]{style="color: blue;"}

- [How to safely update policies.]{style="color: blue;"}
:::

::: fragment
### [Improvement strategy]{style="color: blue;"}

::: incremental
- [Stabilizing $Q$-learning with function approximation.]{style="color: blue;"}
- [Solving the continuous argmax problem.]{style="color: blue;"}
:::
:::
:::

:::

------------------------------------------------------------------------------

# Part (II): Improving $Q$-learning

------------------------------------------------------------------------------

# Deterministic policy gradients

::: small

:::

# Deep deterministic policy gradient (DDPG)

::: small

:::

# TD3

::: small

:::

# Soft actor-critic (SAC)

::: small

:::





# References

::: { #refs }
:::
