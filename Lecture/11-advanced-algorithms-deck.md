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

- TRPO / PPO
- DDPG / TD3 / SAC
- Dueling DQN?


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
|  [11]{style="color: red;"}  | [Advanced algorithms]{style="color: red;"} |        | 
|      | **Model-Based Control**                                   |        |
|      | **Advanced Topics**                                       |        |

Table: Lecture contents
:::

------------------------------------------------------------------------------

# Natural / covariant policy gradients (TRPO)

------------------------------------------------------------------------------

# Ill-conditioned gradients

::: small
::: columns-5-5
::: platzhalter
![Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/).](images/11-advanced/PG-example-covariant.svg){ .embed width=600px }

[$$\begin{align*} 
r_t &= -s_t^2 - a_t^2 \\
\log\pi_\phi\agivenb{a_t}{s_t} &= -\frac{1}{2\sigma^2}(k s_t - a_t)^2 + C, \fragment{ \qquad \phi=(k,\sigma). } \\
\nablaphi\log\pi_\phi\agivenb{a_t}{s_t} &= \begin{pmatrix} -\frac{(k s_t - a_t)s_t}{\sigma^2} \\ \frac{(k s_t - a_t)^2}{\sigma^3} \end{pmatrix}
\end{align*}$$]{.math-incremental}
:::

::: fragment
![Source: [@Peters2008naturalac]](images/11-advanced/Covariant-PG-vanilla.png){ width=400px }
:::

:::

::: columns-5-5
::: incremental
- **Optimum**: $\phi^*=(-1,0)$.
- **Issue**: The gradients do not point towards the optimum for smaller $\sigma$!
  - The $\sigma^{-3}$ term blows up.
:::

::: fragment
Closely related to ill-conditioned optimization problems:
![](images/09-policy-gradients/zig-zag.png){ width=600px }
:::
:::

:::


# References

::: { #refs }
:::
