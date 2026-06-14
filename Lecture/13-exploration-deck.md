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

# Exploration-exploitation dilemma

::: columns-7-3
![Google Maps (go [here](https://maps.app.goo.gl/ixKDSXT6bwzfa6Y9A) for the live version; do you get the same numbers?)](images/01-multi-armed-bandits/MapsDortmund.png){ width=900px }

::: small
::: incremental
- Let's assume we do not have access to travel time estimates
- Which route should I take to minimize my travel time?
- Let's say we can guess the time of one route fairly well
  - should we always take this one?
  - or try something else and see if we can get better?
- This is known as the **exploration-exploitation dilemma**
- The route pickig problem is one example of a **multi-armed bandit**
:::
:::

:::

# Multi-armed bandits

::: columns-7-3

::: small
::: incremental
- Let us assume that we have a slot machine and we repeatedly can choose between $k$ different actions
- After each choice $a_t$ you receive a numerical reward $r_t$ chosen from a stationary probability distribution$^*$
- Objective: maximize the **expected total reward** over some time period (e.g., over 1000 action selections, or *time steps*)
- We refer to this as the **value**: $$ q(a) = \ExpC{r_t}{a_t=a} $$
- If we knew $q(a)$, then it would be easy to choose!
- Instead, we have to rely on estimates $Q_t(a)$ which we can iteratively update based on past experience
:::
:::

![A multi-armed bandit [@Ferreira2024mab]](images/01-multi-armed-bandits/multi-armed-bandit.png){ height=150px }
:::

# A simple bandit algorithm

::: columns-4-6
::: platzhalter
``` python
# Initialization
for i in range(k):
  Q(a) = 0
  N(a) = 0

# Run forever
while(True):

  # exploration versus exploitation
  if rand() > epsilon:
    a = argmax(Q)        # exploitation
  else:
    a = randint(k)       # exploration

  r = bandit(a)

  # Error correction towards target r 
  N(a) = N(a) + 1
  Q(a) = Q(a) + (1/N(a)) * (r - Q(a))
```
Caption: A simple version of the $\epsilon$ greedy bandit.

::: small

::: incremental
- $\epsilon$-greey is a very simple approach to achieve (local) exploration! 
- Another approach: **optimistic initial values**.
  - For $q(a) \sim \Normal{1}{0}$, an initial guess of $Q_1(a)=5$ is certainly unrealistically high.
:::
:::

:::

::: platzhalter
::: fragment
![](images/01-multi-armed-bandits/mab-rewards.svg){ width=600px }
:::

::: fragment
![](images/01-multi-armed-bandits/mab-actions-opt.svg){ .embed width=600px }
:::
:::
:::



------------------------------------------------------------------------------

# Advanced exploration methods for tabular approaches

------------------------------------------------------------------------------

# Upper confidence bounds

::: small

:::

# Thompson sampling

::: small

:::


# Information gain

::: small

:::


------------------------------------------------------------------------------

# Exploration in continuous spaces

------------------------------------------------------------------------------

# The counting problem

::: small

:::


# Intrinsic motivation

::: small

:::

# Curiosity-driven exploration

::: small

:::

# Random network distillation

::: small

:::


# References

::: { #refs }
:::
