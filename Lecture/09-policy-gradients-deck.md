---
subtitle:    Policy Gradients
chapter:     8
feedback:
  deck-id:  'deeprl-policy-gradients'
...


------------------------------------------------------------------------------

# Content

------------------------------------------------------------------------------

# Content

- From value-based methods to direct policy optimization

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
| [9]{style="color: red;"}  | [Policy gradients]{style="color: red;"} | [Direct optimization of the policy]{style="color: red;"}       | 
|  10  | Actor-critic algorithms                                   |        | 
|  11  | Advanced algorithms                                       |        | 
|      | **Model-Based Control**                                   |        |
|      | **Advanced Topics**                                       |        |

Table: Lecture contents
:::

------------------------------------------------------------------------------

# From value-based methods to direct policy optimization

------------------------------------------------------------------------------

# From value-based methods to direct policy optimization

::: small
::: columns-7-3
::: platzhalter
::: incremental
- Until now, all approaches followed the GPI concept.
- These are all **value-based methods**: Need an estimate of $V^\pi$ or $Q^\pi$ to improve $\pi$.
- But: what are we after in RL, really? [$\Rightarrow$ The **optimal policy** $\pi^*$!]{.fragment}
[![Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/).](images/09-policy-gradients/CNN-policy.svg){ .embed width=900px }]{.fragment}
:::
:::

::: platzhalter
![Value-based methods: Generalized Policy Iteration (GPI) [@Sutton1998].](images/08-deep-q-learning/SuttonBarto_GPI.svg){ width=350px }
:::
:::

::: columns-3-7
::: platzhalter
[**Alternative approach:**]{.fragment}

::: incremental
1. Define probabilities over entire trajectories.
2. Formulate RL objective; optimize dynamics via policy parameters $\phi$.
:::
:::

::: platzhalter
[$$ \begin{align*} 
&\Rightarrow\quad p_\phi(\underbrace{s_0,a_0,\ldots,s_{T-1},a_{T-1},s_{T}}_{=\tau}) \fragment{= p_\phi(\tau)} \fragment{= p(s_0) \prod_{t=0}^{T-1} \pi_\phi\agivenb{a_t}{s_t} p\agivenb{s_{t+1}}{s_t,a_t}}. \\
&\Rightarrow\quad \phi^* = \arg\max_{\phi}\Expsub{\sum_{t=1}^{T-1}r_t}{\tau\sim p_\phi}.
\end{align*} $$]{.math-incremental}
:::
:::

:::

# The reinforcement learning objective
[$$ \begin{align*} 
\phi^* &= \arg\max_{\phi}\Expsub{\sum_{t=1}^{T-1}r_t}{\tau\sim p_\phi}\\
\text{Infinite horizon case:}\quad\phi^* &= \arg\max_{\phi}\Expsub{r}{(s,a)\sim p_\phi(s,a)}\\
\text{Finite horizon case:}\quad\phi^* &= \arg\max_{\phi}\sum_{t=1}^{T-1}\Expsub{r_t}{(s_t,a_t)\sim p_\phi(s_t,a_t)}
\end{align*} $$]{.math-incremental}

::: incremental
- Infinite horizon: expected reward according to the stationary distributions of $s$ and $a$.
- Finite horizon: linearity $\Rightarrow$ expected reward of individual stages (may differ with $t$).
- We are sticking to the finite-horizon case here.\
 
:::

::: fragment
::: footer
:bulb: We are not considering the discount factor $\gamma$ for now.
:::
:::

# Evaluating the objective


::: columns-6-4
::: small
$$ \phi^* = \arg\max_{\phi}\underbrace{\Expsub{\sum_{t=1}^{T-1}r_t}{\tau\sim p_\phi}}_{L(\phi)} $$

::: incremental
- First, let's think about evaluating the $L(\phi)$ for a fixed policy $\pi$.
- How do we estimate expectations if we don't have access to a model?
:::

[$\Rightarrow$ Monte Carlo sampling!]{.fragment}

[$$J(\phi) = \Expsub{\sum_{t=1}^{T-1}r_t}{\tau\sim p_\phi} \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=1}^{T-1}r_{i,t} $$]{.fragment}
:::

[![Taken from Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/).](images/09-policy-gradients/Samples-trajectory.png){ width=500px }]{.fragment}

:::

::: small
::: incremental
- Sample $N$ trajectories with $s_0$ according to the initial distribution and then following $\pi$.
- Take the sample average over these trajectories.
:::
:::

# Differentiating the policy

::: small
::: columns-6-4
$$ \begin{align*} \phi^* = \arg\max_{\phi}\underbrace{\Expsub{\sum_{t=1}^{T-1}r_t}{\tau\sim p_\phi}}_{L(\phi)}\\ 
L(\phi) = \Expsub{r(\tau)}{\tau\sim p_\phi(\tau)} = \int p_\phi(\tau) r(\tau) \dtau. \end{align*}$$

::: definition
### A convenient identity

$$
\fragment{ \textcolor{blue}{p_\phi(\tau) \nablaphi \log p_\phi(\tau)} } \fragment{ = p_\phi(\tau) \frac{\nablaphi p_\phi(\tau)}{p_\phi(\tau)} } \fragment{ = \textcolor{red}{\nablaphi p_\phi(\tau)} }
$$
:::
:::


:::

# References

::: { #refs }
:::
