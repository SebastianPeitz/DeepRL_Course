---
subtitle:    Temporal Difference Learning \& Q Learning
chapter:     5
feedback:
  deck-id:  'deeprl-td-learning'
...
------------------------------------------------------------------------------

# Content

------------------------------------------------------------------------------

# Content

- Monte Carlo simulation
- MC prediction
- MC esitmation of action values
- MC control
- On- and off-policy learning
- Off-policy MC control
- Summary


# Content
- Recall GPI
- GLIE
- SARSA (on-policy)
- Q-learning (off-policy)

# Greedy in the limit with infinite exploration (GLIE)

A learning policy $\pi$ is called GLIE if it satisfies the following two properties:

- If a state is visited inifinitely often, then each action is choosen inifintely often:
$$ \lim_{i \to \infty} \pi_i(a|s) = 1 \quad \forall  \{s \in \Sc, a \in \Ac\}$$

- In the limit ($i \to \infty$) the learning policy is greedy with respect to the learned action value:
$$ \lim_{i \to \infty} \pi_i(a|s) = \pi(s) = \arg \max_{a \in \Ac} Q(s,a) \quad \forall s \in \Sc $$

# GLIE Monte Carlo control

MC-based contro using $\epsilon$-greedy exploration is GLIE, if $\epsilon$ is decreased at a rate 
$$ \epsilon_i = \frac{1}{i} $$
with $i$ being the increasing episode index. In this case,
$$ \hat Q(s,a) = Q^*(s,a) $$
follows.

# SARSA
- **S**tate-**A**ction-**R**eward-Next-**S**tate-Next-**A**ction
- on-policy

::: fragment
::: {.definition}
### Algorithm: SARSA.

**initialize**

- $Q(s,a)$ arbitrarily for $s \in \Sc, a \in \Ac$ 
- $Q($terminal-state$,\cdot) = 0$
- $\pi = \epsilon$-greedy$(Q)$

**for** $j = 1, 2, \ldots, J$ episodes:\
$\quad$ Initialize $s_t \gets s_0$, $t \gets 0$\
$\quad$ **while** $s_t$ is not terminal:\
$\quad\quad$ Take action $a_t \sim \pi(s_t)$ and observe $(r_t,s_{t+1})$\
$\quad\quad$ Select $a_{t+1} \sim \pi(s_{t+1})$\
$\quad\quad$ Update $Q$ given $(s_t,a_t,r_t,s_{t+1},a_{t+1})$:
$\quad$ $$Q(s_t,a_t) \gets Q(s_t,a_t) + \alpha \left[r_t + \gamma Q(s_{t+1},a_{t+1})- Q(s_t,a_t)\right]$$
$\quad\quad$ Update policy $\pi = \epsilon$-greedy$(Q)$\
$\quad\quad$ $t \gets t+1$\
:::
:::

# Convergence of SARSA

[Based on Marius Lindauer's lecture]

SARSA for finite-state and finite-action MDPs converges to the optimal action-value, $Q(s, a) \to  Q^*(s, a)$, under the following conditions:

1. The policy sequence $\pi_t(a \mid s)$ satisfies the condition of GLIE
2. The step-sizes $\alpha_t$ satisfy the Robbins-Munro sequence such that 
  $$ 
    \sum_{t=1}^{\infty} \alpha_t = \infty \nonumber \\
    \sum_{t=1}^{\infty} \alpha^2_t < \infty \nonumber
  $$
	
For example, $\alpha_t = \frac{1}{t}$ satisfies the above condition.

# Q-Learning
- SARSA estimates $Q$ of the current policy
- Q-learning is similar but directly estimates $Q^*$
- off-policy update, since the optimal action-value function is updated independent of a given behavior policy

::: fragment
::: {.definition}
### Algorithm: Q-Learning.

**initialize**

- $Q(s,a)$ arbitrarily for $s \in \Sc, a \in \Ac$ 
- $Q($terminal-state$,\cdot) = 0$
- $\pi = \epsilon$-greedy$(Q)$

**for** $j = 1, 2, \ldots, J$ episodes:\
$\quad$ Initialize $s_t \gets s_0$, $t \gets 0$\
$\quad$ **while** $s_t$ is not terminal:\
$\quad\quad$ Take action $a_t \sim \pi(s_t)$ and observe $(r_t,s_{t+1})$\
$\quad\quad$ Select $a_{t+1} \sim \pi(s_{t+1})$\
$\quad\quad$ Update $Q$ given $(s_t,a_t,r_t,s_{t+1},a_{t+1})$:
$\quad$ $$Q(s_t,a_t) \gets Q(s_t,a_t) + \alpha \left[r_t + \gamma \textcolor{red}{\max_a Q(s_{t+1},a)}- Q(s_t,a_t)\right]$$
$\quad\quad$ Update policy $\pi = \epsilon$-greedy$(Q)$\
$\quad\quad$ $t \gets t+1$\
:::
:::


# References

::: { #refs }
:::
