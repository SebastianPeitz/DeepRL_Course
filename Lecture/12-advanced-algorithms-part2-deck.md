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
|  11  | Advanced algorithms (Part I): From policy gradient to PPO | The PG route to modern RL algorithms | 
|  [12]{style="color: red;"}  | [Advanced algorithms (Part II): From $Q$-learning to Soft Actor-Critic]{style="color: red;"} | [The AC route to modern RL algorithms]{style="color: red;"} | 
|      | **Model-Based Control**                                   |        |
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

# Recall: Policy gradient and actor-critic

::: small
::: definition
### Policy gradient theorem -- formulation via reward trajectories ([sampling version in blue]{style="color: blue;"})
$$ \nablaphi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log\piphi\agivenb{a_t}{s_t}\cbracket{\sum_{t'=t}^{T-1}r_{t'}}}{\tau\sim p_\phi(\tau)} \approx \textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t'=t}^{T-1} \nablaphi \log\,\piphi\agivenb{a_{i,t}}{s_{i,t}}\cbracket{\sum_{t'=t}^{T-1} r_{i,t'} }}}. $$
:::

::: definition
### Policy gradient theorem -- formulation using the $Q$-function and the Actor-Critic architecture
<!-- $$ \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} \Qpiphi(s_t, a_t)}{\tau\sim p_\phi(\tau)} \fragment{ \approx\textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\,\piphi\agivenb{a_{i,t}}{s_{i,t}} \Qpiphi(s_{i,t},a_{i,t})} }. } $$ -->
$$\nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} A_\theta(s_t, a_t)}{\tau\sim p_\phi(\tau)} \approx \textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\,\piphi\agivenb{a_{i,t}}{s_{i,t}} A_\theta(s_{i,t},a_{i,t})} }. $$
:::

### Main drawbacks

::: incremental
1. **High-variance** gradient.
1. **Inefficient for continuous actions**, since sampling in high-dimensional settings becomes inefficient.
1. **On-policy** $\Rightarrow$ sample-inefficient (importance sampling makes problem 1. even worse!)
:::

[**Approach**: Find a more sample-efficient and off-policy capable version]{.fragment} [$\Rightarrow$ deterministic policy!]{.fragment}

:::


# Off-policy policy gradients (1)

::: small
::: incremental
- [@Degris2012offpac] presented an alternative form for the off-policy policy gradient (for finite $\Sc$ and $\Ac$) using an approximation. [The starting point is the definition of the value function:
$$\Vpiphi = \sum_{s\in\Ac} \rho_\beta(s) \sum_{a\in\Ac} \piphi\agivenb{a}{s} \Qpiphi(s,a),$$
where $\beta$ is some behavior policy and $\rho_\beta(s) = \lim_{t\to\infty}p\agivenb{s}{s_0,\beta}$ is the limiting distribution of states when following $\beta$.]{.fragment}
- Then, taking the gradient of $\Vpiphi$ with respect to $\phi$, we get (via the product rule of differentiation):
$$ \nablaphi\Vpiphi = \nablaphi \rbracket{\sum_{s\in\Ac} \rho_\beta(s) \sum_{a\in\Ac} \piphi\agivenb{a}{s} \Qpiphi(s,a)} \fragment{ = \sum_{s\in\Ac} \rho_\beta(s) \sum_{a\in\Ac} \rbracket{\nablaphi\piphi\agivenb{a}{s} \Qpiphi(s,a) + \piphi\agivenb{a}{s} \nablaphi\Qpiphi(s,a)}. } $$
- Ignoring the second part (i.e., $\cancel{\piphi\agivenb{a}{s} \nablaphi\Qpiphi(s,a)}$; justification details in [@Degris2012offpac]), we get 
$$ \nablaphi\Vpiphi \approx \sum_{s\in\Ac} \rho_\beta(s) \sum_{a\in\Ac} \nablaphi\piphi\agivenb{a}{s} \Qpiphi(s,a) \fragment{ = \Expsub{\sum_{a\in\Ac} \nablaphi\piphi\agivenb{a}{s} \Qpiphi(s,a)}{s\sim\rho_\beta}. } $$
<!-- - [@Silver2014dpg]  -->
:::
:::

::: fragment
::: footer
:bulb: In contrast to the on-policy case (see the Actor-Critic lecture), here it is not possible to formulate a recursive formula which would allow us to find a closed-form statement for $\nablaphi\Qpiphi(s,a)$ via $\nablaphi\Vpiphi(s)$. This only works in the on-polcy setting!
:::
:::

# Off-policy policy gradients (2)

::: small
$$ \nablaphi\Vpiphi \approx \sum_{s\in\Ac} \rho_\beta(s) \sum_{a\in\Ac} \nablaphi\piphi\agivenb{a}{s} \Qpiphi(s,a) = \Expsub{\sum_{a\in\Ac} \nablaphi\piphi\agivenb{a}{s} \Qpiphi(s,a)}{s\sim\rho_\beta}. $$

::: incremental
- To make this off-policy w.r.t. the actions as well, we introduce *importance sampling* for $\beta$:
[$$\begin{align*}
\nablaphi\Vpiphi &\approx \sum_{s\in\Ac} \rho_\beta(s) \sum_{a\in\Ac} \textcolor{red}{\beta\agivenb{a}{s}} \underbrace{\frac{\textcolor{blue}{\piphi\agivenb{a}{s}}}{\textcolor{red}{\beta\agivenb{a}{s}}}}_{=\kappa_\phi(s,a)} \frac{\nablaphi\piphi\agivenb{a}{s}}{\textcolor{blue}{\piphi\agivenb{a}{s}}} \Qpiphi(s,a) \fragment{ = \sum_{s\in\Ac} \rho_\beta(s) \sum_{a\in\Ac} \beta\agivenb{a}{s} \kappa_\phi(s,a) \nablaphi\log\,\piphi\agivenb{a}{s} \Qpiphi(s,a) } \\
&= \Expsub{\kappa_\phi(s,a) \nablaphi\log\,\piphi\agivenb{a}{s} \Qpiphi(s,a)}{s\sim\rho_\beta,a\sim \beta\agivenb{\cdot}{s}}.
\end{align*}$$]{.math-incremental}
- The starting point in [@Silver2014dpg] was very similar, but using continuous state and action spaces:
$$
\nablaphi\Vpiphi \approx \int_\Sc\rho_\beta(s)\int_\Ac  \kappa_\phi(s,a) \nablaphi\log\,\piphi\agivenb{a}{s} \Qpiphi(s,a) \dint{a} \dint{s} = \Expsub{\kappa_\phi(s,a) \nablaphi\log\,\piphi\agivenb{a}{s} \Qpiphi(s,a)}{s\sim\rho_\beta,a\sim \beta\agivenb{\cdot}{s}}.
$$
- We now have derived a simplified formula for the off-policy policy gradient.
- However, we still suffer from the large variance issue.
:::
:::

------------------------------------------------------------------------------

# Deterministic policy gradient

------------------------------------------------------------------------------

# Deterministic off-policy policy gradients

::: small
::: incremental
- Drop the randomness from the policy $\mu$ such that it becomes a **deterministic function**: $a = \mu_\phi(s)$.
- Reformulate $V$ via $Q$: no need for expecations (i.e., sum / integrate over $\Ac$): $V^{\mu_\phi}(s) = Q^{\mu_\phi}(s,\mu_\phi(s))$.
- Deterministic: we are incapable of exploration! [The theorem is only practically useful if we sample from off-policy data:
$$L(\phi)= \Expsub{Q^{\mu_\phi}(s,\mu_\phi(s))}{s\sim\rho_{\beta}} = \int_\Sc \rho_{\beta}(s) Q^{\mu_\phi}(s,\mu_\phi(s)) \dint{s}. $$]{.fragment}
:::


::: fragment
::: definition
## Deterministic policy gradient theorem [@Silver2014dpg]

### Exact version: on-policy ($\rho_{\beta} = \rho_{\mu_\phi}$) 

$$\begin{equation}\begin{aligned}
\nablaphi L(\phi) &= \nablaphi\rbracket{\int_\Sc \rho_{\beta}(s) Q^{\mu_\phi}(s,\mu_\phi(s)) \dint{s}} = \int_\Sc \rho_{\beta}(s) \nablaa Q^{\mu_\phi}(s,\mu_\phi(s)) \big|_{a=\mu_\phi(s)} \nablaphi \mu_\phi(s) \dint{s} \\
&= \Expsub{\nablaa Q^{\mu_\phi}(s,\mu_\phi(s)) \big|_{a=\mu_\phi(s)} \nablaphi \mu_\phi(s)}{s\sim\rho_{\beta}}. 
\end{aligned} \label{eq:Adv2_dpg} \end{equation}$$

::: fragment
**Approximate version: off-policy** ($\rho_{\beta} \neq \rho_{\mu_\phi}$, and we use the same simplification as earlier, i.e., $\cancel{\mu_\phi\agivenb{a}{s} \nablaa Q^{\mu_\phi}(s,a)}$) 

$$\begin{equation}
\nablaphi L(\phi) \approx \Expsub{\nablaa Q^{\mu_\phi}(s,\mu_\phi(s)) \big|_{a=\mu_\phi(s)} \nablaphi \mu_\phi(s)}{s\sim\rho_{\beta}}.\label{eq:Adv2_dpg_approx} 
\end{equation}$$
:::

[:bulb: \eqref{eq:Adv2_dpg} is the limit case of the stochastic policy gradient theorem (i.e., $\Var{\pi}\to 0$) [@Silver2014dpg, Theorem 2].]{.fragment}
:::
:::
:::

:::


# On-policy deterministic actor-critic

::: small
The simplest algorithm we can derive from this: SARSA-type on-policy Actor-Critic:

::: columns-5-6
::: definition
### Algorithm: On-policy deterministic actor-critic

::: incremental
1. Sample $\set{s_i,a_i,r_i,s'_i,a'_i}_{i=1}^N$ using $a = \mu_\phi(s)$.
1. TD error: $$\delta_i = r_i + \gamma Q_\theta(s'_i,a'_i) - Q_\theta(s_i,a_i).$$
1. Semi-gradient $Q$-function update: 
$$\theta \gets \theta + \alpha_\theta \frac{1}{N} \sum_{i=1}^N \delta_i \nablatheta Q_\theta(s_i,a_i).$$
1. Update policy $\mu_\phi$ by sampling \eqref{eq:Adv2_dpg}: $$\phi \gets \phi + \alpha \frac{1}{N} \sum_{i=1}^N \nablaa Q_\theta(s_i,a_i) \nablaphi \mu_\phi(s_i).$$
:::
:::

::: fragment
### Summary of the concept

To update a deterministic policy off-policy, your algorithm does this for a batch of states from the replay buffer: 

::: incremental
1. Pass state $s$ into the Actor: $a = \mu_\phi(s)$. 
1. Pass $s$ and $a$ into the Critic network $Q_\theta(s, a)$.
1. Compute how the Critic's output changes with respect to that action ($\nabla_a Q$).
1. Compute how the Actor's weights change to produce that action change ($\nablaphi \mu_\phi$).
1. Multiply them together to update the Actor. 
:::
:::
:::

:::


------------------------------------------------------------------------------

# Deep deterministic policy gradient (DDPG)

------------------------------------------------------------------------------

# Deep deterministic policy gradient (DDPG)

::: small
::: columns-4-6
::: platzhalter
## Problems with vanilla DPG

::: fragment
### The original algorithm assumed:
:::
::: incremental
- tabular / linear approximators ("Compatible Function Approximation" in [@Silver2014dpg]),
- stable $Q$ estimation.
:::

::: fragment
### When neural networks are used, several problems appear:
:::
::: incremental
- Bootstrapping instability
- Correlated samples from trajectories
- Targets change too quickly
- Poor exploration due to deterministic policy 
:::
:::

::: platzhalter
::: fragment
## Deep deterministic policy gradient (DDPG) [@Lillicrap2015ddpg] 
addresses these by importing techniques from Deep Q-Networks (DQN). 
:::
\

::: fragment
### Core components:
:::

::: incremental
- Actor network $\mu_\phi(s)$
- Critic network $Q_\theta(s,a)$
- Target actor $\mu_{\phi'}$​
- Target critic $Q_{\theta'}$​
- Replay buffer
:::
:::
:::
:::

# DDPG Training

::: small
::: definition
::: columns-6-4
::: incremental
1. **Interact**: Sample $\set{s_t,a_t,r_t,s_{t+1}}$ using $a_t=\mu_\phi(s_t)$ and store in the replay buffer $\Dc$.  
1. **Sample**: A random mini-batch of $N$ transitions.
1. **Update Critic**: Calculate the target $y_i$ using the Target Networks:
$$y_i = r_i + \gamma Q_{\theta'}(s_{i+1}, \mu_{\phi'}(s_{i+1}))$$
The *online Critic* is updated by minimizing the Bellman error:
:::

![Source: [@DDPG_Image]](images/11-advanced/DDPG_v2.png){width=450px}
:::

[$$L(\theta) = \frac{1}{N}\sum_{i} \left( y_i - Q_\theta(s_i, a_i) \right)^2, \qquad \theta \gets \theta + \alpha_\theta \frac{2}{N} \sum_{i=1}^N \left( y_i - Q_\theta(s_i, a_i) \right) \nablatheta Q_\theta(s_i,a_i).$$]{.fragment}

::: incremental
4. **Update Actor**: The *online Actor* is updated using the sampled deterministic policy gradient (Eq. \eqref{eq:Adv2_dpg}): 
$$\phi \gets \phi + \alpha \frac{1}{N} \sum_{i=1}^N \nablaa Q_\theta(s_i,a_i) \nablaphi \mu_\phi(s_i).$$
<!-- $$\nablaphi L(\phi) \approx \frac{1}{N}\sum_{i} \nabla_a Q_\phi(s_i, a) \Big|_{a=\mu_\phi(s_i)} \cdot \nabla_\theta \mu_\phi(s_i)$$ -->
5. **Soft Updates**: The target Networks (i.e., $\phi'$ and $\theta'$) are updated incrementally. 
:::


:::
:::

# The four main changes over DPG

::: small
::: columns-5-5
::: incremental
1. **Integration of deep neural networks (CNNs)**  
2. **Introduction of the replay buffer**
  - Larger datasets to train from.
  - Allows for i.i.d. sampling / breaks the temporal correlation of data.
3. **Explicit action noise for exploration** 
- Deterministic actor in DPG: it cannot explore on its own.
- DDPG adds an explicit random process $\Nc$ directly to the action selection during environment interaction.  
- The original paper utilized [Ornstein-Uhlenbeck](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) noise, which creates temporally correlated, mean-reverting patterns that mimic inertial drift.
:::

::: incremental
4. **Transition to "Soft updates" for target networks**
- In DQN, target network weights are periodically copied exactly from the online network every few thousand steps.
- For continuous actor-critic configurations, hard updates changed the value landscape too abruptly. 
- Instead: soft update, where target networks track the online networks smoothly at every single training step using an interpolation factor $\tau \ll 1$ (e.g., $\tau = 0.001$): 
$$\theta' \leftarrow \tau \theta + (1 - \tau)\theta', \qquad \phi' \leftarrow \tau \phi + (1 - \tau)\phi'.$$
[$\Rightarrow$ targets change slowly, providing an unmoving baseline that stabilizes the deep network's gradients.]{.fragment}
- We have seen this before under *Polyak averaging*.
:::
:::
::: fragment
![](images/08-deep-q-learning/Polyak-averaging.svg){ width=850px }
:::
:::

------------------------------------------------------------------------------

# Twin Delayed Deep Deterministic Policy Gradient (TD3) [@Fujimoto2018TD3] {menu-title="Twin Delayed Deep Deterministic Policy Gradient (TD3)"}

------------------------------------------------------------------------------

# TD3

::: small

:::

------------------------------------------------------------------------------

# Soft actor-critic (SAC) [@Haarnoja2018sac] {menu-title="Soft actor-critic (SAC)"}

------------------------------------------------------------------------------

# Soft actor-critic (SAC)

::: small

:::





# References

::: { #refs }
:::
