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
- Approach (I): Improving policy gradient methods
  - Natural PG
  - TRPO
  - PPO
- Approach (II): Improving $Q$-learning
  - Deterministic Policy Gradient 
  - DDPG
  - TD3 
  - Soft Actor-Critic
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
|  [11]{style="color: red;"}  | [Advanced algorithms]{style="color: red;"} | [The evolution of moderl RL algorithms]{style="color: red;"} | 
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

# Approach (I): Improving policy gradient methods

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

# Constraining the ascent direction

::: small
::: columns-9-4
::: incremental
- Recall the policy gradient update: $\phi \gets \phi + \alpha \nablaphi L(\phi)$.
- A natural idea: **Constrain the length of our update step**!
  - Linearization of our optimization problem around $\phi$ using [Taylor series expansion](https://en.wikipedia.org/wiki/Taylor_series):
  [$$\begin{align*} 
  L(\phi') &= L(\phi) + \sum_{i=1}^n \pdiff{L}{\phi_i} (\phi'_i - \phi_i) + \Oc((\phi' - \phi)^2) \\
  &= L(\phi) + (\phi' - \phi)^\top \nablaphi L(\phi).
  \end{align*}$$]{.math-incremental}
  - Constrained optimization of $\phi'$ along the steepest ascent direction:
  $$\phi' \gets \arg\max_{\phi'} (\phi' - \phi)^\top \nablaphi L(\phi) \qquad \fragment{ \text{subject to}\qquad \norm{\phi' - \phi}^2\leq \epsilon. } $$
- **But**: We just saw that some parameters change probabilities *a lot more than others*! 
:::

[![Source: [@Peters2008naturalac]](images/11-advanced/Covariant-PG-vanilla.png){ width=350px }]{.fragment}
:::

::: incremental
- **Alternative**: Rescale the gradient so that this does not happen!
- A better way to constrain the update distance: The **change in distribution** via the [**Kullback-Leibler (KL) divergence**](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)!
$$ \KLdiv{\pi_\phi}{\pi_\phi'}. $$
:::

:::

# Aside: KL-divergence and Fisher information matrix (1)

::: small
::: columns-7-3
::: incremental
- The **Kullback-Leibler divergence** (or **KL divergence**) measures the "distance" between two distributions (e.g., $p(x)$ and $q(x)$)
$$ \KLdiv{p}{q}= \begin{cases} \sum_{x\in\Xc} p(x) \log\cbracket{\frac{p(x)}{q(x)}} & \text{if}~\Xc~\text{is finite} \\ \int{\Xc} p(x) \log\cbracket{\frac{p(x)}{q(x)}}\dx & \text{if}~\Xc~\text{is continuous} \end{cases}. $$
  - It is zero if and only if $p(x) = q(x)$.
- **Intuition**: Imagine you have two different maps of the exact same city.
  - Map A is completely accurate. Every street, etc., is exactly where it should be.
  - Map B was drawn from memory by a friend. It's mostly right with some minor flaws.
  - If you use Map B to navigate the city, you are going to make some mistakes.
  - The KL divergence answers the question: "How much extra 'gas' (or information) will I waste if I use Map B instead of Map A?"
:::

![[[Source](https://medium.com/@yian.chen261/introduction-to-kullback-leibler-divergence-2d76979d1d8c)].](images/11-advanced/KLdivergence-example.png){ width=450px}
:::

::: incremental
- KL Divergence measures the *surprise* or *unfair penalty* you get when you use $q$ to predict something that actually follows $p$. If $q$ is a perfect match for $p$ (i.e., $q=p$), your surprise is zero. The worse your guess ($q$) is, the higher the KL divergence.
- **Question**: What happens if our "model" $q$ is absolutely certain that something is **not** going to happen ($q(x)=0$ for some $x$)?
:::
:::

::: footer
:bulb: The KL divergence is not a real distance. For instance, it it not symmetric (i.e., $\KLdiv{p}{q} \neq \KLdiv{q}{p}$ in general), which means that it also does not satisfy the triangle inequality. Still, it gives us a measure of how far two distributions are apart.
:::

# Aside: KL-divergence and Fisher information matrix (2)

::: small
::: columns-5-4
::: incremental
- The **Fisher information matrix** (**FIM**) measures how much information an observable random variable $x$ carries about an unknown parameter $\phi$ (e.g., the weights of a neural network).
- Mathematically, if we have a log-likelihood function $\log p(x|\theta)$, the FIM (denoted as $F$) is the variance of its gradient (the "*score*"):
$$F = \Exp{\cbracket{\nablaphi \log\, p\agivenb{x}{\phi}} \cbracket{\nablaphi \log\, p\agivenb{x}{\phi}}^\top}.$$
:::

::: fragment
::: definition
**In simpler terms**: How do variations of $\phi$ influence the resulting probability distribution?

::: incremental
- High Fisher Information: Small change in $\phi$ $\Rightarrow$ large shift in the distribution.
- Low Fisher Information: Change in $\phi$ $\Rightarrow$ only small shift in the distribution.
:::
:::
:::
:::

::: fragment
### Relation to the KL divergence
:::

::: incremental
- Consider a very small change in $\phi$ by $\delta\phi$, and study the KL divergence $\KLdiv{p\agivenb{x}{\phi}}{p\agivenb{x}{\phi + \delta\phi}}$.
- If we perform a Taylor series expansion of $D_{\mathsf{KL}}$ around $\delta\phi=0$, then the first two terms are zero ($\KLdiv{p\agivenb{x}{\phi}}{p\agivenb{x}{\phi}} = 0$ and since $\delta\phi=0$ is the minimum, the first derivative is zero as well).
- The second order term is thus the leading term! $\Rightarrow$ for small $\delta\phi$, we have
$$\KLdiv{p\agivenb{x}{\phi}}{p\agivenb{x}{\phi + \delta\phi}} \approx \frac{1}{2} \delta\phi^\top F \delta\phi.$$
:::

:::

# Aside: KL-divergence and Fisher information matrix (3)

::: small
::: incremental
- When using the KL divergence or the Fisher information matrix in RL, we often do so in terms of different policies.
- This means that we consider the difference in the **action distribution** $\KLdiv{\pi_\phi}{\pi_{\phi'}}$.
- However, this expression is incomplete or even misleading. 
- Instead, we consider either
$$ \KLdiv{\pi_\phi\agivenb{\cdot}{s}}{\pi_{\phi'}\agivenb{\cdot}{s}} = \int_{\Ac} \pi_\phi\agivenb{a}{s} \log\cbracket{\frac{\pi_\phi\agivenb{a}{s}}{\pi_{\phi'}\agivenb{a}{s}}}\dint{a}$$
for a fixed $s$, [or the expectation over the state space, i.e.,
$$\Expsub{\KLdiv{\pi_\phi\agivenb{\cdot}{s}}{\pi_{\phi'}\agivenb{\cdot}{s}}}{s \sim p_\phi}. $$]{.fragment}
- The relation between KL divergence and FIM for small changes can be derived analogously to the previous slide and reads
$$\Expsub{\KLdiv{\pi_\phi}{\pi_{\phi'}}}{s \sim p_\phi} \approx \frac{1}{2} \delta\phi^\top \big( \underbrace{\Expsub{F(s)}{s \sim p_\phi}}_{=F} \big) \delta\phi \fragment{ = \frac{1}{2} \delta\phi^\top F \delta\phi.}$$
:::
:::

# Back to constrained ascent directions

::: small
::: incremental
- Recall: constraining w.r.t. the parameter $\phi$ does not solve the issue of strongly different impacts of individual parameters:
$$\phi' \gets \arg\max_{\phi'} (\phi' - \phi)^\top \nablaphi L(\phi) \qquad \text{subject to}\qquad \norm{\phi' - \phi}^2\leq \epsilon.\qquad\qquad\qquad\qquad $$
- **Question**: What is a good alternative to avoid too large (and thus, harmful) steps?\
[$\Rightarrow$ the KL divergence!
$$\phi' \gets \arg\max_{\phi'} (\phi' - \phi)^\top \nablaphi L(\phi) \qquad \text{subject to}\qquad \Expsub{\KLdiv{\pi_{\phi'}\agivenb{\cdot}{s}}{\pi_{\phi}\agivenb{\cdot}{s}}}{s \sim p_\phi}\leq \epsilon. $$]{.fragment} 
- Since we want to have small updates (constrained by $\epsilon$): approximation via the Fisher information matrix,
$$\begin{align*} \phi' \gets \arg\max_{\phi'} (\phi' - \phi)^\top \nablaphi L(\phi) \qquad &\text{subject to}\qquad (\phi'-\phi)^\top F (\phi'-\phi), \\
&\text{with}\qquad F = \Expsub{\cbracket{\nablaphi \log\, \pi_{\phi}\agivenb{a}{s}} \cbracket{\nablaphi \log\, \pi_{\phi}\agivenb{a}{s}}^\top}{s \sim p_\phi, a \sim \pi_\phi\agivenb{\cdot}{s}}.
\end{align*} $$

:::
:::

# Naturla policy gradient

::: small
::: columns-7-3
::: incremental
- We have now defined a policy gradient whose update length is constrained in terms of the KL divergence: 
$$\begin{equation} \phi' \gets \arg\max_{\phi'} (\phi' - \phi)^\top \nablaphi L(\phi) \quad \text{subject to}\quad (\phi'-\phi)^\top F (\phi'-\phi). \label{eq:Adv_Natural_PG} \end{equation}$$
- What remains is the question of solving \eqref{eq:Adv_Natural_PG}.
- Without going into further details (see [@Peters2008naturalac] for that): The solution is to **scale** the policy gradient **by the inverse Fisher information matrix**:
$$\phi \gets \phi + \alpha F^{-1} \nablaphi L(\phi).$$
- **Intuition** behind the inverse FIM $F^{-1}$:
  - Parameters with a high impact have high Fisher information.
  - Scaling by the inverse "normalizes" the individual impacts.
:::

::: platzhalter
[![](images/11-advanced/Covariant-PG-vanilla.png){ width=300px }]{.fragment}

\

[![Source: [@Peters2008naturalac]](images/11-advanced/Covariant-PG.png){ width=300px }]{.fragment}
:::


:::
:::




------------------------------------------------------------------------------

# Approach (II): Improving $Q$-learning

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
