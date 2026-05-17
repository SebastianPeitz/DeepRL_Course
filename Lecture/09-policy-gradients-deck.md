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
- Policy gradients
  - The objective
  - Evaluation the objective
  - Differentiating the objective
- The REINFORCE algorithm
- Understanding and challenges
- Reducing the variance
  - Reward to go
  - Baselines

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
&\Rightarrow\quad \phi^* = \arg\max_{\phi}\Expsub{\sum_{t=0}^{T-1}r_t}{\tau\sim p_\phi(\tau)}.
\end{align*} $$]{.math-incremental}
:::
:::

:::

# The reinforcement learning objective
[$$ \begin{align*} 
\phi^* &= \arg\max_{\phi}\Expsub{\sum_{t=0}^{T-1}r_t}{\tau\sim p_\phi(\tau)}\\
\text{Infinite horizon case:}\quad\phi^* &= \arg\max_{\phi}\Expsub{r}{(s,a)\sim p_\phi(s,a)}\\
\text{Finite horizon case:}\quad\phi^* &= \arg\max_{\phi}\sum_{t=0}^{T-1}\Expsub{r_t}{(s_t,a_t)\sim p_\phi(s_t,a_t)}
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
$$ \phi^* = \arg\max_{\phi}\underbrace{\Expsub{\sum_{t=0}^{T-1}r_t}{\tau\sim p_\phi(\tau)}}_{L(\phi)} $$

::: incremental
- First, let's think about evaluating the $L(\phi)$ for a fixed policy $\pi$.
- How do we estimate expectations if we don't have access to a model?
:::

[$\Rightarrow$ Monte Carlo sampling!
$$L(\phi) = \Expsub{\sum_{t=0}^{T-1}r_t}{\tau\sim p_\phi(\tau)} \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T-1}r_{i,t} $$]{.fragment}

:::

<!-- [![Source: Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/).](images/09-policy-gradients/Samples-trajectory.png){ width=500px }]{.fragment} -->
[![Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/).](images/09-policy-gradients/PG-visualization_1.svg){ .embed width=500px }]{.fragment}

:::

::: small
::: incremental
- Sample $N$ trajectories with $s_0$ according to the initial distribution and then following $\pi$.
- Take the sample average over these trajectories.
:::

[**Policy gradient**: Depending on the return, the plan is to increase or reduce the trajectories' likelihoods by adapting the policy $\pi$.]{.fragment}

:::

# Differentiating the policy (1)

::: small
::: columns-5-5
[$$ \begin{align*} \phi^* &= \arg\max_{\phi}\underbrace{\Expsub{\sum_{t=0}^{T-1}r_t}{\tau\sim p_\phi(\tau)}}_{L(\phi)}\\ 
L(\phi) &= \E_{\tau\sim p_\phi(\tau)}[\underbrace{r(\tau)}_{=\sum_{t=0}^{T-1}r_t}] = \int p_\phi(\tau) r(\tau) \dtau\\
\nablaphi L(\phi) &= \nablaphi \rbracket{\int p_\phi(\tau) r(\tau) \dtau}
\end{align*}$$]{.math-incremental}

::: platzhalter
::: definition
### Definition of the expectation in continuous spaces

The expected value of a function $x(\tau)$ is

$$ \Expsub{x(\tau)}{\tau\sim p} = \int p(\tau) x(\tau) \dtau. $$

Here, $p$ is the density according to which $\tau$ is distributed, with $\int p(\tau) \dtau = 1$.
:::
[**Note**: Both integration and differentiation are *linear* operations]{.fragment}
[$\Rightarrow$ We can swap!]{.fragment}
:::
:::

::: fragment
::: columns-6-5
[$$\begin{align*}
~&= \int \textcolor{red}{\nablaphi p_\phi(\tau)} r(\tau) \dtau  \\
&= \int \textcolor{blue}{p_\phi(\tau) \nablaphi \log p_\phi(\tau)} r(\tau)
\end{align*} $$]{.math-incremental}
[$$\begin{align}
\quad = \Expsub{\nablaphi \log p_\phi(\tau) r(\tau)}{\tau\sim p_\phi(\tau)} \label{eq:PG_policy_gradient}
\end{align} $$]{.fragment}

::: definition
### A convenient identity

$$
\begin{equation}
\textcolor{blue}{p_\phi(\tau) \nablaphi \log p_\phi(\tau)} = p_\phi(\tau) \frac{\nablaphi p_\phi(\tau)}{p_\phi(\tau)} = \textcolor{red}{\nablaphi p_\phi(\tau)} \label{eq:PG_log_identity}
\end{equation}
$$
:::
:::
:::

:::


# Differentiating the policy (2)

::: small
::: columns-4-6
::: platzhalter
\
\
\
$$ \begin{align*} \phi^* &= \arg\max_{\phi}L(\phi)\\ 
L(\phi) &= \Expsub{r(\tau)}{\tau\sim p_\phi(\tau)} = \int p_\phi(\tau) r(\tau) \dtau\\
\nablaphi L(\phi) &= \Expsub{\textcolor{blue}{\nablaphi \log p_\phi(\tau)} r(\tau)}{\tau\sim p_\phi(\tau)} 
\end{align*}$$
:::

::: fragment
::: definition
[$$\begin{align*} 
\underbrace{p_\phi(s_0,a_0,\ldots,s_{T-1},a_{T-1},s_{T})}_{=p_\phi(\tau)} &= p(s_0) \prod_{t=0}^{T-1} \pi_\phi\agivenb{a_t}{s_t} p\agivenb{s_{t+1}}{s_t,a_t} \\ 
\text{take log on both sides!}\quad &\Downarrow \quad\fragment{ \textcolor{gray}{(\log(a\cdot b) = \log(a) + \log(b))} }  \\
\log p_\phi(\tau) = \log p(s_0) + &\sum_{t=0}^{T-1} \rbracket{\log\pi_\phi\agivenb{a_t}{s_t} + \log p\agivenb{s_{t+1}}{s_t,a_t}}
\end{align*}$$]{.math-incremental}
:::
:::
:::

::: columns-6-4
[$$\begin{align*} 
\textcolor{blue}{\nablaphi \log p_\phi(\tau)} &= \nablaphi \rbracket{\log p(s_0) + \sum_{t=0}^{T-1} \rbracket{\log\pi_\phi\agivenb{a_t}{s_t} + \log p\agivenb{s_{t+1}}{s_t,a_t}}}\\
 &= \nablaphi \log p(s_0) + \sum_{t=0}^{T-1} \rbracket{\nablaphi \log\pi_\phi\agivenb{a_t}{s_t} + \nablaphi \log p\agivenb{s_{t+1}}{s_t,a_t}}\\
&= \textcolor{red}{\cancel{\nablaphi \log p(s_0)}} + \sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_t}{s_t} + \textcolor{red}{\cancel{\nablaphi \log p\agivenb{s_{t+1}}{s_t,a_t}}}
\end{align*}$$]{.math-incremental}

::: fragment
::: definition
### Policy gradient theorem (simplified)

[$$\begin{align*} \nablaphi L(\phi) = \Expsub{\cbracket{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_t}{s_t}}r(\tau)}{\tau\sim p_\phi(\tau)}\\
= \Expsub{\cbracket{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_t}{s_t}}\cbracket{\sum_{t=0}^{T-1}r_t}}{\tau\sim p_\phi(\tau)}
 \end{align*}$$]{.math-incremental}
:::
:::
:::

:::

# Policy gradient theorem
::: small
::: columns-5-5

::: platzhalter
[**What does this mean?**]{.fragment}

::: incremental
1. We want to find a policy that maximizes the value: $$\max_{\pi}\Expsub{\sum_{t=0}^{T-1}r_t}{\tau\sim p^\pi} = \max_{\pi}\Expsub{r(\tau)}{\tau\sim p^\pi}.$$
:::
:::

::: definition
### Policy gradient theorem (simplified)

$$\nablaphi L(\phi) = \Expsub{\cbracket{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_t}{s_t}}\cbracket{\sum_{t=0}^{T-1}r_t}}{\tau\sim p_\phi(\tau)}$$
:::
:::

::: incremental
2. Using a neural network approximator $\pi_\phi$ $\Rightarrow$ maximization w.r.t. the policy parameters: $\max_{\phi}\Expsub{r(\tau)}{\tau\sim p_\phi(\tau)} = \max_{\phi}L(\phi).$
3. This yields a challenging loss function to optimize: Integration over $\tau$ (i.e., the space of trajectories) is infeasible!\
[$\Rightarrow$ Approximate via [Monte Carlo sampling]{style="color: blue;"}:]{.fragment}
$$L(\phi) = \Expsub{r(\tau)}{\tau\sim p_\phi(\tau)} = \int p_\phi(\tau) r(\tau) \dtau \fragment{ \textcolor{blue}{\approx \frac{1}{N}\sum_{i=1}^N \sum_{t=0}^{T-1}r_{i,t}}. }$$
4. Use $\log$ identity to derive a formulation of the gradient that we can *approximate using sampling*:
$$ \nablaphi L(\phi) = \Expsub{\cbracket{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_t}{s_t}}\cbracket{\sum_{t=0}^{T-1}r_t}}{\tau\sim p_\phi(\tau)}.$$
:::

:::

# REINFORCE

::: small
We now have a formulation of the policy gradient that we can **approximate using [Monte Carlo sampling]{style="color: blue;"}**:
$$ \nablaphi L(\phi) = \Expsub{\cbracket{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_t}{s_t}}\cbracket{\sum_{t=0}^{T-1}r_t}}{\tau\sim p_\phi(\tau)} \fragment{ \approx \textcolor{blue}{\frac{1}{N} \sum_{i=1}^N} \underbrace{\textcolor{blue}{\cbracket{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_{i,t}}{s_{i,t}}}}}_{\fragment{ \text{(I)} }}\underbrace{\textcolor{blue}{\cbracket{\sum_{t=0}^{T-1}r_{i,t}}}}_{\fragment{ \text{(II)} }}. }$$

::: columns-5-5
::: platzhalter
[(I) This term is simpliy the derivative of the policy network w.r.t. the weights $\Rightarrow$ backpropagation!\
]{.fragment}
[(II) This term is the experience we collect from interactions with the environment.]{.fragment}
:::

![](images/09-policy-gradients/CNN-policy.svg){ width=600px }
:::

::: fragment
::: definition

### The REINFORCE algorithm

::: incremental
1. Sample $\set{\tau_i}_{i=1}^N$ using $\pi_\phi\agivenb{a}{s}$ $\Rightarrow$ $\set{((s_{i,0},a_{i,0},r_{i,0}),\ldots,(s_{i,T-1},a_{i,T-1},r_{i,T-1}))}_{i=1}^N$.
2. $\nablaphi L(\phi) \approx \frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_{i,t}}{s_{i,t}}}\cbracket{\sum_{t=0}^{T-1}r_{i,t}}$.
3. Gradient ascent: $\phi \gets \phi + \alpha \nablaphi L(\phi)$.
:::

:::
:::
:::


------------------------------------------------------------------------------

# Understanding the policy gradient 

------------------------------------------------------------------------------

# Continuous action example: A Gaussian policy

::: small
::: incremental
- Let's assume that we have a Gaussian policy whose mean is modeled by a neural network:
$$ \pi_\phi\agivenb{a}{s} = \Normal{f_{\mathsf{NN}}(s)}{\Sigma} = \frac{1}{C}\exp\cbracket{-\frac{1}{2}(a - f_{\mathsf{NN}}(s))^\top \Sigma^{-1} (a - f_{\mathsf{NN}}(s))} = \frac{1}{C}\exp\cbracket{-\frac{1}{2}\norm{a - f_{\mathsf{NN}}(s)}_\Sigma^2}.$$
- Taking the log yields (with $\log C^{-1} = - \log C$): $$ \log \pi_\phi\agivenb{a}{s} = -\frac{1}{2}\norm{a - f_{\mathsf{NN}}(s)}_\Sigma^2 - \log C.$$
- Taking the gradient leads to the following expression: $$-\frac{1}{2}\Sigma^{-1} (f_{\mathsf{NN}}(s) - a) \pdiff{f_{\mathsf{NN}}}{\phi}.$$
:::

::: definition
### The REINFORCE algorithm

1. Sample $\set{\tau_i}_{i=1}^N$ using $\pi_\phi\agivenb{a}{s}$ $\Rightarrow$ $\set{((s_{i,0},a_{i,0},r_{i,0}),\ldots,(s_{i,T-1},a_{i,T-1},r_{i,T-1}))}_{i=1}^N$.
2. $\nablaphi L(\phi) \approx \frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_{i,t}}{s_{i,t}}}\cbracket{\sum_{t=0}^{T-1}r_{i,t}}$.
3. Gradient ascent: $\phi \gets \phi + \alpha \nablaphi L(\phi)$.
:::
:::

# Comparison to maximum likelihood

::: small
::: columns-5-5
::: platzhalter
::: incremental
- Consider the task of maximizing the likelihood $L$ of a training dataset $\Dc=\set{(x_i,y_i)}_{i=1}^N$ in supervised learning, e.g., using a neural network with parameter vector $\theta$:
$$ \begin{align*} \max_\theta L(\theta) &= \max_\theta p_\theta\agivenb{y_1}{x_1} \times \ldots \times p_\theta\agivenb{y_N}{x_N} \\ &= \max_\theta \prod_{i=1}^N p_\theta\agivenb{y_i}{x_i}. \end{align*} $$
- Optimizing over products is hard $\Rightarrow$ take the log:
$$ \log L(\theta) = \log \cbracket{\prod_{i=1}^N p_\theta\agivenb{y_i}{x_i}}= \sum_{i=1}^N \log p_\theta\agivenb{y_i}{x_i}. $$
:::
::: fragment
::: definition
**Maximum likelihood gradient**:
$$\nablatheta \log L(\theta) = \sum_{i=1}^N \log \nablatheta p_\theta\agivenb{y_i}{x_i}.$$
:::
:::
:::

::: fragment
::: definition
**Comparison against the policy gradient**:

$$ \nablaphi L(\phi) \approx \frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_{i,t}}{s_{i,t}}}\cbracket{\sum_{t=0}^{T-1}r_{i,t}}.$$
:::
::: incremental
- The policy gradient can be seen as a weighted version of the maximum likelihood objective.
- Why is weighting is necessary?
  - In maximum likelihood, we always want to maximize the likelihood of the "correct" labels.
  - In policy gradients, we may also want to reduce the likelihood of low-reward samples!
:::
:::
:::
:::

# Understanding the policy gradient

::: small
::: columns-3-7

![Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/).](images/09-policy-gradients/PG-visualization.svg){ .embed width=350px }

::: platzhalter
::: incremental
- What did we just do?\
[$\Rightarrow$ let's rewrite the formula a little and *compare against maixmum likelihood*:
$$ \nablaphi L(\phi) \approx \frac{1}{N} \sum_{i=1}^N \underbrace{\nablaphi \log\pi_\phi(\tau_i)}_{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_{i,t}}{s_{i,t}}}\underbrace{r(\tau_i)}_{\sum_{t=0}^{T-1}r_{i,t}} \quad \text{vs.} \quad \nablatheta L(\theta) = \sum_{i=1}^N \log \nablatheta \pi_\phi(\tau_i). $$
]{.fragment}
- Good experience is made more likely: We increase the proability of the policy to produce similar trajectories
- Bad experience is made less likely.
- This simply formalizes the notion of "trial and error"!
:::
\

::: fragment
::: definition
### A note on partial observability (Without going into details)

The policy gradient also holds for partially observed MDPs (POMDPs). That is, for policies $\pi\agivenb{a}{o}$. In simple terms, the reason is that the policy gradient theorem does not make use of the Markov property.
:::
:::
:::
:::
:::


------------------------------------------------------------------------------

# Challenges with policy gradients

------------------------------------------------------------------------------

# High variance

::: small
::: columns-4-6

![Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/).](images/09-policy-gradients/PG-variance.svg){ .embed width=500px }

::: platzhalter
$$\nablaphi L(\phi) \approx \frac{1}{N} \sum_{i=1}^N \nablaphi \log\pi_\phi(\tau_i) r(\tau_i). $$

::: incremental
- Adding a constant to the reward should not change the optimal policy!
  - This is true for *any* optimization problem, e.g., $$\arg\min_x f(x) = \arg\min_x \rbracket{f(x) + 1000}.$$
- In the limit $N\rightarrow\infty$, this is true for policy gradients as well...
  - ... but it does not hold for finite sample sizes, in particular few sample trajectories.
- Depending on the sign of $r(\tau)$, we **either increase or decrease** the probability of seeing similar trajectories $\tau_i$ (and their rewards $r(\tau_i)$) in the future.
- This is an instance of the **high variance** issue with policy gradients.
- An even more "catastropic" version of this: What if we scale some of the rewards to be exactly zero?
:::
:::
:::
:::

# Reducing variance -- baselines

::: small
::: columns-4-6
::: platzhalter
$$\nablaphi L(\phi) \approx \frac{1}{N} \sum_{i=1}^N \nablaphi \log\pi_\phi(\tau_i) r(\tau_i). $$

::: incremental
- **Question**: Is there a systematic way to fix the challenge of reward offsets?\
[$\Rightarrow$ Let's balance the "weight" of of our likelihood objective in such a way that ...]{.fragment}
  - Better-than-average rewards *increase* the probability of the respective trajectories.
  - Worse-than-average rewards *decrease* the probability.
- Several **advantages**!
  - This aligns with our intuition that better-than-average is reinforced and worse-than-average is penalized.
  - This approach makes the policy gradient independent of the actual size of the reward signal.
:::
:::

::: platzhalter

::: fragment
::: definition
*Approach*: subtract a **baseline** $b$ from the reward signal
$$ \nablaphi L(\phi) \approx \frac{1}{N} \sum_{i=1}^N \nablaphi \log\pi_\phi(\tau_i) \rbracket{r(\tau_i) - b}, $$
[where $b=\frac{1}{N} \sum_{i=1}^N r(\tau_i)$. ]{.fragment}
:::
:::

::: fragment
::: definition
### Theorem

Subtracting any constant $b$ is *unbiased in expectation*
:::
:::

::: fragment
**Proof**: We start with the exact formulation of the policy gradient,
$$ \nablaphi L(\phi) \stackrel{\eqref{eq:PG_policy_gradient}}{=} \Expsub{\nablaphi \log p_\phi(\tau) r(\tau) }{\tau\sim p_\phi(\tau)}. $$
[Unbiased in expectation $\Rightarrow$ baseline term is zero in expectation:]{.fragment}
[$$\begin{align*}
&\Expsub{\nablaphi \log p_\phi(\tau) r(\tau) }{\tau\sim p_\phi(\tau)} \fragment{ = \int p_\phi(\tau) \nablaphi \log p_\phi(\tau) b \dtau } \\
&\stackrel{\eqref{eq:PG_log_identity}}{=} \int \nablaphi p_\phi(\tau) b \dtau \fragment{ = b \nablaphi \int p_\phi(\tau) \dtau } \fragment{ = \cbracket{\nablaphi 1} b } \fragment{ = 0. \qquad \square }
\end{align*}$$]{.math-incremental}
:::
:::
:::
:::


# The optimal baseline

::: small

::: columns-6-5-3

::: definition
$$ \nablaphi L(\phi) = \Expsub{\nablaphi \log p_\phi(\tau) \cbracket{r(\tau) - b} }{\tau\sim p_\phi(\tau)}. $$
:::

::: platzhalter
How do we find the optimal baseline?\
[$\Rightarrow$ optimization: minimize the variance]{.fragment}
:::

::: fragment
::: definition
$$ \Var{x} = \Exp{x^2} - \Exp{x}^2 $$
:::
:::
:::

[$$\begin{align*} \mathsf{var} &= \E_{\tau\sim p_\phi(\tau)}\big[(\underbrace{\nablaphi \log p_\phi(\tau)}_{=g(\tau)} \cbracket{r(\tau) - b})^2\big] - \Expsub{\nablaphi \log p_\phi(\tau) \cbracket{r(\tau) - \textcolor{red}{\cancel{b}}} }{\tau\sim p_\phi(\tau)}^2 \quad\text{(\textcolor{red}{unbiased baseline})} \\
\diff{\mathsf{var}}{b} &=\diff{\mathsf{}}{b} \Expsub{g(\tau)^2 \cbracket{r(\tau) - b}^2}{\tau\sim p_\phi(\tau)} \fragment{ = \diff{\mathsf{}}{b} \cbracket{\Expsub{g(\tau)^2 r(\tau)^2}{\tau\sim p_\phi(\tau)} - 2\Expsub{g(\tau)^2 r(\tau)b}{\tau\sim p_\phi(\tau)} + \Expsub{g(\tau)^2 b^2}{\tau\sim p_\phi(\tau)} } } \\
&= \diff{\mathsf{}}{b} \cbracket{\cancel{\Expsub{g(\tau)^2 r(\tau)^2}{\tau\sim p_\phi(\tau)}} - 2b\Expsub{g(\tau)^2 r(\tau)}{\tau\sim p_\phi(\tau)} + b^2 \Expsub{g(\tau)^2 }{\tau\sim p_\phi(\tau)} }\\
&= - 2\Expsub{g(\tau)^2 r(\tau)}{\tau\sim p_\phi(\tau)} +2 b \Expsub{g(\tau)^2 }{\tau\sim p_\phi(\tau)} \fragment{\stackrel{!}{=}0}.
\end{align*}$$]{.math-incremental}

::: fragment
::: columns-7-4
**The optimal baseline** $b^*$ is the expected reward, *weighted by gradient magnitudes*:
$$ b^* = \frac{\Expsub{g(\tau)^2 r(\tau)}{\tau\sim p_\phi(\tau)}}{\Expsub{g(\tau)^2 }{\tau\sim p_\phi(\tau)}}. $$

::: fragment
::: definition
**Note**: $b^*$ is hard to calculate. In practice, we usually resort to the average reward $$b=\frac{1}{N} \sum_{i=1}^N r(\tau_i).$$
:::
:::
:::
:::

:::


# Reducing variance -- causality

::: small
$$ \nablaphi L(\phi) \approx \frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_{i,t}}{s_{i,t}}}\cbracket{\sum_{t'=0}^{T-1}r_{i,t'}}.$$

::: incremental
- **Causality**: The policy at time $t'$ cannot affect the reward at time $t$ when $t<t'$.
  - "What you do now, is not going to change the rewards you received in the past."
- **Question**: Are we making use of causality in the above equation?
  - Let's rewrite it and make use of the distributive property ($a \cdot (b + c) = a\cdot b + a\cdot c$):
  $$ \nablaphi L(\phi) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_{i,t}}{s_{i,t}}\cbracket{\sum_{t'=0}^{T-1}r_{i,t'}}. $$
  [$\Rightarrow$ Past rewards (i.e., $t'<t$) have an impact on the policy $\pi_\phi$!]{.fragment}\
  - In expectation, these factors have to cancel out (and one can prove this). [**But**: for finite sample sizes, they do not and instead increase the variance.]{.fragment}
- **Simple fix**: "*reward to go*" $\hat{Q}_{i,t} = \sum_{\textcolor{red}{t'=t}}^{T-1}r_{i,t'}$ (that is, the only change is $0 \to t$),
$$ \nablaphi L(\phi) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_{i,t}}{s_{i,t}}\hat{Q}_{i,t}. $$
:::
:::
\

::: fragment
::: footer
:bulb: Do not confuse causality with the Markov property! The Markov propery (which may hold for a system, but does not have to) says that your future states do not depend on past states, just the present state. Causality is always true: "Rewards in the past are independent of decisions in the present."
:::
:::


# Off-policy policy gradients


# Advanced policy gradients

 Covariant/natural PG


# References

::: { #refs }
:::
