---
subtitle:    Actor-Critic Algorithms
chapter:     10
feedback:
  deck-id:  'deeprl-actor-critic'
...


------------------------------------------------------------------------------

# Content

------------------------------------------------------------------------------

# Content

- Two alternative derivations of the policy gradient theorem
  - Bottom-up derivation via the Q-function
  - Sampling with reduced variance
- Advantage functions


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
|  [10]{style="color: red;"}  | [Actor-critic algorithms]{style="color: red;"} | [Improved policy gradients via value functions]{style="color: red;"} | 
|  11  | Advanced algorithms                                       |        | 
|      | **Model-Based Control**                                   |        |
|      | **Advanced Topics**                                       |        |

Table: Lecture contents
:::

------------------------------------------------------------------------------

# A more detailed derivation of the policy gradient

------------------------------------------------------------------------------

# The Q-function version of the policy gradient

::: small
::: columns-5-5


::: incremental
- Recall that we started with the objective of maximizing the value directly by optimizing over our policy network weights:
[$$ \begin{align*} \phi^* &= \arg\max_{\phi}\Expsub{\sum_{t=0}^{T-1}r_t}{\tau\sim p_\phi(\tau)} \\ &= \arg\max_{\phi}\Expsub{V_\phi(s_0)}{s_0 \sim p_0}. \end{align*} $$]{.math-incremental}
:::

![Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse-fa23/).](images/09-policy-gradients/CNN-policy.svg){ width=700px }
:::

::: incremental
- Using the definition of the expectation of continuous random variables and the $\log$ derivative trick, we arrived at a formulation of the policy gradient from which we can sample using trajectory data ($\tau=(s_0,a_0,\ldots,s_{T-1},a_{T-1},s_T)$):
$$ \nablaphi L(\phi) = \Expsub{\cbracket{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_t}{s_t}}\cbracket{\sum_{t=0}^{T-1}r_t}}{\tau\sim p_\phi(\tau)} \fragment{ \approx \frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\,\pi_\phi\agivenb{a_{i,t}}{s_{i,t}}} \cbracket{\sum_{t=0}^{T-1}r_{i,t}}.}$$
- **Goal**: derive the policy gradient a second time, but this time using the Q-function instead of the return $\sum_{t=0}^{T-1}r_{i,t}$.
- **Starting point**: the RL objective and its gradient:
$$\begin{equation} L(\phi)=\Expsub{V_\phi(s_0)}{s_0 \sim p_0} \qquad \Rightarrow \qquad \nablaphi L(\phi)=\Expsub{\nablaphi V_\phi(s_0)}{s_0 \sim p_0} = \int_{\Sc} p_0(s) \nablaphi V_\phi(s) \ds. \label{eq:AC_RL_objective} \end{equation}$$
:::

:::

# Deriving the policy gradient (1)

:::small
::: incremental
- To derive the policy gradient, we need to take the **gradient of the value function**, i.e., $\nablaphi V_\phi(s)$.
- Our **goal is to introduce the Q-function** (we will see the reason for this later). [What's the relation between the Q-function and the value function for continuous actions?]{.fragment}
[$$ \underbrace{V^\pi(s) = \sum_{a\in\Ac} \pias Q^\pi(s,a)}_{\text{finite }\Ac} \fragment{ \qquad \text{vs.}\qquad \underbrace{V^\pi(s) = \int_\Ac \pi\agivenb{a}{s} Q^\pi(s,a)\dint{a}}_{\text{continuous }\Ac}. } $$]{.fragment}
- Now let's **take the derivative**! [Both $\pi_\phi$ and $Q_\phi$ depend on $\phi$ $\Rightarrow$ product rule [($\frac{d}{dx} (f(x)\cdot g(x)) = f(x) \frac{dg}{dx} + \frac{df}{dx} g(x)$)]{style="color: gray;"}:]{.fragment}
[$$\begin{align*} 
\nablaphi V_\phi(s) &= \nablaphi \cbracket{\int_\Ac \pi\agivenb{a}{s} Q_\phi(s,a)\dint{a}} \fragment{ = \int_\Ac \nablaphi \rbracket{\pi\agivenb{a}{s} Q_\phi(s,a)}\dint{a} } \fragment{ = \int_\Ac \nablaphi\pi\agivenb{a}{s} Q_\phi(s,a) + \pi\agivenb{a}{s} \nablaphi Q_\phi(s,a)\dint{a}} \\
&=  \underbrace{\int_\Ac\nablaphi\pi\agivenb{a}{s} Q_\phi(s,a)\dint{a}}_{= \xi(s)} + \int_\Ac\pi\agivenb{a}{s} \nablaphi Q_\phi(s,a)\dint{a}
\end{align*}$$]{.math-incremental}
- In total, we obtain the following **expression for $\nablaphi V_\phi(s)$**:
$$\begin{equation}
\nablaphi V_\phi(s) = \xi(s) + \int_\Ac\pi\agivenb{a}{s} \nablaphi Q_\phi(s,a)\dint{a}.
\label{eq:AC_grad_V}\end{equation}$$

:::
:::

# Deriving the policy gradient (2)

:::small
::: incremental
- Before we continue with \eqref{eq:AC_grad_V}, let's study $Q_\phi(s,a)$ in some more detail, with the **goal to find a Bellman recursion** formula:
[$$\begin{equation}  Q_\phi(s, a) = r + \int_{\Sc} \psprimesa V_\phi(s') \dint{s'}. \label{eq:AC_Q_definition} \end{equation}$$]{.fragment}
- Since neither $r$ nor $\psprimesa$ depend on $\phi$, taking the gradient of \eqref{eq:AC_Q_definition} yields
$$ \begin{equation} \nablaphi Q_\phi(s, a) = \int_{\Sc} \psprimesa \nablaphi V_\phi(s') \dint{s'}. \label{eq:AC_grad_Q} \end{equation} $$
- We can substitute this expression into \eqref{eq:AC_grad_V} and swap integrals due to linearity:
$$ \nablaphi V_\phi(s) = \xi(s) + \int_\Ac\pi\agivenb{a}{s} \underbrace{\rbracket{\int_{\Sc} \psprimesa \nablaphi V_\phi(s') \dint{s'}}}_{\eqref{eq:AC_grad_Q}}\dint{a} \fragment{ = \xi(s) + \int_{\Sc} \underbrace{\rbracket{\int_\Ac\pi\agivenb{a}{s} \psprimesa \dint{a}}}_{=p\agivenb{s \to s'}{1,\pi_\phi}} \nablaphi V_\phi(s') \dint{s'}.}$$
- Here, $p\agivenb{s \to s'}{1,\pi_\phi}$ denotes the probability of moving from $s$ to $s'$ in exactly one step, given the policy $\pi_\phi$:
  $$ \begin{equation} \nablaphi V_\phi(s) = \xi(s) + \int_{\Sc} p\agivenb{s \to s'}{1,\pi_\phi} \nablaphi V_\phi(s') \dint{s'}. \label{eq:AC_grad_V_bellman} \end{equation}$$
:::
:::

# Deriving the policy gradient (3)

:::small
::: incremental
- Equation \eqref{eq:AC_grad_V_bellman} is a Bellman recursion equation, and we can insert the same expression at $s'$ on the right:
$$ \nablaphi V_\phi(s) = \xi(s) + \int_{\Sc} p\agivenb{s \to s'}{1,\pi_\phi} \underbrace{\rbracket{\xi(s') + \int_{\Sc} p\agivenb{s' \to s''}{1,\pi_\phi} \nablaphi V_\phi(s'') \dint{s''}}}_{= \nablaphi V_\phi(s')} \dint{s'}. $$
- We can unroll this expression for an entire trajectory $\tau$ of length $T$. [After some resorting, we obtain:
$$ \nablaphi V_\phi(s) = \xi(s) + \int_{\Sc} p\agivenb{s \to s'}{1,\pi_\phi} \xi(s') \dint{s'} + \int_{\Sc} p\agivenb{s \to s''}{2,\pi_\phi}\xi(s'') \dint{s''} + \ldots$$]{.fragment}
- We can write this cleanly as a sum over all future timesteps $t$:
<!-- $$ \begin{equation} \nablaphi V_\phi(s) = \int_{\Sc} \sum_{t=0}^{\infty} p\agivenb{s \to x}{t, \pi_\phi} \xi(x) \dx. \label{eq:AC_grad_V_recursion} \end{equation} $$ -->
$$ \begin{equation} \nablaphi V_\phi(s) = \int_{\Sc} \sum_{t=0}^{T-1} p\agivenb{s \to x}{t, \pi_\phi} \xi(x) \dx. \label{eq:AC_grad_V_recursion} \end{equation} $$
[Note: the previously separate case $\xi(s)$ is included, i.e., $\int_{\Sc} p\agivenb{s \to s}{0, \pi_\phi} \xi(x)\dx =\xi(s)$.]{.fragment}
:::
:::

# Deriving the policy gradient (4)

:::small
::: incremental
- We can now insert \eqref{eq:AC_grad_V_recursion} into the gradient of our RL objective (Eq. \eqref{eq:AC_RL_objective}):
<!-- $$ \nablaphi L(\phi)=\Expsub{\nablaphi V_\phi(s_0)}{s_0 \sim p_0} \fragment{ = \int_{\Sc} p_0(s) \nablaphi V_\phi(s) \ds  }\fragment{ = \int_{\Sc} p_0(s) \rbracket{\int_{\Sc} \sum_{t=0}^{\infty} p\agivenb{s \to x}{t, \pi_\phi} \xi(x) \dx} \ds. }$$ -->
$$ \nablaphi L(\phi)=\Expsub{\nablaphi V_\phi(s_0)}{s_0 \sim p_0} \fragment{ = \int_{\Sc} p_0(s) \nablaphi V_\phi(s) \ds  }\fragment{ = \int_{\Sc} p_0(s) \rbracket{\int_{\Sc} \sum_{t=0}^{T-1} p\agivenb{s \to x}{t, \pi_\phi} \xi(x) \dx} \ds. }$$
- By changing the order of integration, we isolate the total expected time spent in state $x$:
<!-- $$\nablaphi L(\phi) = \int_{\Sc} \rbracket{ \int_{\Sc} p_0(s) \sum_{t=0}^{\infty} p\agivenb{s \to x}{t, \pi_\phi} \ds} \xi(x) \dx.$$ -->
$$\nablaphi L(\phi) = \int_{\Sc} \rbracket{ \int_{\Sc} p_0(s) \sum_{t=0}^{T-1} p\agivenb{s \to x}{t, \pi_\phi} \ds} \xi(x) \dx.$$
- The term inside the brackets is the **undiscounted state visitation frequency** (or the expected number of times state $x$ is visited in an episode), which we denote as $\eta_\phi(s)$:
<!-- $$\begin{equation} \eta_\phi(s) = \sum_{t=0}^{\infty} p\agivenb{s_t = s}{p_0, \pi_\phi}. \label{eq:AC_state_visitation_measure} \end{equation}$$ -->
$$\begin{equation} \eta_\phi(s) = \sum_{t=0}^{T-1} p\agivenb{s_t = s}{\pi_\phi}. \label{eq:AC_state_visitation_measure} \end{equation}$$
- This simplifies our expression to:
$$\nablaphi L(\phi) = \int_{\Sc} \eta_\phi(s) \xi(s) \ds.$$
:::
:::

# Deriving the policy gradient (5)

:::small
::: columns-6-4
::: incremental
- Reintroducing the full definition of $\xi(s) = \int_\Ac\nablaphi\pi\agivenb{a}{s} Q_\phi(s,a)\dint{a}$ (and keeping $s$ as our state variable):
$$\nablaphi L(\phi) = \int_{\Sc} \eta_\phi(s) \int_{\Ac} \nablaphi \pi_\phi\agivenb{a}{s} Q_\phi(s, a) \dint{a}\ds.$$
- Apply the $\log$-derivative identity:
$$\nabla_\phi L(\phi) = \textcolor{blue}{\int_{\Sc} \eta_\phi(s)} \textcolor{red}{\int_{\Ac} \pi_\phi\agivenb{a}{s}} \nabla_\phi \log \pi_\phi\agivenb{a}{s} Q_\phi(s, a) \textcolor{red}{\dint{a}} \textcolor{blue}{\ds}.$$
- In conclusion, the gradient can be expressed using the unnormalized state visitation measure $\eta_\phi(s)$:
<!-- $$\begin{equation} \nabla_\phi L(\phi) = \Expsub{\nablaphi \log \pi_\phi\agivenb{a}{s} Q_\phi(s, a)}{s \sim \eta_\phi, a \sim \pi_\phi}. \label{eq:AC_policy_gradient_Q} \end{equation}$$ -->
$$\begin{equation} \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \pi_\phi\agivenb{a_t}{s_t} Q_\phi(s_t, a_t)}{\tau\sim p_\phi(\tau)}. \label{eq:AC_policy_gradient_Q_episodic} \end{equation}$$
:::

::: platzhalter
::: definition
### Expectations in continuous spaces

The expected value of a function $f(s)$ is

$$ \Expsub{f(s)}{s\sim p} = \int p(s) f(s) \ds. $$

Here, $p$ is the density according to which $s$ is distributed, with $\int p(s) \ds = 1$.
:::
::: definition
### A convenient identity

$$\nabla_\phi \pi_\phi\agivenb{a}{s} = \pi_\phi\agivenb{a}{s} \nabla_\phi \log \pi_\phi\agivenb{a}{s}$$
:::
:::
:::

::: fragment
::: definition
**Note**: In the infinite-horizon case, \eqref{eq:AC_state_visitation_measure} becomes $\eta_\phi(s) = \sum_{t=0}^{\infty} p\agivenb{s_t = s}{p_0, \pi_\phi}$ [$\Rightarrow$ Equation \eqref{eq:AC_policy_gradient_Q_episodic} becomes
<!-- $$\begin{equation} \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \pi_\phi\agivenb{a_t}{s_t} Q_\phi(s_t, a_t)}{\tau\sim p_\phi(\tau)}. \label{eq:AC_policy_gradient_Q_episodic} \end{equation}$$]{.fragment} -->
$$\nabla_\phi L(\phi) = \Expsub{\nablaphi \log \pi_\phi\agivenb{a}{s} Q_\phi(s, a)}{s \sim \eta_\phi, a \sim \pi_\phi}.$$]{.fragment}
:::
:::

:::

# Comparison of the two formulations

:::small
::: fragment
::: definition
### Formulation via reward trajectories ([sampling version in blue]{style="color: blue;"}),
$$ \nablaphi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log\pi_\phi\agivenb{a_t}{s_t}\cbracket{\sum_{t'=t}^{T-1}r_{t'}}}{\tau\sim p_\phi(\tau)} \fragment{ \approx \textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t'=t}^{T-1} \nablaphi \log\,\pi_\phi\agivenb{a_{i,t}}{s_{i,t}}\cbracket{\sum_{t'=t}^{T-1} r_{i,t'} }}}. } $$
:::
:::

::: fragment
::: definition
### Derivation via formal relation between $V$ and $Q$ (Eq. \eqref{eq:AC_policy_gradient_Q_episodic}),
$$ \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \pi_\phi\agivenb{a_t}{s_t} Q_\phi(s_t, a_t)}{\tau\sim p_\phi(\tau)} \fragment{ \approx\textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\,\pi_\phi\agivenb{a_{i,t}}{s_{i,t}} Q_\phi(s_{i,t},a_{i,t})} }. } $$
:::
:::

::: columns-2-8
[**A much simpler way to arrive at this formulation**]{.fragment}

::: incremental
- Observe that the **reward-to-go** $\hat{Q}_{i,t}=\sum_{t'=t}^{T-1}r_{i,t'}$ is an **unbiased estimate** of $Q(s_t,a_t)$.
- Also, $\hat{Q}_{i,t}$ is a single-sample estimate, and summing over $N$ samples gives us a **batch estimate**.
- To reduce the variance, it would be a lot better to simply use the **true expected reward-to-go**:
$$\sum_{t'=t}^{T-1} \ExpCsub{r_t}{s_{t},a_{t}}{\pi_\phi} = Q_\phi(s_{i,t},a_{i,t}).$$
:::
:::
:::

# References

::: { #refs }
:::
