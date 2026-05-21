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
  - Bottom-up derivation via the $Q$-function
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
|   8  | Deep $Q$-learning                                           |   $Q$-learning with neural networks     | 
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

# The $Q$-function version of the policy gradient

::: small
::: columns-5-5


::: incremental
- Recall that we started with the objective of maximizing the value directly by optimizing over our policy network weights:
[$$ \begin{align*} \phi^* &= \arg\max_{\phi}\Expsub{\sum_{t=0}^{T-1}r_t}{\tau\sim p_\phi(\tau)} \\ &= \arg\max_{\phi}\Expsub{\Vpiphi(s_0)}{s_0 \sim p_0}. \end{align*} $$]{.math-incremental}
:::

![Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse/).](images/09-policy-gradients/CNN-policy.svg){ width=700px }
:::

::: incremental
- Using the definition of the expectation of continuous random variables and the $\log$ derivative trick, we arrived at a formulation of the policy gradient from which we can sample using trajectory data ($\tau=(s_0,a_0,\ldots,s_{T-1},a_{T-1},s_T)$):
$$ \nablaphi L(\phi) = \Expsub{\cbracket{\sum_{t=0}^{T-1} \nablaphi \log\piphi\agivenb{a_t}{s_t}}\cbracket{\sum_{t=0}^{T-1}r_t}}{\tau\sim p_\phi(\tau)} \fragment{ \approx \frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\,\piphi\agivenb{a_{i,t}}{s_{i,t}}} \cbracket{\sum_{t=0}^{T-1}r_{i,t}}.}$$
- **Goal**: derive the policy gradient a second time, but this time using the $Q$-function instead of the return $\sum_{t=0}^{T-1}r_{i,t}$.
- **Starting point**: the RL objective and its gradient:
$$\begin{equation} L(\phi)=\Expsub{\Vpiphi(s_0)}{s_0 \sim p_0} \qquad \Rightarrow \qquad \nablaphi L(\phi)=\Expsub{\nablaphi \Vpiphi(s_0)}{s_0 \sim p_0} = \int_{\Sc} p_0(s) \nablaphi \Vpiphi(s) \ds. \label{eq:AC_RL_objective} \end{equation}$$
:::

:::

# Deriving the policy gradient (1)

::: small
::: incremental
- To derive the policy gradient, we need to take the **gradient of the value function**, i.e., $\nablaphi \Vpiphi(s)$.
- Our **goal is to introduce the $Q$-function** (we will see the reason for this later). [What's the relation between the $Q$-function and the value function for continuous actions?]{.fragment}
[$$ \underbrace{\Vpiphi(s) = \sum_{a\in\Ac} \piphi\agivenb{a}{s} \Qpiphi(s,a)}_{\text{finite }\Ac} \fragment{ \qquad \text{vs.}\qquad \underbrace{\Vpiphi(s) = \int_\Ac \piphi\agivenb{a}{s} \Qpiphi(s,a)\dint{a}}_{\text{continuous }\Ac}. } $$]{.fragment}
- Now let's **take the derivative**! [Both $\pi$ and $\Qpiphi$ depend on $\phi$ $\Rightarrow$ product rule [($\frac{d}{dx} (f(x)\cdot g(x)) = f(x) \frac{dg}{dx} + \frac{df}{dx} g(x)$)]{style="color: gray;"}:]{.fragment}
[$$\begin{align*} 
\nablaphi \Vpiphi(s) &= \nablaphi \cbracket{\int_\Ac \piphi\agivenb{a}{s} \Qpiphi(s,a)\dint{a}} \fragment{ = \int_\Ac \nablaphi \rbracket{\piphi\agivenb{a}{s} \Qpiphi(s,a)}\dint{a} } \fragment{ = \int_\Ac \nablaphi\piphi\agivenb{a}{s} \Qpiphi(s,a) + \piphi\agivenb{a}{s} \nablaphi \Qpiphi(s,a)\dint{a}} \\
&=  \underbrace{\int_\Ac\nablaphi\piphi\agivenb{a}{s} \Qpiphi(s,a)\dint{a}}_{= \xi(s)} + \int_\Ac\piphi\agivenb{a}{s} \nablaphi \Qpiphi(s,a)\dint{a}
\end{align*}$$]{.math-incremental}
- In total, we obtain the following **expression for $\nablaphi \Vpiphi(s)$**:
$$\begin{equation}
\nablaphi \Vpiphi(s) = \xi(s) + \int_\Ac\piphi\agivenb{a}{s} \nablaphi \Qpiphi(s,a)\dint{a}.
\label{eq:AC_grad_V}\end{equation}$$

:::
:::

# Deriving the policy gradient (2)

::: small
::: incremental
- Before we continue with \eqref{eq:AC_grad_V}, let's study $\Qpiphi(s,a)$ in some more detail, with the **goal to find a Bellman recursion** formula:
[$$\begin{equation}  \Qpiphi(s, a) = r + \int_{\Sc} \psprimesa \Vpiphi(s') \dint{s'}. \label{eq:AC_Q_definition} \end{equation}$$]{.fragment}
- Since neither $r$ nor $\psprimesa$ depend on $\phi$, taking the gradient of \eqref{eq:AC_Q_definition} yields
$$ \begin{equation} \nablaphi \Qpiphi(s, a) = \int_{\Sc} \psprimesa \nablaphi \Vpiphi(s') \dint{s'}. \label{eq:AC_grad_Q} \end{equation} $$
- We can substitute this expression into \eqref{eq:AC_grad_V} and swap integrals due to linearity:
$$ \nablaphi \Vpiphi(s) = \xi(s) + \int_\Ac\piphi\agivenb{a}{s} \underbrace{\rbracket{\int_{\Sc} \psprimesa \nablaphi \Vpiphi(s') \dint{s'}}}_{\eqref{eq:AC_grad_Q}}\dint{a} \fragment{ = \xi(s) + \int_{\Sc} \underbrace{\rbracket{\int_\Ac\piphi\agivenb{a}{s} \psprimesa \dint{a}}}_{=p\agivenb{s \to s'}{1,\piphi}} \nablaphi \Vpiphi(s') \dint{s'}.}$$
- Here, $p\agivenb{s \to s'}{1,\piphi}$ denotes the probability of moving from $s$ to $s'$ in exactly one step, given the policy $\piphi$:
  $$ \begin{equation} \nablaphi \Vpiphi(s) = \xi(s) + \int_{\Sc} p\agivenb{s \to s'}{1,\piphi} \nablaphi \Vpiphi(s') \dint{s'}. \label{eq:AC_grad_V_bellman} \end{equation}$$
:::
:::

# Deriving the policy gradient (3)

::: small
::: incremental
- Equation \eqref{eq:AC_grad_V_bellman} is a Bellman recursion equation, and we can insert the same expression at $s'$ on the right:
$$ \nablaphi \Vpiphi(s) = \xi(s) + \int_{\Sc} p\agivenb{s \to s'}{1,\piphi} \underbrace{\rbracket{\xi(s') + \int_{\Sc} p\agivenb{s' \to s''}{1,\piphi} \nablaphi \Vpiphi(s'') \dint{s''}}}_{= \nablaphi \Vpiphi(s')} \dint{s'}. $$
- We can unroll this expression for an entire trajectory $\tau$ of length $T$. [After some resorting, we obtain:
$$ \nablaphi \Vpiphi(s) = \xi(s) + \int_{\Sc} p\agivenb{s \to s'}{1,\piphi} \xi(s') \dint{s'} + \int_{\Sc} p\agivenb{s \to s''}{2,\piphi}\xi(s'') \dint{s''} + \ldots$$]{.fragment}
- We can write this cleanly as a sum over all future timesteps $t$:
<!-- $$ \begin{equation} \nablaphi \Vpiphi(s) = \int_{\Sc} \sum_{t=0}^{\infty} p\agivenb{s \to x}{t, \piphi} \xi(x) \dx. \label{eq:AC_grad_V_recursion} \end{equation} $$ -->
$$ \begin{equation} \nablaphi \Vpiphi(s) = \int_{\Sc} \sum_{t=0}^{T-1} p\agivenb{s \to x}{t, \piphi} \xi(x) \dx. \label{eq:AC_grad_V_recursion} \end{equation} $$
[Note: the previously separate case $\xi(s)$ is included, i.e., $\int_{\Sc} p\agivenb{s \to s}{0, \piphi} \xi(x)\dx =\xi(s)$.]{.fragment}
:::
:::

# Deriving the policy gradient (4)

::: small
::: incremental
- We can now insert \eqref{eq:AC_grad_V_recursion} into the gradient of our RL objective (Eq. \eqref{eq:AC_RL_objective}):
<!-- $$ \nablaphi L(\phi)=\Expsub{\nablaphi \Vpiphi(s_0)}{s_0 \sim p_0} \fragment{ = \int_{\Sc} p_0(s) \nablaphi \Vpiphi(s) \ds  }\fragment{ = \int_{\Sc} p_0(s) \rbracket{\int_{\Sc} \sum_{t=0}^{\infty} p\agivenb{s \to x}{t, \piphi} \xi(x) \dx} \ds. }$$ -->
$$ \nablaphi L(\phi)=\Expsub{\nablaphi \Vpiphi(s_0)}{s_0 \sim p_0} \fragment{ = \int_{\Sc} p_0(s) \nablaphi \Vpiphi(s) \ds  }\fragment{ = \int_{\Sc} p_0(s) \rbracket{\int_{\Sc} \sum_{t=0}^{T-1} p\agivenb{s \to x}{t, \piphi} \xi(x) \dx} \ds. }$$
- By changing the order of integration, we isolate the total expected time spent in state $x$:
<!-- $$\nablaphi L(\phi) = \int_{\Sc} \rbracket{ \int_{\Sc} p_0(s) \sum_{t=0}^{\infty} p\agivenb{s \to x}{t, \piphi} \ds} \xi(x) \dx.$$ -->
$$\nablaphi L(\phi) = \int_{\Sc} \rbracket{ \int_{\Sc} p_0(s) \sum_{t=0}^{T-1} p\agivenb{s \to x}{t, \piphi} \ds} \xi(x) \dx.$$
- The term inside the brackets is the **undiscounted state visitation frequency** (or the expected number of times state $x$ is visited in an episode), which we denote as $\eta_\phi(s)$:
<!-- $$\begin{equation} \eta_\phi(s) = \sum_{t=0}^{\infty} p\agivenb{s_t = s}{p_0, \piphi}. \label{eq:AC_state_visitation_measure} \end{equation}$$ -->
$$\begin{equation} \eta_\phi(s) = \sum_{t=0}^{T-1} p\agivenb{s_t = s}{\piphi}. \label{eq:AC_state_visitation_measure} \end{equation}$$
- This simplifies our expression to:
$$\nablaphi L(\phi) = \int_{\Sc} \eta_\phi(s) \xi(s) \ds.$$
:::
:::

# Deriving the policy gradient (5)

::: small
::: columns-6-4
::: incremental
- Reintroducing the full definition of $\xi(s) = \int_\Ac\nablaphi\piphi\agivenb{a}{s} \Qpiphi(s,a)\dint{a}$ (and keeping $s$ as our state variable):
$$\nablaphi L(\phi) = \int_{\Sc} \eta_\phi(s) \int_{\Ac} \nablaphi \piphi\agivenb{a}{s} \Qpiphi(s, a) \dint{a}\ds.$$
- Apply the $\log$-derivative identity:
$$\nabla_\phi L(\phi) = \textcolor{blue}{\int_{\Sc} \eta_\phi(s)} \textcolor{red}{\int_{\Ac} \piphi\agivenb{a}{s}} \nabla_\phi \log \piphi\agivenb{a}{s} \Qpiphi(s, a) \textcolor{red}{\dint{a}} \textcolor{blue}{\ds}.$$
- In conclusion, the gradient can be expressed using the unnormalized state visitation measure $\eta_\phi(s)$:
<!-- $$\begin{equation} \nabla_\phi L(\phi) = \Expsub{\nablaphi \log \piphi\agivenb{a}{s} \Qpiphi(s, a)}{s \sim \eta_\phi, a \sim \piphi}. \label{eq:AC_policy_gradient_Q} \end{equation}$$ -->
$$\begin{equation} \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} \Qpiphi(s_t, a_t)}{\tau\sim p_\phi(\tau)}. \label{eq:AC_policy_gradient_Q_episodic} \end{equation}$$
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

$$\nabla_\phi \piphi\agivenb{a}{s} = \piphi\agivenb{a}{s} \nabla_\phi \log \piphi\agivenb{a}{s}$$
:::
:::
:::

::: fragment
::: definition
**Note**: In the infinite-horizon case, \eqref{eq:AC_state_visitation_measure} becomes $\eta_\phi(s) = \sum_{t=0}^{\infty} p\agivenb{s_t = s}{p_0, \piphi}$ [$\Rightarrow$ Equation \eqref{eq:AC_policy_gradient_Q_episodic} becomes
<!-- $$\begin{equation} \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} \Qpiphi(s_t, a_t)}{\tau\sim p_\phi(\tau)}. \label{eq:AC_policy_gradient_Q_episodic} \end{equation}$$]{.fragment} -->
$$\nabla_\phi L(\phi) = \Expsub{\nablaphi \log \piphi\agivenb{a}{s} \Qpiphi(s, a)}{s \sim \eta_\phi, a \sim \piphi}.$$]{.fragment}
:::
:::

:::

# Comparison of the two formulations

::: small
Here's the policy gradient theorem in the two versions we have derived ([Sampling versions in blue]{style="color: blue;"}). 

::: fragment
::: definition
### Policy gradient theorem -- formulation via reward trajectories
$$ \nablaphi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log\piphi\agivenb{a_t}{s_t}\cbracket{\sum_{t'=t}^{T-1}r_{t'}}}{\tau\sim p_\phi(\tau)} \fragment{ \approx \textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t'=t}^{T-1} \nablaphi \log\,\piphi\agivenb{a_{i,t}}{s_{i,t}}\cbracket{\sum_{t'=t}^{T-1} r_{i,t'} }}}. } $$
:::
:::

::: fragment
::: definition
### Policy gradient theorem -- formulation using the $Q$-function
$$ \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} \Qpiphi(s_t, a_t)}{\tau\sim p_\phi(\tau)} \fragment{ \approx\textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\,\piphi\agivenb{a_{i,t}}{s_{i,t}} \Qpiphi(s_{i,t},a_{i,t})} }. } $$
:::
:::

::: columns-1-8
[**Questions**:]{.fragment}

::: platzhalter
::: incremental
1. Which formulation is more accurate? [$\Rightarrow$ The second version has smaller variance! (next slide)]{.fragment}
2. Which type of sampling can we use for the formulations?\
[$\Rightarrow$ First one: Monte Carlo sampling.]{.fragment}\
[$\Rightarrow$ Second: Temporal Difference learning.]{.fragment}
3. How can we even sample from $\Qpiphi$ if it is not explicitly defined? [$\Rightarrow$ We'll see soon ("*critic*").]{.fragment}
:::
:::
:::
:::

# A much simpler way to arrive at this formulation

::: small
::: columns-3-6
\ 

::: incremental
- Observe that the **reward-to-go** $\hat{Q}_{i,t}=\sum_{t'=t}^{T-1}r_{i,t'}$ is an **unbiased estimate** of $Q(s_t,a_t)$.
- But: $\hat{Q}_{i,t}$ is a **single-sample estimate** of the reward-to-go.
:::
:::

::: columns-3-6

![Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse/).](images/10-actor-critic/reward-to-go.svg){ .embed width=400px }

::: platzhalter
::: incremental
- To reduce the variance, it would be a lot better to simply use the **true expected reward-to-go**:
$$\sum_{t'=t}^{T-1} \ExpCsub{r_t}{s_{t},a_{t}}{\piphi} = \Qpiphi(s_{i,t},a_{i,t}).$$
- Replacing $\sum_{t'=t}^{T-1}r_{i,t'}$ by $Q^\pi(s_{i,t},a_{i,t})$ yields
$$ \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} Q^\pi(s_t, a_t)}{\tau\sim p_\phi(\tau)}. $$
:::

::: fragment
::: definition
**Note**: The inconsistency between the supersripts $\pi$ and $\pi_\phi$ for $Q$ is not accidental. We will soon see what's the reason behind this.
:::
:::
:::
:::
:::


# What about the baseline?

::: small
::: incremental
- We would like to reduce the variance of the new formulation using a baseline $b$:
$$ \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} \rbracket{Q^\pi(s_t, a_t) - b}}{\tau\sim p_\phi(\tau)} \approx\textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\,\piphi\agivenb{a_{i,t}}{s_{i,t}} \rbracket{Q^\pi(s_{i,t},a_{i,t}) - b} } }. $$
- How do we choose $b$?
  - A natural choice (analogous to the baseline for version one) might be the average the $Q^\pi$ value over our samples: $$b=\frac{1}{N} Q^\pi(s_{i,t},a_{i,t}).$$
  - Unfortunately, an action-dependent average leads to a biased policy gradient (i.e., leads to the wrong gradient) :weary:
- An alternative (and even lower-variance) approach:
  - Average over **all the possibilities starting in the state $s_t$** (not just in the time step $t$)!
  - How do we do this? 
  $$\fragment{b = \Expsub{Q^\pi(s_t,a_t)}{a_t\sim \pi\agivenb{\cdot}{s_t}} } \fragment{ = V^\pi(s_t). } $$

:::

:::

# A very good baseline for policy gradients

::: small
::: incremental
- If we're using the value function $V^\pi(s_t)$ as our baseline, we obtain the following expression:
$$ \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} \rbracket{Q^\pi(s_t, a_t) - V^\pi(s_t)}}{\tau\sim p_\phi(\tau)} \approx\textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\,\piphi\agivenb{a_{i,t}}{s_{i,t}} \rbracket{Q^\pi(s_{i,t},a_{i,t}) - V^\pi(s_{i,t})} } }. $$ 
- This one has a very intuitive interpretation:
  - If $Q^\pi(s,a) - V^\pi(s) > 0$, then $a$ is **better than the average action** according to our current policy $\pi$.
  - If $Q^\pi(s,a) - V^\pi(s) < 0$, then $a$ is **worse than the average action**.
:::

::: fragment
::: definition
### Policy gradient with advantage function

The *advantage function* $A^\pi(s,a)$ describes how much better the action $a$ is over the average action when following $\pi$:
$$ A^\pi(s,t) = Q^\pi(s,t) - V^\pi(s,t). $$
<!-- $$ A^\pi(s,t) = Q^\pi(s,t) - V^\pi(s,t) \fragment{ = \ExpCsub{\sum_{k=0}^{\infty}\gamma^k r_{t+k}}{s_t=s,a_t=a}{\pi} - \ExpCsub{\sum_{k=0}^{\infty}\gamma^k r_{t+k}}{s_t=s}{\pi}. } $$ -->

[*Policy gradient with value baseline*: "maximize the policy likelihood, weighted by the advantage function":
$$ \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} A^\pi(s_t, a_t)}{\tau\sim p_\phi(\tau)} \fragment{ \approx\textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\,\piphi\agivenb{a_{i,t}}{s_{i,t}} A^\pi(s_{i,t},a_{i,t})} }. } $$ ]{.fragment}
:::
:::

:::

# The actor-critic framework

::: small
::: columns-6-4
::: incremental
- Remember the inconsistency (supersripts $\pi$ and $\pi_\phi$ for $Q$) in our "simple derivation" of the policy gradient formulation,
$$ \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} Q^\pi(s_t, a_t)}{\tau\sim p_\phi(\tau)}? $$
:::

::: incremental
- As well as Question 3. when we discussed the difference between the $r$-based and the $Q$-based versions of the policy gradient: "How can we even sample from $\Qpiphi$ if it is not explicitly defined"?
:::
:::

<!-- ::: fragment
**Fix: re-introduce value function approximation!**
::: -->

::: columns-6-4
::: fragment
::: definition
### The actor-critic framework

[**Actor**: Neural network with parameters $\phi$ approximates the policy: 
$$\pi_\phi \approx \pi.$$]{.fragment}

[**Critic**: Approximates $V^\pi$ / $Q^\pi$ / $A^\pi$ and *criticizes* the actor:
$$ V_\theta \approx V^\pi, \qquad Q_\theta \approx Q^\pi, \qquad A_\theta \approx A^\pi.$$]{.fragment}

[**Policy gradient** using, e.g., $Q_\theta$:
$$\nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} Q_\theta(s_t, a_t)}{\tau\sim p_\phi(\tau)}.$$]{.fragment}

:::
:::

[![Inspired by Sergey Levine's [CS285 lecture](https://rail.eecs.berkeley.edu/deeprlcourse/).](images/10-actor-critic/Concept-5.svg){ .embed width=430px }]{.fragment}

:::
:::

# An actor-critic algorithm (1)

::: small
::: incremental
- Now consider the baseline version where we use the advantage function $A^\pi$ to weight the log gradient update:
$$\nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} A^\pi(s_t, a_t)}{\tau\sim p_\phi(\tau)} \approx \textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\,\piphi\agivenb{a_{i,t}}{s_{i,t}} A^\pi(s_{i,t},a_{i,t})} }. $$
- **Key questions**: 
  - What should we fit? $V^\pi$, $Q^\pi$ or $A^\pi$?
  - To which target should we fit?
- Recall the definition of the $Q$-function (with $\gamma=1$): 
[$$\begin{align*} Q^\pi(s_t, a_t) &= \ExpCsub{r_{t}+ Q^\pi(s_{t+1}, a_{t+1})}{s_t,a_t}{\pi} \fragment{ = \ExpCsub{r_{t}}{s_t,a_t}{\pi} + \ExpCsub{\sum_{t'=t+1}^{T-1} r_{t'}}{s_t,a_t}{\pi}}  \fragment{ = \ExpCsub{r_{t}}{s_t,a_t}{\pi} + \underbrace{\sum_{t'=t+1}^{T-1} \ExpCsub{r_{t'}}{s_t,a_t}{\pi}}_{\fragment{ =V^\pi(s_{t+1}) }} } \\ 
&= \ExpCsub{r_{t}}{s_t,a_t}{\pi} + \Expsub{V^\pi(s_{t+1})}{s_{t+1}\sim p\agivenb{\cdot}{s_t,a_t}}.
\end{align*}$$]{.math-incremental}
- Next, we're going to make *two assumptions* to turn this into an algorithm in which we can use experience.
:::
:::

# An actor-critic algorithm (2)

::: small
Two assumptions that lead to the following approximation:
$$Q^\pi(s_t, a_t) = \ExpCsub{r_{t}}{s_t,a_t}{\pi} + \Expsub{V^\pi(s_{t+1})}{s_{t+1}\sim p\agivenb{\cdot}{s_t,a_t}} \fragment{ \approx r_{t} + V^\pi(s_{t+1}). }$$

::: incremental
1. We just take the next reward we experience, instead of considering the expectation: 
$$r_{t} \approx \ExpCsub{r_{t}}{s_t,a_t}{\pi}.$$
  [$\circ$ This is often very reasonable.]{.fragment}\
  [$\circ$ If the rewards depends deterministically on the state $s_t$ and the action $a_t$, then this isn't even an approximation.]{.fragment}\
  [:bulb: In this case, people often use the notation $r(s,a)$ to denote that the reward is a deterministic function.]{.fragment}
2. Instead of considering the expectation over all next possible states, we take the value of the next state we see in our executed trajectory as a representative:
$$ \Expsub{V^\pi(s_{t+1})}{s_{t+1}\sim p\agivenb{\cdot}{s_t,a_t}} \approx V^\pi(s_{t+1}). $$
  [$\circ$ This is an assumption that introduces bias.]{.fragment}\
  [$\circ$ But it is often still a very reasonable assumption.]{.fragment}\
  [$\circ$ The strong reduction in variance justifies such a biased (and simple-to-assess) estimator.]{.fragment}
:::
:::

# An actor-critic algorithm (3)

::: small
::: columns-6-4
::: incremental
- Now that we have our approximation $Q^\pi(s_t, a_t) \approx r_{t} + V^\pi(s_{t+1})$, what to do with it? 
- Let's take another look at the *advantage function* $A^\pi$:
$$\fragment{ A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t) } \fragment{ \approx r_{t} + V^\pi(s_{t+1}) - V^\pi(s_t) } \fragment{ = \hat{A}^\pi(s_t,a_t). }$$
- So let's just approximate $V^\pi \approx V_\theta$!
- How do we do this? $\Rightarrow$ We have seen this already.\
[$\circ$ Monte Carlo estimates from entire trajectories:
  $$ L(\theta) = \sum_{k=1}^N \big(g_k - V_\theta(s_k)\big)^2 \fragment{ \quad \Rightarrow \quad \theta \gets \theta + \alpha\rbracket{g - V_\theta(s)} \nablatheta V_\theta(s). } $$]{.fragment}
[$\circ$ TD estimates from single-sample transitions using semi-gradients:]{.fragment}
  [$$\begin{align*}  L(\theta) &= \sum_{k=1}^N \big(r_k + V_\theta(s_{k+1}) - V_\theta(s_k)\big)^2 \\
  \Rightarrow \quad \theta &\gets \theta + \alpha\rbracket{r + V_\theta(s') - V_\theta(s)} \nablatheta V_\theta(s).  \end{align*}$$]{.math-incremental}
:::

::: fragment
::: definition
### Algorithm: Batch actor-critic

::: incremental
1. Sample $\set{s_i,a_i,s'_i}_{i=1}^N$ using $\pi_\phi\agivenb{a}{s}$.
2. Fit $V_\theta(s)$ to the sampled rewards.
3. Compute advantage: $$\hat{A}_\theta(s_i,a_i) = r_{i} + V_\theta(s'_i) - V_\theta(s_i).$$
4. Gradient: $$\nablaphi L(\phi) \approx \frac{1}{N} \sum_{i=1}^N \nablaphi \log\pi_\phi\agivenb{a_{i,t}}{s_{i,t}} \hat{A}_\theta(s_i,a_i).$$
5. Gradient ascent: $\phi \gets \phi + \alpha \nablaphi L(\phi)$.
:::
:::
:::
:::
:::

# Re-introducing the discount factor (1)

::: small
::: incremental
- For "simplicity", we have disregarded the discount factor $\gamma$ for now (i.e., $\gamma=1$).
- However, the derivations can be done with a discount factor in a very similar fashion.
- The changes we observe occur in two places:
  1. The obvious one: Our $Q$-function is now the discounted version: $$Q^\pi(s, a) = \ExpCsub{r + \gamma Q^\pi(s', a')}{s,a}{\pi}.$$
  [:bulb: The same obviously holds for $V$ and $A$.]{.fragment}
  2. The subtle one: The state distribution ("state visitation probability") changes: $$\eta_\phi(s) = \sum_{t=0}^{T-1} \gamma^t p\agivenb{s_t = s}{\piphi}.$$
  [:bulb: The proof is quite technical and requires swapping the sum over the time steps $t$ and the integration over $\Sc$ in the policy gradient derivation.]{.fragment}
- When sampling, point 2. is taken care of automatically, as we will sample according to this new distribution automatically. [We thus obtain the same formulation (here using $A$):
$$ \nabla_\phi L(\phi) = \Expsub{\sum_{t=0}^{T-1} \nablaphi \log \piphi\agivenb{a_t}{s_t} A^\pi(s_t, a_t)}{\tau\sim p_\phi(\tau)} \fragment{ \approx\textcolor{blue}{\frac{1}{N} \sum_{i=1}^N \cbracket{\sum_{t=0}^{T-1} \nablaphi \log\,\piphi\agivenb{a_{i,t}}{s_{i,t}} A^\pi(s_{i,t},a_{i,t})} }, } $$]{.fragment}
[where point 1. is "hidden" in $A^\pi$ and point 2. is "hidden" in the distribution $\tau\sim p_\phi(\tau)$.]{.fragment}
:::
:::



# References

::: { #refs }
:::
