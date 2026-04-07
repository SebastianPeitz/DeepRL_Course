---
subtitle:    Monte Carlo Methods
feedback:
  deck-id:  'deeprl-monte-carlo'
...


------------------------------------------------------------------------------

# Monte Carlo simulation

------------------------------------------------------------------------------

# What does *Monte Carlo sampling* mean?

![Monte Carlo Port ([Source](https://commons.wikimedia.org/wiki/File:Monte_Carlo_Port_Hercules_b.jpg))](images/04-Monte-Carlo/Monte_Carlo_Port.jpg){ width=500px }

::: small
::: incremental
- We're trying to estimate the expected value of function of a random variable $x=X(\omega)$:
$$\Exp{f(X)} = \sum_{\omega\in\Omega} p(x) f(x) \qquad / \qquad \Exp{f(X)} = \int_{\omega\in\Omega} p(x) f(x) \dx. $$
- Not knowing $f$ or the random variable $X$, we need to rely on *experience*.
- Monte Carlo sampling: Having no prior knowledge, we can estimate the expected value from repeated experiments.
- This lecture: **Let's use this for (episodic, i.e., $T<\infty$) RL when we don't know the MDP!**
:::
:::

# Example: Estimating the value for $\pi$

::: columns-7-4
::: platzhalter

::: small
::: incremental
- The area of a quarter circle with radius $r=1$: $$A=\frac{1}{4} \pi r^2 = \frac{\pi}{4}.$$
- Randomly and uniformly place points in the unit square: $$(x,y)\sim U([0,1]^2).$$
- Probability of landing **inside** the quarter circle is the ratio of the areas: $$ p(x^2+y^2\leq 1) = \frac{A}{1^2} = \frac{\pi}{4}.$$
- Monte Carlo estimate: $$\pi = 4 \cdot \Exp{p(x^2+y^2\leq 1)}.$$
:::
:::
:::

![](images/04-Monte-Carlo/EstimatePi.svg){ .embed width=500px }

:::

------------------------------------------------------------------------------

# Monte Carlo prediction

------------------------------------------------------------------------------

# Monte Carlo prediction

::: small
::: columns-5-6
::: platzhalter
We begin by learning $V^\pi$ for a
given policy $\pi$.

::: incremental
- Recall that $V^\pi(s)$ is the expected return (expected cumulative future discounted reward) starting from that state: 
$$ V(s) = \ExpC{g_t}{s_t = s}=\ExpC{ \sum_{k=0}^T \gamma^k r_{t+k}}{s_t = s} .$$
- Simple way to estimate from experience: average the returns observed after visits to a state $s$. 
- As more returns are observed, the average should converge to the expected value.
- Two different approaches:
  - First-visit MC: approximate the return only from the first visit to a state $s$.
  - Every-visit MC: calculate an update for the $V(s)$ estimate for every visit to a state $s$.
:::
:::

::: fragment
::: {.definition}
### Algorithm: First-visit MC prediction for estimating $V \approx V^\pi$.

**initialize**

- $V(s)$ arbitrarily for $s \in \Sc$
- $\ell(s)$: an empty list of returns for all $s \in \Sc$

**for** $j = 1, 2, \ldots, J$ episodes:\
$\quad$ $g = 0$\
$\quad$ Generate a sequence following $\pi$:
$$((s_0,a_0,r_0),(s_1,a_1,r_1),\ldots,(s_{T_j-1},a_{T_j-1},r_{T_j-1}))$$
$\quad$ **for** $t \in \{T_j-1,T_j-2,T_j-3,\ldots,0\}$:\
$\quad\quad$ $g = \gamma g+ r_t$\
$\quad\quad$ **if** $s_t \notin \{s_0,\ldots,s_{t-1}\}$: $\qquad$ ([*that's the first-visit condition*]{style="color: red;"})\
$\quad\quad\quad$ Append $g$ to $\ell(s_t)$\
$\quad\quad\quad$ $V(s_t) = \mathsf{average}(\ell(s_t))$
:::
:::
:::

::: {.definition .fragment}
**Convergence?** [Yes, for every state $s\in\Sc$ that is visited infinitely often ($\rightarrow$ exploration!)]{.fragment}
:::

:::

# Advantages of MC methods

::: incremental
1. We can learn from experience, even if we do not know the model
  - actual experience,
  - simulated experience, if we have a model (but do not necessarily know the transition probability $\psprimesa$).
2. The estimates for each state are independent of each other\
$\Rightarrow$ we are not bootstrapping, i.e., building estimates based on other estimates.
3. As a consequence, we can estimate parts of the value function that are of particular interest.
:::

# Example: Gridworld

TBD

------------------------------------------------------------------------------

# Monte Carlo estimation of action values

------------------------------------------------------------------------------

# Monte Carlo estimation of action values

::: small
Is a model available (i.e., the MDP $(\Sc, \Ac, p, r, \gamma)$ is known)?

::: incremental
- Yes $\Rightarrow$ state values plus one-step predictions deliver an optimal policy.
- No $\Rightarrow$ action values are very useful to directly obtain optimal policies.
:::

[**Estimating $Q^\pi(s,a)$ using MC**:]{.fragment}

::: incremental
- Analog to the MC prediction algorithm for $V^\pi(s)$.
- Small variation: A visit refers to a state-action pair $(s,a)$.
- First-visit and every-visit variants exist
:::

[**Possible challenges?**]{.fragment}

::: incremental
- Certain state-action pairs $(s,a)$ may never be visited.
- Missing degree of exploration (in particular for deterministic $\pi$).
- Workaraound: **exploring starts** $\Rightarrow$ start episodes in random state-action pairs $(s,a)$.
:::

:::

------------------------------------------------------------------------------

# Monte Carlo control

------------------------------------------------------------------------------

# Monte Carlo control (1)

::: small
::: columns-6-5
::: platzhalter
We learned about the Generalized policy iteration (GPI) in the last lecture.

[Let's use the same procedure here:
$$
\pi_0 \stackrel{E}{\longrightarrow} Q^{\pi_0} \stackrel{I}{\longrightarrow} \pi_1 \stackrel{E}{\longrightarrow} Q^{\pi_1} \ldots \stackrel{I}{\longrightarrow} \pi^* \stackrel{E}{\longrightarrow} Q^*=Q^{\pi^*}.
$$]{ .fragment data-fragment-index=1 }

[**Policy improvement**: Make the policy greedy w.r.t. the current value function:]{ .fragment data-fragment-index=2 }

[
Since we have the *action-value function* $Q$, we don't need a model to construct the policy:
$$ \pi(s) = \arg\max_{a\in\Ac} Q(s,a). $$
]{ .fragment data-fragment-index=3 }
:::

::: platzhalter
![GPI [@Sutton1998]](images/03-dynamic-programming/SuttonBarto_GPI.svg){ width=500px }

\ 

[
  ![GPI for MC control [@Sutton1998]](images/04-Monte-Carlo/SuttonBarto_GPI_Q.svg){ width=200px}
]{ .fragment data-fragment-index=3 }
:::
:::
:::

# Monte Carlo control (2)
::: small
Even if we're operating in an unknown MDP, the policy improvement theorem remains valid for MC-based control due to the underlying MDP structure:

::: definition
**Policy improvement theorem for MC-based control** 
[$$
\begin{align*}
  Q^{\pi_k}(s,\pi_{k+1}(s)) &= Q^{\pi_k}\left(s,\arg\max_{a\in\Ac} Q^{\pi_k}(s,a)\right) \\
  &=\max_{a\in\Ac} Q^{\pi_k}(s,a) \\
  &\geq Q^{\pi_k}(s,\pi_k(s)) \\
  &\geq V^{\pi_k}(s).
\end{align*}
$$]{.math-incremental}
:::

::: incremental
- **Policy improvement**: Construct the greedy policy $\pi_{k+1}(s)$ w.r.t. the current approximation $Q^{\pi_k}$
- **Assumptions required**:
  - The episodes have epxloring starts. $\qquad\qquad\qquad\qquad$ *(We will consider this later)*
  - We are training on an infinite number of episodes.
:::
:::

# Monte Carlo control (3)
::: small
::: columns-7-3
::: {.definition}
### Algorithm: Monte Carlo ES (Exploring Starts) for estimating $\pi \approx \pi^*$.

**initialize**

- $\pi(s)$ arbitrarily for $s \in \Sc$
- $Q(s,a)$ arbitrarily for $s \in \Sc$, $a\in\Ac$
- $\ell(s,a)$: an empty list of returns for all $s \in \Sc$, $a\in\Ac$

**for** $j = 1, 2, \ldots, J$ episodes:\
$\quad$ $g = 0$\
$\quad$ Choose $s_0\in\Sc$ and $a_0\in\Ac$ randomly such that all pairs have probability $>0$\
$\quad$ Generate a sequence starting at $(s_0, a_0)$ and following $\pi$:
$$((s_0,a_0,r_0),(s_1,a_1,r_1),\ldots,(s_{T_j-1},a_{T_j-1},r_{T_j-1}))$$
$\quad$ **for** $t \in \{T_j-1,T_j-2,T_j-3,\ldots,0\}$:\
$\quad\quad$ $g = \gamma g+ r_t$\
$\quad\quad$ **if** $(s_t,a_t) \notin \{(s_0,a_0),\ldots,(s_{t-1},a_{t-1})\}$:\
$\quad\quad\quad$ Append $g$ to $\ell(s_t)$\
$\quad\quad\quad$ $Q(s_t,a_t) = \mathsf{average}(\ell(s_t,a_t))$\
$\quad\quad\quad$ $\pi(s_t) = \arg\max_{a\in\Ac}Q(s_t, a)$
:::

::: incremental
- It is clear that this algorithm cannot converge to a suboptimal policy
- Stability is only achieved when both $\pi$ and $Q$ are optimal
:::
:::
:::

# Example: Gridworld

TBD

------------------------------------------------------------------------------

# On-policy and off-policy learning

------------------------------------------------------------------------------

# On-policy versus off-policy (1)

::: small
::: columns-5-5
::: platzhalter
**On-Policy: "Learning by Doing"**

::: incremental
- The agent only learns from what it's currently doing.
- It evaluates and improves the exact same policy it is using to make decisions.
- For example,
  - if it is being cautious, it learns how to improve cautious behavior.
  - if it tries something risky and fails, it learns from that failure but can't easily "imagine" what would have happened if it had taken a different action earlier.
:::
:::

::: platzhalter
[**Off-Policy: "Learning by Observing"**]{.fragment}

::: incremental
- is more like a student watching an expert or looking at a variety of different actors once. 
- It can follow one path (the "behavior policy") while learning the best possible path (the "target policy").
- It separates "how I am acting" from "what I am learning."
- Benefit: It can learn from old data, from human demonstrations, or by watching other agents.
:::
:::
:::

\

[**Question**: Which class does the *Monte Carlo ES* algorithm belong to?]{.fragment}

[**Answer**: It is on-policy!]{.fragment}
[Each time we update our policy (the counter $j$ for the episodes), we collect the returns anew, following the current policy $\pi$.]{.fragment}

:::

::: fragment
::: footer
We will cover the topic on-policy versus off-policy in more detail later in this course.
:::
:::

# On-policy versus off-policy (2)
::: small

::: fragment
![](images/04-Monte-Carlo/On-off-policy.svg){ .embed width=800px }
:::

\

|  Feature | On-policy | Off-policy |
| :------- | :-------- | :--------- | 
| Learning Source |	Learns from its current actions only.	| Can learn from any data (past, random, or expert). |
| Exploration	| Often more stable but can get stuck in "safe" habits.	| More aggressive at finding the absolute best strategy. |
| Efficiency|	Lower. Throws away data once the policy changes.| Higher. Can "re-use" old experiences (Experience Replay). |

Table: Key differences at a glance:
:::

# $\epsilon$-greedy as an on-policy alternative (1)

::: small
**Motivating question**: How do we get rid of the restrictive (and often not achievable) requirement of *exploring starts (ES)*?

::: incremental
- We need to make sure that we visit all state-action pairs, irrespective of where we start!
- Exploration requirement: $$ \pias > 0 \quad \forall s\in\Sc,a\in\Ac. $$
- Policies of this type are called **soft**.
- The level of exploration should be tunable during the learning process.
:::

[**Where have we seen something like this before?**]{.fragment}
[$\Rightarrow$ In the multi-armed bandits lecture!]{.fragment}

::: fragment
::: definition
**$\epsilon$-greedy on-policy learning**

::: incremental
- With probability $\epsilon$, the agent’s choice (i.e., the policy output) is overwritten by a random action.
- Probability of all non-greedy actions: $\frac{\epsilon}{\abs{\Ac}}$.
- Probability of the greedy action: $1 - \epsilon + \frac{\epsilon}{\abs{\Ac}}$.
:::

:::
:::
:::

# $\epsilon$-greedy as an on-policy alternative (2)
::: small
::: {.definition}
### Algorithm: On-policy first-visit MC control (for $\epsilon$-soft policies).

**parameter:** small $\epsilon>0$ 

**initialize**

- Arbitrary $\epsilon$-soft policy $\pi$
- $Q(s,a)$ arbitrarily for $s \in \Sc$, $a\in\Ac$
- $\ell(s,a)$: an empty list of returns for all $s \in \Sc$, $a\in\Ac$

<!-- **for** $j = 1, 2, \ldots, J$ episodes:\
$\quad$ $g = 0$\
$\quad$ Choose $s_0\in\Sc$ and $a_0\in\Ac$ randomly such that all pairs have probability $>0$\
$\quad$ Generate a sequence starting at $(s_0, a_0)$ and following $\pi$:
$$((s_0,a_0,r_0),(s_1,a_1,r_1),\ldots,(s_{T_j-1},a_{T_j-1},r_{T_j-1}))$$
$\quad$ **for** $t \in \{T_j-1,T_j-2,T_j-3,\ldots,0\}$:\
$\quad\quad$ $g = \gamma g+ r_t$\
$\quad\quad$ **if** $(s_t,a_t) \notin \{(s_0,a_0),\ldots,(s_{t-1},a_{t-1})\}$:\
$\quad\quad\quad$ Append $g$ to $\ell(s_t)$\
$\quad\quad\quad$ $Q(s_t,a_t) = \mathsf{average}(\ell(s_t,a_t))$\
$\quad\quad\quad$ $\pi(s_t) = \arg\max_{a\in\Ac}Q(s_t, a)$ -->
:::
:::

------------------------------------------------------------------------------

# Off-policy Monte Carlo control

------------------------------------------------------------------------------

# Off-policy Monte Carlo control







------------------------------------------------------------------------------

# Summary / what you have learned

------------------------------------------------------------------------------

# Summary / what you have learned

# References

::: { #refs }
:::
