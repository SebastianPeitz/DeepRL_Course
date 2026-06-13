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

- Definitions and concepts from information theory
- Learning without a reward function by reaching goals
- A state-distribution-matching formulation of reinforcement learning
- Is coverage of valid states a good exploration objective?
- Beyond state coverage: covering the space of skills

# Unsupervised learning of diverse behaviours

*Preparing for an unknown future goal*

::: small

So far the agent explored in order to solve a *known* task. We now ask a different question: what if we want to acquire useful, diverse behaviour with *no reward function at all*?

::: definition
### Why learn behaviours without a reward?
- Learn **skills** without supervision, then compose them to accomplish goals.
- Learn reusable **sub-skills** for hierarchical reinforcement learning.
- **Explore** the space of possible behaviours to learn what the environment affords.
:::

**Example scenario.** A robot placed in a kitchen before any task is set can practise opening drawers, moving objects, and reaching many configurations. When a concrete goal arrives later --- "set the table" --- this repertoire makes learning fast.

[**Key question:** how can an agent prepare for an unknown future goal?]{style="color: red;"}

::: footer
Unsupervised skill/behaviour learning: [@eysenbach2018diayn; @gupta2018unsupervised].
:::
:::

------------------------------------------------------------------------------

# Information theory

------------------------------------------------------------------------------

# Information theory

*Entropy: how uncertain is a distribution?*

::: small

Let $p(x)$ be a distribution over a variable $x$ --- for us, typically an observation or state $x\in\Sc$.

::: definition
### Entropy
$$ H(p) = -\sum_x p(x)\log p(x) = \E_{x\sim p}\big[-\log p(x)\big]. $$
(For continuous $x$, this becomes differential entropy, obtained by replacing the sum with an integral. Differential entropy need not be nonnegative. The statement that entropy is maximized by a uniform distribution should be read for finite/discrete spaces or bounded support.)
:::

For a discrete distribution, entropy measures the average uncertainty, or "spread", of $p$: it is largest for a uniform distribution and zero for a deterministic one.

::: footer
Information-theoretic background: [@cover1999elements].
:::
:::


# Information theory

*Information gain, KL divergence, and mutual information*

::: small

::: definition
### Relative entropy (KL divergence)
$$ D_{\mathrm{KL}}(p\,\|\,q) = \E_{x\sim p}\!\left[\log\tfrac{p(x)}{q(x)}\right] \ge 0, \qquad D_{\mathrm{KL}}(p\,\|\,q)=0 \iff p=q. $$
How far a distribution $p$ is from a reference $q$ (not symmetric).
:::

::: definition
### Mutual information
$$ I(X;Y) = D_{\mathrm{KL}}\big(p(x,y)\,\|\,p(x)p(y)\big) = H(X)-H(X\mid Y) = H(Y)-H(Y\mid X). $$
The expected reduction in uncertainty about $X$ from observing $Y$.
:::

**Information gain** is mutual information used for exploration: the reduction in uncertainty about an unknown quantity $Z$ (e.g. the dynamics parameters $\theta$) from a new observation $Y$,
$$ \mathrm{IG} = H(Z) - \E_{Y}\big[H(Z\mid Y)\big] = I(Z;Y) = \E_{Y}\big[ D_{\mathrm{KL}}\big(p(Z\mid Y)\,\|\,p(Z)\big) \big]. $$

::: footer
Information-theoretic background: [@cover1999elements].
:::
:::

# Information theory

*State coverage and empowerment*

::: small

For a policy $\pi$, let $p_\pi(s)$ be the **state-marginal distribution** --- the probability of being in state $s$ while following $\pi$. (We write $p_\pi$, not $\pi(s)$, since $\pi\agivenb{a}{s}$ is reserved for the policy.)

::: definition
### State-marginal entropy = coverage
$$ H(p_\pi) = \E_{s\sim p_\pi}\big[-\log p_\pi(s)\big]. $$
High state-marginal entropy means the policy **covers** many states --- a natural reward-free exploration objective.
:::

::: definition
### Empowerment: an example of mutual information
The agent's "control authority" at a state $s$ is the channel capacity from actions to resulting states,
$$ \mathcal E(s) = \max_{\omega}\; I(S'; A \mid s), \qquad A\sim\omega,\;\; S'\sim p\agivenb{\cdot}{s,A}. $$
States with high empowerment are those from which the agent can reach many distinct outcomes.
:::

::: footer
Empowerment: [@klyubin2005empowerment; @salge2014empowerment].
:::
:::

------------------------------------------------------------------------------

# Learning without any reward functions

------------------------------------------------------------------------------

# Reinforcement learning with imagined goals

*Practise reaching goals you propose for yourself*

::: small

With no reward function, the agent **generates its own goals**: it fits a generative model (a VAE) to its observations, samples a goal, tries to reach it, and learns from the attempt.

::: definition
### The self-supervised goal-reaching loop
1. **Propose** a goal in latent space and decode it: $z_g \sim p(z)$, $\; x_g \sim p_\theta(x\mid z_g)$.
2. **Attempt** to reach it with a goal-conditioned policy $\pi\agivenb{a}{x,x_g}$.
3. **Update the policy** from the collected data (off-policy RL with goal relabelling).
4. **Refit the model** of observations: decoder $p_\theta(x\mid z)$ and encoder $q_\phi(z\mid x)$.
:::

The reward is the distance to the goal *in latent space*, $r = -\,\lVert z - z_g\rVert$ with $z=q_\phi$-encoding of $x$ --- a dense, self-supervised signal needing no hand-designed reward. Hindsight relabelling treats any reached state as a goal, so *every* attempt yields useful data.

::: footer
Reinforcement learning with imagined goals (RIG): [@nair2018visual].
:::
:::

# A latent generative model of observations

*Where the goals and the reward come from (VAE)*

::: small

Goals, the latent space, and the reward metric all come from a **variational autoencoder** trained on the agent's own observations.

::: definition
### Variational autoencoder (VAE)
A decoder $p_\theta(x\mid z)$ with prior $p(z)=\mathcal N(0,I)$ models observations; an encoder $q_\phi(z\mid x)$ approximates the intractable posterior. Both are trained by maximizing the ELBO,
$$ \max_{\theta,\phi}\; \E_{q_\phi(z\mid x)}\big[\log p_\theta(x\mid z)\big] - D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big), $$
using the reparameterization $z = \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon$, $\;\varepsilon\sim\mathcal N(0,I)$, for low-variance gradients.
:::

The VAE supplies everything the loop needs: a **prior** $p(z)$ to sample imagined goals, an **encoder** to embed observations, and a structured latent space where **distance is a usable reward**.

::: footer
Auto-encoding variational Bayes (VAE): [@kingma2013auto]; used for goal generation in RIG: [@nair2018visual].
:::
:::

# How do we get diverse goals?

*The objective: maximize the mutual information $I(S;G)$*

::: small

Let $G$ be the commanded goal (the latent $z_g$ / observation $x_g$) and $S$ the achieved final state. We want goals that are both **diverse** and **reachable** --- both follow from maximizing
$$ I(S;G) = H(G) - H(G\mid S) = H(S) - H(S\mid G). $$

::: columns-5-5
::: definition
### $H(G)$ --- coverage
High goal entropy means diverse goals spread over the state space. **Skew-Fit raises this term** by skewing the goal distribution toward uniform.
:::

::: definition
### $H(G\mid S)$ --- goal reaching
The goal-conditioned policy $\pi\agivenb{a}{x,x_g}$ is trained to reach its goal. As it improves, $S$ lands near $G$, so $G$ becomes (nearly) determined by $S$ and this term **shrinks**.
:::
:::

[**Together:** raising $H(G)$ (Skew-Fit) and lowering $H(G\mid S)$ (RL goal-reaching) both increase $I(S;G)$ --- good exploration *and* reliable goal reaching at once.]{style="color: red;"}

::: footer
Imagined goals (RIG): [@nair2018visual]; state-covering goals (Skew-Fit): [@pong2019skew].
:::
:::

# Skew-Fit: weighted MLE for coverage

*Up-weight rare states so the goal distribution spreads*

::: small

If the goal model is fit directly to visited observations, it proposes goals where the agent *already* spends time: common states dominate, rare states are almost never set as goals, and **coverage stalls**.

::: columns-5-5
::: definition
### Standard MLE
$$
\theta,\phi \leftarrow \arg\max_{\theta,\phi}\, \E_{\bar x\sim\Dc}\big[\log p_\theta(\bar x)\big].
$$
Fits the model to where the agent already goes.
:::

::: definition
### Weighted MLE (Skew-Fit)
$$
\theta,\phi \leftarrow \arg\max_{\theta,\phi}\, \E_{\bar x\sim\Dc}\big[w(\bar x)\log p_\theta(\bar x)\big],
\quad
w(\bar x) \propto p_{\theta_{\mathrm{old}}}(\bar x)^{\alpha}.
$$
For $\alpha\in[-1,0)$, the refit distribution behaves like $p_{\theta_{\mathrm{old}}}(\bar x)^{1+\alpha}$, flattened toward uniform ($\alpha=-1$ gives uniform).
:::
:::

[**Key result:** for any $\alpha\in[-1,0)$, skewed refitting does not decrease the entropy of the learned goal distribution --- it rises toward uniform over reachable states. So the agent keeps proposing novel, reachable goals and steadily expands the part of the state space it can reach.]{style="color: red;"}

::: footer
Weighted MLE and the entropy guarantee (Skew-Fit): [@pong2019skew]; builds on imagined goals: [@nair2018visual].
:::
:::

# Recap: intrinsic motivation as state-entropy maximization

*Reward visiting novel states; rarely-visited states are novel*

::: small

A state is novel if it is rarely visited, so add a bonus that is large where the state marginal $p_\pi(s)$ is small:
$$ r^{+}(s) = r(s) - \log p_\pi(s). $$

::: definition
### Alternating scheme
1. Update $\pi\agivenb{a}{s}$ to maximize $\E_\pi[r^{+}(s)]$.
2. Update the density $p_\pi(s)$ to fit the current state marginal.
:::

With no extrinsic reward ($r\equiv 0$), maximizing $\E_{s\sim p_\pi}[-\log p_\pi(s)] = H(p_\pi)$ is exactly **maximum state-entropy exploration** --- provably efficient, and solved by a *mixture* of policies.

::: footer
Provably efficient maximum-entropy exploration: [@hazan2019provably].
:::
:::

# State marginal matching

*Shape the state distribution toward a target, not just spread it out*

::: small

Maximum-entropy exploration drives $p_\pi(s)$ toward uniform. **State marginal matching (SMM)** generalizes this: make the policy's state distribution match a *target* density $p^\star(s)$ (uniform for pure coverage, or any prior over where the agent should spend time).

::: definition
### The SMM objective
$$ \min_{\pi}\; D_{\mathrm{KL}}\big(p_\pi(s)\,\|\,p^\star(s)\big) = \min_{\pi}\; \E_{s\sim p_\pi}\big[\log p_\pi(s) - \log p^\star(s)\big]. $$
:::

Equivalently, **maximize** the intrinsic reward
$$ r^{+}(s) = \log p^\star(s) - \log p_\pi(s), \qquad \E_\pi[r^{+}(s)] = -D_{\mathrm{KL}}\big(p_\pi\,\|\,p^\star\big). $$
For uniform $p^\star$ this reduces to the entropy bonus $-\log p_\pi(s)$ of the previous slide.

::: footer
State marginal matching: [@lee2019efficient].
:::
:::

# Solving SMM: a two-player game

*Why a single policy is not enough*

::: small

Naively maximizing $r^{+}(s)=\log p^\star(s)-\log p_\pi(s)$ with one policy does **not** match the marginal: the reward chases a moving target, because $p_\pi$ shifts as $\pi$ changes. This can make the policy oscillate instead of converging.

::: definition
### Fictitious play
1. At round $k$, train $\pi^k$ to maximize $\E_{\pi^k}\!\big[r^{+,k}(s)\big]$ against the current density.
2. Refit the density to **all states seen so far** using the historical mixture, not just the marginal of the newest policy $\pi^k$.
3. Return the **mixture policy** $\pi^{\mathrm{mix}}$: sample $k\in\{1,\dots,K\}$ uniformly at the start of each episode, then follow $\pi^k$.
:::

The matched solution $p_\pi(s)=p^\star(s)$ is the **Nash equilibrium** of this two-player game between the policy and the density model. In general, it is realized by a *mixture* of policies, not necessarily by the last policy alone.

::: footer
State marginal matching via fictitious play: [@lee2019efficient]; mixture-of-policies for max-entropy exploration: [@hazan2019provably].
:::
:::

# Is state entropy really a good objective?

*Coverage objectives all point toward state entropy*

::: small

The methods so far optimize closely related quantities, but the connection to state entropy depends on the setting and representation.

::: columns-5-5
::: definition
### Skew-Fit
Skew-Fit raises the entropy of the learned goal distribution by up-weighting rare, reachable observations. If the goal-conditioned policy reliably reaches its commanded goals, then achieved states become diverse as well. In that case,
$$
I(S;G)=H(S)-H(S\mid G)
$$
is large because $H(S)$ is large and $H(S\mid G)$ is small.
:::

::: definition
### State marginal matching
With a uniform target $p^\star$, minimizing
$$
D_{\mathrm{KL}}(p_\pi\,\|\,p^\star)
$$
is equivalent to maximizing the entropy of the policy's state marginal $p_\pi$. Thus SMM with a uniform target is also a state-coverage objective.
:::
:::

Both methods therefore motivate **spreading the state distribution over reachable, meaningful states**. But why is maximum coverage a good objective *before the task is known*?

[**Setup for a justification:** suppose that at test time an adversary picks the worst-case goal $G$. Which goal distribution should you train on?]{style="color: red;"}

::: footer
Max-entropy exploration: [@hazan2019provably]; unsupervised meta-RL: [@gupta2018unsupervised]; state-covering goals: [@pong2019skew].
:::
:::

# Maximum entropy is the robust answer

*Prepare equally for every goal you might be asked to reach*

::: small

Suppose that no downstream goal distribution is known in advance. Gupta et al. analyze this using **worst-case regret**: after unsupervised training, an adversary may choose the goal distribution that makes the learner perform worst.

::: definition
### Answer under goal-reaching assumptions
For goal-reaching tasks, the robust solution is to prepare uniformly over the goal states. Equivalently, use an exploration policy whose state marginal is as close as possible to uniform over the reachable goal states $\Sc_g$:
$$
p_\pi(s) \approx \mathrm{Unif}(\Sc_g).
$$
:::

A peaked training distribution leaves some goals under-practised, and the adversary can choose exactly those goals. Uniform coverage equalizes preparation across possible goals.

[**Caveat:** this is a minimax justification under the goal-reaching / trajectory-matching assumptions of the analysis, not a universal theorem that raw-state entropy is always optimal. With prior knowledge of likely goals, train on that non-uniform target; and if raw states contain irrelevant or uncontrollable factors, prefer meaningful representations or skill coverage.]{style="color: red;"}

::: footer
Worst-case task distribution / unsupervised meta-RL: [@gupta2018unsupervised]; provably efficient max-entropy exploration: [@hazan2019provably].
:::
:::

------------------------------------------------------------------------------

# Beyond state coverage: covering the space of skills

------------------------------------------------------------------------------

# Beyond state coverage

*Diverse states are useful, but diverse behaviours are richer*

::: small

State coverage asks the agent to visit many states. Skill discovery asks for something stronger: learn a family of behaviours that can be reused later.

::: definition
### Skill-conditioned policy
Sample a skill index $z\sim p(z)$ and condition the policy on it:
$$
\pi_\theta\agivenb{a}{s,z}.
$$
Keeping $z$ fixed during an episode gives one temporally extended behaviour, or **skill**.
:::

Reaching diverse goals is not the same as performing diverse tasks. Some useful behaviours are not naturally described by a single target state: patrolling, circling, balancing, pushing, avoiding, or moving with a particular style.

[**Core idea:** different skills should induce different state distributions $p_{\pi_\theta}(s\mid z)$, so that the skill $z$ can be inferred from the states the agent visits.]{style="color: red;"}

::: footer
Skill discovery without external reward: [@eysenbach2018diayn]; intrinsic options: [@gregor2016variational].
:::
:::

# Diversity-promoting rewards

*Reward states that reveal which skill was executed*

::: small

A skill-conditioned policy can be trained with an intrinsic reward that depends on how distinguishable its visited states are from those of other skills:
$$
\max_{\pi_\theta}\; \sum_z p(z)\,\E_{s\sim p_{\pi_\theta}(\cdot\mid z)}\big[r(s,z)\big].
$$

::: definition
### Discriminator / skill classifier
Train a discriminator $q_\phi\agivenb{z}{s}$ to predict which skill $z$ produced a visited state $s$. Then reward the policy for visiting states that make its skill easy to identify:
$$
r(s,z) \approx \log q_\phi\agivenb{z}{s}.
$$
:::

DIAYN uses the shaped pseudo-reward
$$
r_t = \log q_\phi\agivenb{z}{s_{t+1}} - \log p(z),
$$
where $z$ is sampled at the start of the episode. If $p(z)$ is uniform, the term $-\log p(z)$ is a constant baseline across skills.

[**Intuition:** a skill receives high reward when it visits states that other skills usually do not visit.]{style="color: red;"}

::: footer
DIAYN discriminator and pseudo-reward: [@eysenbach2018diayn].
:::
:::

# A connection to mutual information

*Make the skill predictable from the state*

::: small

The information-theoretic objective is to make states informative about the skill:
$$
I(S;Z)=H(Z)-H(Z\mid S).
$$

::: columns-5-5
::: definition
### $H(Z)$ --- use all skills
Choosing a high-entropy prior $p(z)$, usually uniform, prevents collapse to only one or two skills.
:::

::: definition
### $H(Z\mid S)$ --- make skills distinguishable
If $z$ can be predicted from $s$, then $H(Z\mid S)$ is small. Maximizing $\log q_\phi\agivenb{z}{s}$ trains the policy to visit states that reveal the skill.
:::
:::

The discriminator gives a variational lower bound:
$$
I(S;Z)
= \E_{z,s}\!\left[\log \frac{p\agivenb{z}{s}}{p(z)}\right]
\ge
\E_{z,s}\!\left[\log q_\phi\agivenb{z}{s} - \log p(z)\right].
$$
DIAYN also encourages each skill to remain stochastic by using a maximum-entropy policy, corresponding to the action-entropy term in its objective.

::: footer
Mutual-information skill objective and variational discriminator: [@eysenbach2018diayn]; variational intrinsic control principle: [@gregor2016variational].
:::
:::

# Variational intrinsic control

*Options are diverse if their final states identify them*

::: small

VIC uses the same mutual-information idea with explicit options. Let $\Omega$ denote an option, sampled from a controllability distribution $p^C\agivenb{\Omega}{s_0}$, and let the option-conditioned policy be
$$
\pi_\theta\agivenb{a}{s,\Omega}.
$$
Following option $\Omega$ from $s_0$ induces a distribution over final states $s_f$.

::: definition
### Intrinsic-control objective
Maximize the mutual information between options and final states:
$$
I(\Omega;s_f\mid s_0)=H(\Omega\mid s_0)-H(\Omega\mid s_0,s_f).
$$
Options are useful when different options reliably terminate in distinguishable final states.
:::

Because the true posterior $p\agivenb{\Omega}{s_0,s_f}$ is unknown, VIC learns an option-inference model $q_\phi\agivenb{\Omega}{s_0,s_f}$ and maximizes the variational reward
$$
r^I = \log q_\phi\agivenb{\Omega}{s_0,s_f} - \log p^C\agivenb{\Omega}{s_0}.
$$

::: footer
Variational intrinsic control and option-inference reward: [@gregor2016variational].
:::
:::

# DIAYN and VIC

*The same principle at different levels of temporal abstraction*

::: small

| Method | Latent variable | What predicts the latent variable? | Intrinsic signal |
| :-- | :-- | :-- | :-- |
| VIC | option $\Omega$ | final state $s_f$ via $q_\phi\agivenb{\Omega}{s_0,s_f}$ | $\log q_\phi\agivenb{\Omega}{s_0,s_f}-\log p^C\agivenb{\Omega}{s_0}$ |
| DIAYN | skill index $z$ | visited state $s$ via $q_\phi\agivenb{z}{s}$ | $\log q_\phi\agivenb{z}{s}-\log p(z)$ |

::: definition
### Shared idea
A behaviour is diverse if the latent variable that generated it can be inferred from the states it produces. Thus both methods maximize a variational lower bound on a mutual-information objective.
:::

This moves beyond raw state coverage: the agent is not merely trying to visit many states, but to learn a repertoire of distinguishable, temporally extended behaviours.

::: footer
VIC: [@gregor2016variational]. DIAYN: [@eysenbach2018diayn].
:::
:::

------------------------------------------------------------------------------

# References

::: small
::: { #refs }
:::
:::

------------------------------------------------------------------------------
