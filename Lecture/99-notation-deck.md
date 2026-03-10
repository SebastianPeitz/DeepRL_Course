---
subtitle:    Notation
feedback:
  deck-id:  'deeprl-notation'
...


# Letter styles

- Sets are written in caligraphic upper-case, $\Ac, \Sc, \Oc,\dots$.
- States, actions, or other elements of sets are written as lower-case letters, $s,a,o,\dots$.
- The letter $t$ is reserved for an iterator along a trajectories temporal dimension, its maximal value is given by $T$. Trajectories start at $t = 0$.
- We reserve $i,j,k,n,m$ to label iterators, maximal values achieved by them are denoted by the corresponding upper-case letter, i.e., $I,J,K,N,M$.

# MDPs
- An MDP is a tuple $(\Sc, \Ac, p, r, \gamma)$. It is consists of a set of states $\Sc$ and actions $\Ac$. The transition distribution (discrete case) or density (continuous case) is denoted by $p\agivenb{s'}{s,a}$. Finally, we have a reward function $r :  \Sc \times \Ac \rightarrow \R$ and a discount factor $\gamma \in [0,1]$. 
- For a given MDP, a policy $\pi\agivenb{a}{s}$ is either a distribution (discrete case) or a density (continuous case) over actions.
- Outputs are elements of some set $\Oc$, obtained by possibly implicit measurements of given state-action pairs, $o_t = h(s_t,a_t)$, where $h : \Sc \times \Ac \rightarrow \Oc$.

# Random variables
-  Random variables with distribution (discrete case) or density (continuous case) $q$ are declared by writing $x \sim q$. E.g., $s' \sim p\agivenb{\cdot}{s,a}$ or $a \sim \pi\agivenb{\cdot}{s}$.
-  Trajectories are sequences of state-action pairs $$
            \tau = (\tau_0, \tau_1,\dots) = ((s_0,a_0) , (s_1,a_1),\dots).
        $$
        If the trajectory is sampled according to some policy $\pi$ (i.e., $s_{t+1} \sim p\agivenb{\cdot}{s_t,a_t}$ and $a_t \sim \pi\agivenb{\cdot}{s_t}$ for $t \geq 1$), we get a new distribution/density, denoted by $p^\pi\agivenb{\tau}{s_0}$ which is conditioned on the initial state $s_0$.
        So $\tau \sim p^\pi\agivenb{\cdot}{s_0}$ denotes a random trajectory starting from $s_0$.
-  Instead of using index notation, one can also use $(s,a),(s',a'),(s'',a''),\dots$ to denote the start of a trajectory.

# Expectation and variance
-  Expectations are denoted as
        $$
            \E_{x \sim q} [f(x)] = \underbrace{\E_x[f(x)]}_{\text{if distr. clear}} = \underbrace{\E[f(x)]}_{\text{if R.V. clear}} = \begin{cases} 
                \sum_{x \in \Xc} q(x) f(x) &\text{discrete} \\
                \int_{\Xc} q(x) f(x) dx   &\text{continuous}
            \end{cases}
        $$
        Similarly, the variance is defined by 
        $$
            \Var_{x \sim q}[f(x)] = \Var_x[f(x)] = \Var[f(x)] =  \E_x[(f(x)-\E_{x'}[f(x')])^2].
        $$

# State value and state-action value
-  The state-value function is defined as
        $$
            V^\pi(s_0) := \E \left[ \sum_{t=0}^\infty \gamma^t r(s_t,a_t) \right], \quad\text{where}\quad s_{t+1} \sim p(\cdot |s_t,a_t), a_{t} \sim \pi(\cdot |s_t).
        $$
        alternatively, we can write 
        $$
            V^\pi(s_0) = \E_{\tau \sim p^\pi(\cdot | s_0)}\left[\sum_{t=0}^\infty \gamma^t r(\tau_t)\right].
        $$
- The state-action value function can be defined as
$$
    Q^\pi(s,a)  = \E_{s' \sim p(\cdot |s,a)}[ r(s,a) + \gamma V^\pi(s')].
$$

# Bellman equations
- We have the Bellman equations
\begin{align*}
    V^\pi(s) &= \E_{a \sim \pi(\cdot | s), s' \sim p(\cdot | s, a)}[ r(s,a) + \gamma V^\pi(s') ] &&=\E_{a,s'}[r(s,a)+\gamma V^\pi(s')], \\ 
    Q^\pi(s,a) &= \E_{s' \sim p(\cdot |s,a), a' \sim \pi(\cdot |s')}[ r(s,a) + \gamma Q(s',a')] &&= \E_{s',a'}[r(s,a) + \gamma Q(s',a')].
\end{align*}
-  Optimal value functions are denoted by $V^*$ or $Q^*$. They satisfy the Bellman optimality equations
        \begin{align*}
           V^*(s) &= \max_{a \in \Ac}  \E_{s'}[r(s,a)+\gamma V^*(s')]  \\
           Q^*(s,a) &= \E_{s'}[ r(s,a) + \gamma \max_{a' \in \Ac} Q^*(s',a')].
        \end{align*}
        An optimal policy is denoted by $\pi^*$, satisfying $V^* = V^{\pi^*}$.

# Trainable parameters and function approximation
-  Trainable parameters are denoted by $\theta, \psi, \phi$. Dependence on $\theta$ for function approximators is denoted via subscript notation, i.e., $V^\pi_\theta$, $Q^\pi_\theta$, $\pi_\theta$, etc. Differentiation w.r.t. a parameter is denoted by $\nabla_\theta (\cdot )$.
-  Sequences of value function (approximations) are denoted by $V_0,V_1,V_2,\dots$ when one has, for example, $V_n \rightarrow V^\pi$ as $n \rightarrow \infty$. (So iterator is in subscript notation.) Similarly, sequences of trainable parameters are denoted by $\theta_0, \theta_1,\theta_2,\dots$. 
-  Datasets are denoted by $\Dc$, containing either states, state-action pairs, or other content gathered while sampling from an MDP.
