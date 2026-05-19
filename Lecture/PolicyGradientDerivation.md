When we set the discount factor to one ($\gamma = 1$), we are dealing with the episodic or undiscounted formulation of reinforcement learning.

The overall structure of the derivation remains very similar, but the definitions of the value functions, the recursive expansion, and the final state distribution change accordingly. Here is the step-by-step derivation.

---

### 1. Objective Function and Setup

With $\gamma = 1$, the objective function $J(\theta)$ is the expected total undiscounted return from the initial state distribution $\rho_0$:


$$J(\theta) = \mathbb{E}_{s_0 \sim \rho_0} [V^{\pi_\theta}(s_0)]$$

The state-value function is the continuous integral over the action space:


$$V^{\pi_\theta}(s) = \int_{\mathcal{A}} \pi_\theta(a|s) Q^{\pi_\theta}(s, a) \, da$$

### 2. Gradient of the Value Function

Taking the gradient of $V^{\pi_\theta}(s)$ with respect to $\theta$ using the product rule:

$$\nabla_\theta V^{\pi_\theta}(s) = \int_{\mathcal{A}} \left[ \nabla_\theta \pi_\theta(a|s) Q^{\pi_\theta}(s, a) + \pi_\theta(a|s) \nabla_\theta Q^{\pi_\theta}(s, a) \right] da$$

### 3. Expanding the Undiscounted $Q$-Function Gradient

With $\gamma = 1$, the continuous Bellman equation for the action-value function is:


$$Q^{\pi_\theta}(s, a) = R(s, a) + \int_{\mathcal{S}} P(s'|s, a) V^{\pi_\theta}(s') \, ds'$$

Since the reward $R(s,a)$ and transition dynamics $P(s'|s,a)$ do not depend on the policy parameters $\theta$, taking the gradient gives:


$$\nabla_\theta Q^{\pi_\theta}(s, a) = \int_{\mathcal{S}} P(s'|s, a) \nabla_\theta V^{\pi_\theta}(s') \, ds'$$

### 4. Recursive Substitution

Substituting $\nabla_\theta Q^{\pi_\theta}(s, a)$ back into our gradient expression for $V^{\pi_\theta}(s)$:

$$\nabla_\theta V^{\pi_\theta}(s) = \int_{\mathcal{A}} \nabla_\theta \pi_\theta(a|s) Q^{\pi_\theta}(s, a) \, da + \int_{\mathcal{A}} \pi_\theta(a|s) \left[ \int_{\mathcal{S}} P(s'|s, a) \nabla_\theta V^{\pi_\theta}(s') \, ds' \right] da$$

Let $\phi(s) = \int_{\mathcal{A}} \nabla_\theta \pi_\theta(a|s) Q^{\pi_\theta}(s, a) \, da$. By swapping the order of integration for the second term, we get:

$$\nabla_\theta V^{\pi_\theta}(s) = \phi(s) + \int_{\mathcal{S}} \left[ \int_{\mathcal{A}} \pi_\theta(a|s) P(s'|s, a) \, da \right] \nabla_\theta V^{\pi_\theta}(s') \, ds'$$

The inner integral represents the 1-step state transition probability under the policy, $P(s \rightarrow s' | 1, \pi_\theta)$:

$$\nabla_\theta V^{\pi_\theta}(s) = \phi(s) + \int_{\mathcal{S}} P(s \rightarrow s' | 1, \pi_\theta) \nabla_\theta V^{\pi_\theta}(s') \, ds'$$

### 5. Unrolling the Recursion

Repeatedly unrolling this equation forward through time yields an infinite series without any discount factor damping the future terms:

$$\nabla_\theta V^{\pi_\theta}(s) = \phi(s) + \int_{\mathcal{S}} P(s \rightarrow s' | 1, \pi_\theta) \phi(s') \, ds' + \int_{\mathcal{S}} P(s \rightarrow s'' | 2, \pi_\theta) \phi(s'') \, ds'' + \dots$$

We can write this cleanly as a sum over all future timesteps $t$:

$$\nabla_\theta V^{\pi_\theta}(s) = \int_{\mathcal{S}} \sum_{t=0}^{\infty} P(s \rightarrow x | t, \pi_\theta) \phi(x) \, dx$$

### 6. Integrating over the Initial State Distribution

Now we substitute this back into the gradient of our global objective function $\nabla_\theta J(\theta)$:

$$\nabla_\theta J(\theta) = \int_{\mathcal{S}} \rho_0(s) \nabla_\theta V^{\pi_\theta}(s) \, ds$$

$$\nabla_\theta J(\theta) = \int_{\mathcal{S}} \rho_0(s) \left[ \int_{\mathcal{S}} \sum_{t=0}^{\infty} P(s \rightarrow x | t, \pi_\theta) \phi(x) \, dx \right] ds$$

By changing the order of integration, we isolate the total expected time spent in state $x$:

$$\nabla_\theta J(\theta) = \int_{\mathcal{S}} \left[ \int_{\mathcal{S}} \rho_0(s) \sum_{t=0}^{\infty} P(s \rightarrow x | t, \pi_\theta) \, ds \right] \phi(x) \, dx$$

The term inside the brackets is the **undiscounted state visitation frequency** (or the expected number of times state $x$ is visited in an episode), which we denote as $\eta^{\pi_\theta}(x)$:

$$\eta^{\pi_\theta}(x) = \sum_{t=0}^{\infty} P(s_t = x | \rho_0, \pi_\theta)$$

This simplifies our expression to:


$$\nabla_\theta J(\theta) = \int_{\mathcal{S}} \eta^{\pi_\theta}(s) \phi(s) \, ds$$

### 7. Applying the Log-Derivative Trick

Reintroducing the full definition of $\phi(s)$ (and keeping $s$ as our state variable):

$$\nabla_\theta J(\theta) = \int_{\mathcal{S}} \eta^{\pi_\theta}(s) \int_{\mathcal{A}} \nabla_\theta \pi_\theta(a|s) Q^{\pi_\theta}(s, a) \, da \, ds$$

We apply the log-derivative identity ($\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)$) to convert the inner integral into an expectation:

$$\nabla_\theta J(\theta) = \int_{\mathcal{S}} \eta^{\pi_\theta}(s) \int_{\mathcal{A}} \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s, a) \, da \, ds$$

---

### Conclusion

In the undiscounted setting ($\gamma=1$), the gradient can be expressed using the unnormalized state visitation measure $\eta^{\pi_\theta}(s)$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \eta^{\pi_\theta}, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s, a) \right]$$

> **Note on practical sampling:** Because $\eta^{\pi_\theta}(s)$ is an unnormalized measure (it integrates to the expected episode length $T$, rather than $1$), we often normalize it to a proper probability distribution $d^{\pi_\theta}(s) = \frac{\eta^{\pi_\theta}(s)}{\int_{\mathcal{S}} \eta^{\pi_\theta}(s')ds'}$. If normalized, the gradient is proportional to the expectation:
> 
> $$\nabla_\theta J(\theta) \propto \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s, a) \right]$$
> 
>