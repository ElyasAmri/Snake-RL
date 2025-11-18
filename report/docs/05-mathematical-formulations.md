# Mathematical Formulations in Reinforcement Learning

## 1. Temporal Difference Learning

Temporal Difference (TD) learning is a fundamental concept that combines ideas from Monte Carlo and dynamic programming methods.

### Core Idea

Update value estimates based on other estimates (bootstrapping) rather than waiting for final outcomes.

### TD(0) - One-Step TD

```
V(s_t) <- V(s_t) + alpha[R_{t+1} + gamma V(s_{t+1}) - V(s_t)]
```

Where:
- **R_{t+1} + gamma V(s_{t+1})** is the TD target
- **R_{t+1} + gamma V(s_{t+1}) - V(s_t)** is the TD error (delta_t)

**Characteristics**:
- Updates immediately after each step
- Uses bootstrapping (estimates based on estimates)
- Low variance but higher bias

### n-Step TD

Use n future rewards before bootstrapping:

```
G_t^{(n)} = R_{t+1} + gamma R_{t+2} + ... + gamma^{n-1} R_{t+n} + gamma^n V(s_{t+n})

V(s_t) <- V(s_t) + alpha[G_t^{(n)} - V(s_t)]
```

**Trade-off**:
- Larger n: Lower bias, higher variance
- Smaller n: Higher bias, lower variance

### TD(lambda) with Eligibility Traces

TD(lambda) combines all n-step returns using exponential weighting:

```
G_t^lambda = (1-lambda) sum_{n=1}^{infinity} lambda^{n-1} G_t^{(n)}
```

**Forward View**:
- Theoretical perspective
- Combines all n-step returns
- Not practical for online learning

**Backward View (Eligibility Traces)**:

Maintain a trace for each state indicating how "eligible" it is for update:

```
e_t(s) = {
    gamma * lambda * e_{t-1}(s) + 1,  if s = s_t
    gamma * lambda * e_{t-1}(s),      otherwise
}
```

Update rule:
```
V(s) <- V(s) + alpha * delta_t * e_t(s)  for all s
```

Where delta_t = R_{t+1} + gamma V(s_{t+1}) - V(s_t)

**Properties**:
- lambda = 0: Equivalent to TD(0) (one-step)
- lambda = 1: Equivalent to Monte Carlo (wait until episode end)
- 0 < lambda < 1: Balances bias and variance

**Benefits for Snake**:
- Better credit assignment (reward propagates backward through trajectory)
- Faster learning when rewards are sparse
- More robust to delayed rewards

---

## 2. Policy Gradient Theorem

The policy gradient theorem provides the foundation for policy gradient methods.

### Objective

Maximize expected cumulative reward:

```
J(theta) = E_{tau~pi_theta}[sum_{t=0}^{T} gamma^t r_t] = E_{s~d^{pi_theta}}[V^{pi_theta}(s)]
```

Where:
- **tau**: trajectory (s_0, a_0, r_0, s_1, a_1, r_1, ...)
- **d^{pi_theta}**: state distribution under policy pi_theta

### Policy Gradient Theorem

```
nabla_theta J(theta) = E_{tau~pi_theta}[sum_{t=0}^{T} nabla_theta log pi_theta(a_t|s_t) G_t]
```

Or equivalently using Q-values:

```
nabla_theta J(theta) = E_{s~d^{pi_theta}, a~pi_theta}[nabla_theta log pi_theta(a|s) Q^{pi_theta}(s, a)]
```

### Derivation Sketch

Starting from the objective:

```
J(theta) = E_{tau}[R(tau)]
         = sum_{tau} P(tau; theta) R(tau)

nabla_theta J(theta) = sum_{tau} nabla_theta P(tau; theta) R(tau)
                     = sum_{tau} P(tau; theta) nabla_theta log P(tau; theta) R(tau)
                     = E_{tau}[nabla_theta log P(tau; theta) R(tau)]
```

Using the chain rule on the trajectory probability and the Markov property, this simplifies to the policy gradient theorem.

### REINFORCE Algorithm

Monte Carlo estimate of policy gradient:

```
nabla_theta J(theta) ~= (1/N) sum_{i=1}^{N} sum_{t=0}^{T_i} nabla_theta log pi_theta(a_t^i|s_t^i) G_t^i
```

Update:
```
theta <- theta + alpha * nabla_theta J(theta)
```

### With Baseline

To reduce variance, subtract a baseline b(s):

```
nabla_theta J(theta) = E[nabla_theta log pi_theta(a|s) (Q^{pi_theta}(s, a) - b(s))]
```

Common baseline: V^{pi_theta}(s), which gives the advantage function:

```
A^{pi_theta}(s, a) = Q^{pi_theta}(s, a) - V^{pi_theta}(s)
```

### Actor-Critic Gradient

```
nabla_theta J(theta) = E[nabla_theta log pi_theta(a|s) A^{pi_theta}(s, a)]
```

Where advantage is estimated using a critic network.

---

## 3. Advantage Function

The advantage function measures how much better an action is compared to the average action in that state.

### Definition

```
A^pi(s, a) = Q^pi(s, a) - V^pi(s)
```

### Interpretation

- **A^pi(s, a) > 0**: Action a is better than average
- **A^pi(s, a) < 0**: Action a is worse than average
- **A^pi(s, a) = 0**: Action a is average

### Estimation Methods

#### 1. One-Step TD

```
A(s_t, a_t) ~= r_t + gamma V(s_{t+1}) - V(s_t) = delta_t
```

This is the TD error.

#### 2. n-Step Returns

```
A(s_t, a_t) ~= r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1} + gamma^n V(s_{t+n}) - V(s_t)
```

#### 3. Generalized Advantage Estimation (GAE)

Combines all n-step advantages with exponential weighting:

```
A_t^{GAE(gamma,lambda)} = sum_{l=0}^{infinity} (gamma * lambda)^l * delta_{t+l}
```

Where delta_t = r_t + gamma V(s_{t+1}) - V(s_t)

This can be computed recursively:

```
A_t^{GAE} = delta_t + gamma * lambda * A_{t+1}^{GAE}
```

**Hyperparameters**:
- **gamma**: Discount factor (typically 0.99)
- **lambda**: GAE parameter (typically 0.95)
  - lambda = 0: High bias, low variance (one-step TD)
  - lambda = 1: Low bias, high variance (Monte Carlo)

### Benefits

- Reduces variance in policy gradient estimates
- Provides more informative learning signal
- Balances bias-variance trade-off via lambda parameter
- Essential for stable actor-critic training

### In Dueling DQN

The network explicitly computes value and advantage:

```
Q(s, a) = V(s) + (A(s, a) - (1/|A|) sum_{a'} A(s, a'))
```

The subtraction of mean advantage ensures identifiability (prevents the network from arbitrarily shifting values between V and A).

---

## 4. Q-Learning Update Derivation

### Tabular Q-Learning

Starting from the Bellman optimality equation:

```
Q*(s, a) = E[r + gamma max_{a'} Q*(s', a')]
```

The Q-learning update aims to move Q(s, a) toward this target:

```
Q(s, a) <- Q(s, a) + alpha[r + gamma max_{a'} Q(s', a') - Q(s, a)]
```

Where:
- **alpha**: learning rate
- **r + gamma max_{a'} Q(s', a')**: target
- **r + gamma max_{a'} Q(s', a') - Q(s, a)**: TD error

### Convergence Conditions

Q-learning converges to Q* under the following conditions:

1. All state-action pairs are visited infinitely often
2. Learning rate satisfies:
   - sum_{t} alpha_t = infinity (infinite total learning)
   - sum_{t} alpha_t^2 < infinity (learning rate decreases)

Common learning rate schedules:
- Constant: alpha_t = alpha
- 1/t decay: alpha_t = alpha_0 / t
- Polynomial: alpha_t = alpha_0 / (1 + t)^k

### Deep Q-Learning Loss

With function approximation Q(s, a; theta):

```
L(theta) = E_{(s,a,r,s')~D}[(r + gamma max_{a'} Q(s', a'; theta^-) - Q(s, a; theta))^2]
```

Gradient:

```
nabla_theta L(theta) = E[2(Q(s, a; theta) - y) nabla_theta Q(s, a; theta)]
```

Where y = r + gamma max_{a'} Q(s', a'; theta^-) is treated as constant (target network).

---

## 5. Bellman Backup Operators

### Bellman Expectation Operator T^pi

For a given policy pi:

```
(T^pi V)(s) = sum_a pi(a|s) sum_{s'} P(s'|s,a)[R(s,a) + gamma V(s')]
```

**Properties**:
- Contraction mapping: ||T^pi V - T^pi U|| <= gamma ||V - U||
- Unique fixed point: V^pi
- Repeated application converges to V^pi

### Bellman Optimality Operator T*

```
(T* V)(s) = max_a sum_{s'} P(s'|s,a)[R(s,a) + gamma V(s')]
```

**Properties**:
- Contraction mapping with modulus gamma
- Unique fixed point: V*
- Forms the basis for value iteration

### Value Iteration

```
V_{k+1} = T* V_k
```

Converges to V* as k -> infinity.

### Policy Iteration

Alternates between:

1. **Policy Evaluation**: Compute V^pi by solving or approximating (T^pi)^k V
2. **Policy Improvement**: pi_{k+1}(s) = argmax_a Q^{pi_k}(s, a)

Converges to optimal policy in finite iterations for finite MDPs.

---

## 6. Entropy Regularization

Entropy regularization encourages exploration by penalizing deterministic policies.

### Entropy of a Policy

```
H(pi(.|s)) = -sum_a pi(a|s) log pi(a|s)
```

or for continuous actions:

```
H(pi(.|s)) = -integral pi(a|s) log pi(a|s) da
```

### Regularized Objective

```
J(theta) = E_{tau~pi_theta}[sum_{t=0}^{T} gamma^t (r_t + beta H(pi_theta(.|s_t)))]
```

Where beta is the entropy coefficient.

### Benefits

- Encourages exploration
- Prevents premature convergence to deterministic policy
- Improves robustness
- Can help escape local optima

### In PPO

The entropy bonus is added to the loss:

```
L(theta) = L^CLIP(theta) - c_1 L^VF(theta) + c_2 H(pi_theta)
```

Where c_2 is the entropy coefficient (typically 0.01).

---

## Summary

This document covered key mathematical formulations:

### Temporal Difference Learning
- TD(0) for one-step updates
- n-step TD for bias-variance trade-off
- TD(lambda) with eligibility traces for better credit assignment

### Policy Gradient
- Policy gradient theorem enables direct policy optimization
- REINFORCE algorithm for Monte Carlo policy gradients
- Baselines reduce variance

### Advantage Function
- Measures relative action quality
- GAE balances bias and variance
- Essential for actor-critic methods

### Q-Learning
- Bellman optimality equation foundation
- TD error drives learning
- Convergence guarantees under conditions

### Bellman Operators
- Contraction mappings ensure convergence
- Value iteration and policy iteration algorithms

### Entropy Regularization
- Encourages exploration
- Prevents premature convergence

These mathematical foundations underpin all modern RL algorithms.
