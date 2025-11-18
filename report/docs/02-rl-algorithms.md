# Reinforcement Learning Algorithms

## 1. Q-Learning

Q-learning is a model-free, off-policy reinforcement learning algorithm that learns the optimal action-value function Q*(s, a).

### Algorithm

The Q-learning update rule is:

```
Q(s, a) <- Q(s, a) + alpha[R + gamma max_{a'} Q(s', a') - Q(s, a)]
```

Where:
- **alpha**: learning rate (step size)
- **R**: immediate reward
- **gamma**: discount factor
- **s'**: next state after taking action a in state s
- The term **[R + gamma max_{a'} Q(s', a') - Q(s, a)]** is called the TD (Temporal Difference) error

### Key Properties

1. **Off-policy**: Learns the optimal policy while following a different behavior policy (e.g., epsilon-greedy)
2. **Model-free**: Doesn't require knowledge of transition probabilities T(s, a, s')
3. **Convergence**: Guaranteed to converge to Q* under certain conditions:
   - Visiting all state-action pairs infinitely often
   - Appropriate learning rate decay

### Tabular Q-Learning Pseudocode

```
Initialize Q(s, a) arbitrarily for all s in S, a in A
Repeat for each episode:
    Initialize state s
    Repeat for each step of episode:
        Choose action a from s using policy derived from Q (e.g., epsilon-greedy)
        Take action a, observe reward R and next state s'
        Q(s, a) <- Q(s, a) + alpha[R + gamma max_{a'} Q(s', a') - Q(s, a)]
        s <- s'
    Until s is terminal
```

### Limitations for Snake

- Tabular Q-learning requires storing Q-values for every state-action pair
- Snake has an enormous state space (grid positions, snake body configuration, food location)
- Makes pure tabular Q-learning impractical for realistic Snake implementations
- **Solution**: Use function approximation with Deep Q-Networks

---

## 2. Deep Q-Networks (DQN)

DQN, introduced by Mnih et al. (2015), combines Q-learning with deep neural networks to handle large state spaces. It achieved human-level performance on many Atari games.

### Core Idea

Instead of storing Q-values in a table, approximate the Q-function using a neural network:

```
Q(s, a; theta) ~= Q*(s, a)
```

where **theta** represents the network parameters (weights and biases).

### Key Innovations

#### 1. Experience Replay

- Stores experiences (s, a, r, s') in a replay buffer D
- Randomly samples mini-batches from D for training
- Breaks correlations between consecutive experiences
- Enables more efficient use of experience data
- Typical buffer sizes: 50,000 to 1,000,000 transitions

**Benefits**:
- More stable training by removing sequential correlations
- Each experience can be used for multiple updates
- Reduces variance in updates

#### 2. Target Network

- Maintains a separate network Q(s, a; theta^-) with parameters theta^- for computing target values
- Target network parameters are updated periodically (e.g., every C steps): theta^- <- theta
- Stabilizes training by keeping the target fixed during multiple updates

### DQN Loss Function

```
L(theta) = E_{(s,a,r,s')~D}[(r + gamma max_{a'} Q(s', a'; theta^-) - Q(s, a; theta))^2]
```

The network is trained to minimize this temporal difference (TD) error.

### DQN Algorithm Pseudocode

```
Initialize replay buffer D with capacity N
Initialize Q-network with random weights theta
Initialize target network with weights theta^- = theta
For episode = 1 to M:
    Initialize state s
    For t = 1 to T:
        With probability epsilon select random action a
        Otherwise select a = argmax_a Q(s, a; theta)
        Execute action a, observe reward r and next state s'
        Store transition (s, a, r, s') in D
        Sample random mini-batch of transitions from D
        For each transition in mini-batch:
            If s' is terminal: y = r
            Else: y = r + gamma max_{a'} Q(s', a'; theta^-)
        Perform gradient descent on (y - Q(s, a; theta))^2
        Every C steps: theta^- <- theta
        s <- s'
```

### Network Architecture Considerations

**For feature-based state representation (11-dimensional state vector)**:
- Input layer: State features
- Hidden layers: 2-3 fully connected layers with 64-256 units
- Activation: ReLU for hidden layers
- Output layer: Linear activation, one output per action

**For image-based state representation (grid view)**:
- Input: 84x84x4 (or similar) preprocessed frames
- Convolutional layers to extract spatial features
- Fully connected layers
- Output layer: One Q-value per action

---

## 3. DQN Variants

### 3.1 Double DQN (DDQN)

Standard DQN suffers from overestimation bias because it uses the same network to both select and evaluate actions. Double DQN addresses this by decoupling action selection from action evaluation.

**Target in DQN**:
```
y = r + gamma max_{a'} Q(s', a'; theta^-)
```

**Target in Double DQN**:
```
y = r + gamma Q(s', argmax_{a'} Q(s', a'; theta), theta^-)
```

- Uses the online network (theta) to select the best action
- Uses the target network (theta^-) to evaluate that action
- Substantially reduces overestimation and leads to faster, more reliable training

### 3.2 Dueling DQN

Dueling DQN separates the Q-function into two components:

```
Q(s, a; theta) = V(s; theta_v) + A(s, a; theta_a) - mean_{a'} A(s, a'; theta_a)
```

Where:
- **V(s; theta_v)**: State-value function (how good is this state?)
- **A(s, a; theta_a)**: Advantage function (how much better is action a compared to average?)

**Network architecture**:
- Shared convolutional layers process the input
- Two separate streams:
  - Value stream: Outputs a single value V(s)
  - Advantage stream: Outputs |A| values, one per action
- Aggregation layer combines them to produce Q-values

**Benefits**:
- The agent can evaluate states without considering each action separately
- Particularly useful when many actions have similar values
- Learns more robust feature representations

### 3.3 Prioritized Experience Replay

Instead of uniformly sampling from the replay buffer, prioritize transitions based on their TD error:

```
P(i) = (|delta_i| + epsilon)^alpha / sum_k (|delta_k| + epsilon)^alpha
```

Where:
- **delta_i**: TD error of transition i
- **alpha**: Priority exponent (how much prioritization to use)
- **epsilon**: Small constant to ensure non-zero probability

Requires importance sampling weights to correct for bias:

```
w_i = (N x P(i))^{-beta}
```

**Benefits**:
- Learns more efficiently from important transitions
- Particularly useful in sparse reward environments like Snake

### 3.4 Noisy DQN

Replaces epsilon-greedy exploration with parametric noise in the network weights:

```
Q(s, a; theta + epsilon) where epsilon ~ N(0, sigma^2)
```

- Noise parameters are learned during training
- Provides state-dependent, structured exploration
- Eliminates the need to tune epsilon schedule

### 3.5 Rainbow DQN

Combines multiple improvements into a single algorithm:
1. Double DQN
2. Dueling networks
3. Prioritized replay
4. Multi-step learning
5. Distributional RL
6. Noisy networks

Rainbow achieves state-of-the-art performance on Atari benchmarks.

---

## 4. Policy Gradient Methods

Policy gradient methods directly optimize the policy pi(a|s; theta) rather than learning value functions.

### Basic Policy Gradient

The policy gradient theorem states:

```
nabla_theta J(theta) = E_pi[nabla_theta log pi(a|s; theta) Q^pi(s, a)]
```

Where J(theta) is the expected cumulative reward under policy pi.

### REINFORCE Algorithm

Update rule:
```
theta <- theta + alpha nabla_theta log pi(a_t|s_t; theta) G_t
```

Where G_t is the return from time t onwards.

**Benefits**:
- Can learn stochastic policies
- Effective in high-dimensional or continuous action spaces
- Guaranteed to converge to local optimum

**Drawbacks**:
- High variance in gradient estimates
- Sample inefficient
- Can be slow to converge

---

## 5. Actor-Critic Methods

Actor-Critic methods combine value-based and policy-based approaches:

- **Actor**: Policy network pi(a|s; theta_pi) that selects actions
- **Critic**: Value network V(s; theta_v) or Q(s, a; theta_q) that evaluates actions

### Advantage Actor-Critic (A2C)

Uses the advantage function to reduce variance:

```
A(s, a) = Q(s, a) - V(s) = r + gamma V(s') - V(s)
```

**Actor update**:
```
nabla_theta_pi J ~= nabla_theta_pi log pi(a|s; theta_pi) A(s, a)
```

**Critic update**:
```
L(theta_v) = (r + gamma V(s'; theta_v) - V(s; theta_v))^2
```

### A3C (Asynchronous Advantage Actor-Critic)

- Runs multiple agents in parallel environments
- Each agent computes gradients asynchronously
- Updates are applied to shared global network
- Highly efficient on multi-core CPUs

---

## 6. Proximal Policy Optimization (PPO)

PPO is a state-of-the-art policy gradient method that has become the default RL algorithm at OpenAI.

### Core Innovation

PPO constrains policy updates to prevent large, destabilizing changes. It clips the ratio of new to old policy probabilities:

```
r_t(theta) = pi(a_t|s_t; theta) / pi(a_t|s_t; theta_old)
```

### Clipped Surrogate Objective

```
L^CLIP(theta) = E_t[min(r_t(theta)A_t, clip(r_t(theta), 1-epsilon, 1+epsilon)A_t)]
```

Where:
- **r_t(theta)** is the probability ratio
- **A_t** is the advantage estimate
- **epsilon** is the clipping parameter (typically 0.1 or 0.2)
- **clip(r_t(theta), 1-epsilon, 1+epsilon)** limits r_t to [1-epsilon, 1+epsilon]

### Full PPO Loss (Actor-Critic Style)

```
L(theta) = L^CLIP(theta) - c_1 L^VF(theta) + c_2 S[pi(.|s_t; theta)]
```

Where:
- **L^VF**: Value function loss (MSE)
- **S**: Entropy bonus for exploration
- **c_1, c_2**: Coefficients balancing the terms

### Key Properties

1. **On-policy**: Uses data collected by current policy
2. **First-order method**: Only requires first derivatives (unlike TRPO)
3. **Stable**: Prevents destructive policy updates
4. **Simple**: Easier to implement and tune than TRPO
5. **Effective**: Works well on continuous and discrete action spaces

### PPO vs. DQN for Snake

- PPO can learn stochastic policies (useful if randomization helps)
- DQN is simpler and works well with discrete actions
- PPO may be more sample efficient with proper tuning
- DQN benefits from experience replay (off-policy)

---

## Algorithm Selection Guide for Snake

### Recommended: DQN or Double DQN
- Discrete action space (3-4 actions)
- Off-policy learning (efficient use of data)
- Well-established for game environments
- Relatively simple to implement

### Alternative: PPO
- If you want to explore stochastic policies
- Good for more complex reward structures
- Requires more tuning

### Advanced: Rainbow DQN
- Best performance but more complex
- Combine multiple improvements
- Consider after basic DQN works

### Not Recommended: Tabular Q-Learning
- State space too large for Snake
- Only viable for very small grids (5x5 or smaller)

---

## Summary

This document covered key RL algorithms applicable to Snake:

- **Q-Learning**: Foundation for value-based methods
- **DQN**: Scales Q-learning to large state spaces using neural networks
- **DQN Variants**: Improvements like Double DQN, Dueling DQN, etc.
- **Policy Gradient & Actor-Critic**: Alternative approaches
- **PPO**: State-of-the-art policy optimization

For the Snake game, **DQN or Double DQN** is the recommended starting point due to its proven effectiveness in discrete action game environments.
