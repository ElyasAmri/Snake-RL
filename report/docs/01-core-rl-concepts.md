# Core Reinforcement Learning Concepts

## 1. Markov Decision Processes (MDPs)

A Markov Decision Process provides the mathematical framework for modeling sequential decision-making problems under uncertainty.

### Formal Definition

An MDP is defined as a tuple (S, A, T, R, gamma) where:

- **S**: State space - the set of all possible states in the environment
- **A**: Action space - the set of all possible actions the agent can take
- **T**: Transition function T: S x A x S -> [0,1], where T(s, a, s') = P(S_t = s' | S_{t-1} = s, A_{t-1} = a)
- **R**: Reward function R: S x A -> R, specifying immediate reward for taking action a in state s
- **gamma**: Discount factor gamma in [0, 1], balancing immediate vs. future rewards

### The Markov Property

The Markov property (memorylessness) states that the future is conditionally independent of the past given the present:

```
P(S_{t+1} | S_t, S_{t-1}, ..., S_0) = P(S_{t+1} | S_t)
```

This means knowing the current state provides all necessary information to predict the future, making past states irrelevant.

### Objective

The goal is to find a policy pi that maximizes the expected cumulative reward:

```
E[sum_{t=0}^{T} gamma^t R(s_t, a_t) | s_0 = s]
```

The discount factor gamma ensures convergence and prioritizes nearer rewards over distant ones.

---

## 2. Value Functions

Value functions estimate the expected long-term reward from states or state-action pairs.

### State-Value Function V^pi(s)

The state-value function represents the expected cumulative reward from state s following policy pi:

```
V^pi(s) = E_pi[sum_{t=0}^{infinity} gamma^t R(s_t, a_t) | s_0 = s]
```

**Interpretation**: How good is it to be in this state?

### Action-Value Function Q^pi(s, a)

The action-value function (Q-function) represents the expected cumulative reward from taking action a in state s and then following policy pi:

```
Q^pi(s, a) = E_pi[sum_{t=0}^{infinity} gamma^t R(s_t, a_t) | s_0 = s, a_0 = a]
```

**Interpretation**: How good is it to take this action in this state?

**Key Advantage**: The Q-function enables action selection without requiring a model of the environment's dynamics.

### Optimal Value Functions

The optimal value functions represent the maximum achievable value:

```
V*(s) = max_pi V^pi(s)
Q*(s, a) = max_pi Q^pi(s, a)
```

---

## 3. Policy Definition and Types

A policy defines the agent's behavior - how it selects actions given states.

### Deterministic Policy

A deterministic policy is a function pi: S -> A that maps each state to a single action:

```
a = pi(s)
```

**Characteristics**:
- Always outputs the same action for a given state
- Straightforward to interpret and implement
- Appropriate for precise control tasks
- Useful as the final policy after training

### Stochastic Policy

A stochastic policy is a probability distribution pi: S x A -> [0,1] over actions given states:

```
pi(a|s) = P(A_t = a | S_t = s)
```

**Characteristics**:
- Action selection based on probability distribution
- Better for adversarial/game environments
- Essential during training for exploration
- Can capture uncertainty in the environment

### When to Use Each Type

- **Stochastic during training**: Enables exploration and prevents local optima
- **Deterministic for deployment**: Clear optimal actions in well-defined environments
- **Stochastic in adversarial games**: Prevents exploitation by opponents

---

## 4. Bellman Equations

Bellman equations provide recursive decompositions of value functions, expressing the value of a state in terms of immediate rewards and successor state values.

### Bellman Expectation Equation for V^pi

```
V^pi(s) = E_pi[R(s, a) + gamma V^pi(s')]
        = sum_a pi(a|s) sum_{s'} T(s, a, s')[R(s, a) + gamma V^pi(s')]
```

**Decomposition**:
1. Immediate reward expected from current state
2. Discounted future value from the next state

### Bellman Expectation Equation for Q^pi

```
Q^pi(s, a) = R(s, a) + gamma sum_{s'} T(s, a, s') sum_{a'} pi(a'|s') Q^pi(s', a')
```

### Bellman Optimality Equation for V*

```
V*(s) = max_a [R(s, a) + gamma sum_{s'} T(s, a, s') V*(s')]
```

### Bellman Optimality Equation for Q*

```
Q*(s, a) = R(s, a) + gamma sum_{s'} T(s, a, s') max_{a'} Q*(s', a')
```

**Significance**: The Bellman optimality equations identify the best actions by finding the maximum expected value, forming the foundation for algorithms like Q-learning and DQN.

---

## 5. Exploration vs. Exploitation

The exploration-exploitation dilemma is a fundamental challenge in RL that balances two opposing strategies.

### The Dilemma

**Exploitation**:
- Choose the best action based on current knowledge
- Maximize immediate reward using learned information
- **Risk**: Getting stuck in local optima without discovering better options

**Exploration**:
- Try new actions that may lead to better future outcomes
- Gather information about the environment
- **Risk**: Spending too much time on suboptimal actions

### Epsilon-Greedy Strategy

The most common approach to balance exploration and exploitation:

```
pi(s) = {
  random action from A,        with probability epsilon
  argmax_a Q(s, a),            with probability (1-epsilon)
}
```

### Epsilon Decay

Typically, epsilon starts at 1.0 (full exploration) and decays over time:

**Exponential Decay**:
```
epsilon_t = max(epsilon_min, epsilon_start x decay_rate^t)
```

**Linear Decay**:
```
epsilon_t = epsilon_start - (epsilon_start - epsilon_min) x t/T
```

**Common Values**:
- epsilon_start = 1.0 (start with full exploration)
- epsilon_min = 0.01-0.05 (maintain small exploration)
- decay_rate = 0.995 (per episode)

### Alternative Exploration Strategies

1. **Upper Confidence Bound (UCB)**: Select actions based on potential optimality
2. **Boltzmann Exploration (Softmax)**: Sample actions from probability distribution based on Q-values
3. **Noisy Networks**: Add parametric noise to network weights for state-dependent exploration
4. **Curiosity-driven**: Provide intrinsic rewards for novel states

---

## Summary

These core concepts form the foundation of reinforcement learning:

- **MDPs** provide the mathematical framework for sequential decision-making
- **Value functions** estimate long-term rewards for states and actions
- **Policies** define how agents select actions (deterministic or stochastic)
- **Bellman equations** enable recursive computation of values
- **Exploration-exploitation** balance is crucial for effective learning

Understanding these concepts is essential before implementing RL algorithms for the Snake game or any other application.
