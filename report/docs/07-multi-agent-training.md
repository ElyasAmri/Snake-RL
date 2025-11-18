# Multi-Agent Training for Dual-Snake RL

## 1. Multi-Agent RL Fundamentals

Multi-agent RL extends single-agent RL to scenarios with multiple decision-makers.

### Theoretical Framework

The environment is modeled as a **Stochastic Game** (also called Markov Game):

**Definition**: A stochastic game is a tuple (N, S, {A^i}_{i in N}, T, {R^i}_{i in N}, gamma) where:
- **N = {1, ..., n}** is the set of agents
- **S** is the state space
- **A^i** is the action space for agent i
- **T: S x A^1 x ... x A^n -> Delta(S)** is the transition function
- **R^i: S x A^1 x ... x A^n -> R** is the reward function for agent i
- **gamma in [0, 1)** is the discount factor

### Key Concepts

1. **Joint Action Space**: A = A^1 x A^2 x ... x A^n

2. **Joint Policy**: pi = (pi^1, pi^2, ..., pi^n) where pi^i: S -> Delta(A^i)

3. **Value Function for agent i**:
   ```
   V^i_pi(s) = E_pi[sum_{t=0}^{infinity} gamma^t R^i_t | s_0 = s]
   ```

4. **Nash Equilibrium**: A joint policy pi* is a Nash equilibrium if for all agents i:
   ```
   V^i_{pi*}(s) >= V^i_{(pi^{-i}, pi'^i)}(s)  for all s in S, for all pi'^i
   ```
   where pi^{-i} represents policies of all agents except i.

### Cooperative vs Competitive Scenarios

**Cooperative (Team Reward)**:
```
R^1_t = R^2_t = R_{team}(s_t, a^1_t, a^2_t)
```
Agents share the same reward, leading to fully cooperative behavior.

**Competitive (Zero-Sum)**:
```
R^1_t = -R^2_t
```
One agent's gain is another's loss.

**Mixed (General-Sum)**:
```
R^1_t != R^2_t, and R^1_t + R^2_t != constant
```
Agents have partially aligned, partially conflicting interests.

---

## 2. Dual-Snake Game Scenarios

### Scenario 1: Competitive Snake

**Setup**:
- Two snakes on the same grid
- Single food item at a time
- Collision with other snake = death
- Winner: snake that survives longest or grows largest

**State Representation Options**:

*Global view (5 channels)*:
- Channel 0: Snake 1 body
- Channel 1: Snake 1 head
- Channel 2: Snake 2 body
- Channel 3: Snake 2 head
- Channel 4: Food

*Agent-centric view (4 channels per agent)*:
- Channel 0: Self body
- Channel 1: Self head
- Channel 2: Opponent (body + head)
- Channel 3: Food

**Reward Design (Competitive)**:
```
Reward structure:
- Eating food: +10
- Opponent dies: +20
- Self dies: -20
- Survival: +0.01 per step
- Opponent eats food: -5 (relative disadvantage)
```

### Scenario 2: Cooperative Snake

**Setup**:
- Two snakes work together
- Shared food items or multiple food
- Objective: maximize total food collected
- Collision = both lose

**Reward Design (Cooperative)**:
```
Team reward structure:
- Team food collection: +10 per food (shared)
- Any snake dies: -50 (shared penalty)
- Survival: +0.01 (shared)
- Cooperation bonus: +bonus if coordinating
```

Both agents receive the same reward.

### Scenario 3: Mixed Strategy

**Setup**:
- Multiple food items
- Snakes get individual rewards for food
- But share penalty for either dying
- Encourages competition for food but cooperation for survival

**Reward Design (Mixed)**:
```
Mixed reward:
- Individual food rewards (competitive): +10
- Shared death penalty (cooperative): -30 for both
- Survival bonus: +0.01 each
```

---

## 3. Multi-Agent RL Algorithms

### 3.1 Independent Q-Learning (IQL)

Each agent learns independently, treating other agents as part of the environment.

**Concept**:
- Agent 1 has Q-network Q^1(s, a; theta_1)
- Agent 2 has Q-network Q^2(s, a; theta_2)
- Each learns using standard DQN
- No communication between agents during training

**Advantages**:
- Simple to implement
- Scalable to many agents
- No communication overhead

**Disadvantages**:
- Non-stationarity: environment appears non-stationary from each agent's perspective
- No coordination
- May not converge to Nash equilibrium

**When to use**: Quick prototyping, competitive scenarios

### 3.2 Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

Centralized training, decentralized execution paradigm.

**Key Idea**:
- Agents use global information during training
- But only local observations during execution

**For agent i**:
- **Actor** (policy): mu^i(o^i; theta^i) maps observation to action
- **Critic**: Q^i(o, a; phi^i) evaluates joint action given global observation

**Centralized Critic Loss**:
```
L(phi^i) = E[(Q^i(o, a) - y)^2]
where y = r^i + gamma Q^i_{target}(o', a')
and a' = (mu^1(o^1), ..., mu^n(o^n))
```

**Actor Loss**:
```
nabla_{theta^i} J(theta^i) = E[nabla_{theta^i} mu^i(o^i) nabla_{a^i} Q^i(o, a)|_{a^i=mu^i(o^i)}]
```

**Advantages**:
- Handles non-stationarity better
- Enables coordination
- Decentralized execution

**Disadvantages**:
- More complex implementation
- Requires global state during training
- Designed for continuous actions (needs adaptation for Snake)

**When to use**: Cooperative tasks, when global state available during training

### 3.3 Value Decomposition Networks (VDN/QMIX)

Designed for cooperative scenarios. Decomposes global Q-value into agent-specific Q-values.

**VDN**: Simple additive decomposition
```
Q_tot(s, a) = sum_i Q^i(o^i, a^i)
```

**QMIX**: More expressive mixing using a mixing network
```
Q_tot(s, a) = f_mix(Q^1(o^1, a^1), ..., Q^n(o^n, a^n); s)
```
where f_mix is a monotonic function.

**Monotonicity Constraint**:
```
partial Q_tot / partial Q^i >= 0 for all i
```

This ensures that:
- If agent i improves its Q^i, total Q_tot doesn't decrease
- Global optimum aligned with local optima
- Enables decentralized execution

**Advantages**:
- Excellent for cooperative tasks
- Theoretically grounded
- Enables credit assignment

**Disadvantages**:
- Only for fully cooperative scenarios
- More complex than IQL
- Requires careful hyperparameter tuning

**When to use**: Cooperative dual-snake, team objectives

### 3.4 Self-Play

Train a single agent against itself.

**Concept**:
- Main agent plays as Snake 1
- Opponent is either:
  - Current agent (self-play)
  - Historical snapshot of agent (opponent pool)

**Opponent Pool Strategy**:
1. Train main agent
2. Periodically snapshot current agent
3. Add to opponent pool
4. Sample opponent from pool for next episode

**Selection Strategy**:
```
Recent opponents prioritized:
prob(opponent_i) = 1 / (N - i)
where N = pool size, i = opponent index
```

**Advantages**:
- Simple for competitive scenarios
- Continuous curriculum (opponent improves)
- Single network to train

**Disadvantages**:
- Can lead to cyclic strategies (rock-paper-scissors)
- May overfit to self
- Requires careful opponent selection

**When to use**: Competitive dual-snake, adversarial training

---

## 4. Handling Non-Stationarity

Multi-agent environments are non-stationary from each agent's perspective because other agents' policies change during training.

### Challenge

From Agent 1's perspective:
- Environment transition: P(s'|s, a_1, a_2)
- But a_2 depends on pi_2, which changes
- Breaks Markov assumption

### Solutions

#### 1. Experience Replay with Policy Versioning

Track which policy version generated each experience:
```
Experience = (state, action, reward, next_state, done, policy_version)

Sample only recent experiences (within version threshold)
```

#### 2. Opponent Modeling

Learn to predict opponent's actions:
```
Opponent model: M(s) -> P(a_opponent|s)

Use prediction in decision-making:
Q(s, a_self) accounts for likely opponent actions
```

#### 3. Stabilization Techniques

- Update target networks less frequently (every 10k steps instead of 1k)
- Use larger replay buffers
- Slower learning rates
- Gradual policy updates (PPO-style clipping)

---

## 5. Curriculum Learning for Dual-Snake

Progressive difficulty for multi-agent training.

### Curriculum Stages

**Stage 0: Learn basic movement (no opponent)**
- Grid size: 10x10
- Opponent: inactive
- Food respawn: yes
- Steps: 100,000

**Stage 1: Static opponent**
- Grid size: 10x10
- Opponent: doesn't move
- Steps: 50,000

**Stage 2: Random opponent**
- Grid size: 12x12
- Opponent policy: random
- Steps: 100,000

**Stage 3: Scripted opponent**
- Grid size: 12x12
- Opponent policy: simple heuristic (e.g., always move toward food)
- Steps: 100,000

**Stage 4: Learning opponent (self-play)**
- Grid size: 15x15
- Opponent: learning agent
- Steps: 500,000

### Benefits

- Gradual increase in difficulty
- More stable learning
- Better final performance
- Faster convergence

---

## 6. Training Strategies

### Simultaneous Training

Both agents learn at the same time:
```
for episode in episodes:
    while not done:
        a1 = agent1.select_action(obs1)
        a2 = agent2.select_action(obs2)
        obs1', obs2', r1, r2, done = env.step(a1, a2)
        agent1.update(obs1, a1, r1, obs1')
        agent2.update(obs2, a2, r2, obs2')
```

**Pros**: Realistic, both agents improve together
**Cons**: Highly non-stationary, harder to converge

### Alternating Training

Train agents one at a time:
```
# Phase 1: Train agent 1, freeze agent 2
for episodes:
    train agent1 while agent2 acts with fixed policy

# Phase 2: Train agent 2, freeze agent 1
for episodes:
    train agent2 while agent1 acts with fixed policy

# Repeat
```

**Pros**: More stable, easier to debug
**Cons**: Less realistic, slower adaptation

### Population-Based Training

Maintain population of agents, evolve over time:
```
Population = [agent1, agent2, ..., agent_N]

Each generation:
- Evaluate all agents
- Select best performers
- Create offspring (copy + mutate)
- Replace worst performers
```

**Pros**: Diverse strategies, robust to local optima
**Cons**: Computationally expensive, complex implementation

---

## 7. Implementation Considerations

### State Representation

**Option 1: Global state** (both agents see everything)
- Full grid with both snakes and food
- Pro: Complete information
- Con: Large state space

**Option 2: Local observations** (each agent sees own view)
- Agent-centric grid
- Pro: Smaller state space, more scalable
- Con: Partial observability

**Recommendation**: Start with global, move to local if needed

### Action Synchronization

**Simultaneous actions**:
```
Both agents select actions
Environment updates with both actions
Both agents receive next state
```

**Sequential actions** (turn-based):
```
Agent 1 selects action
Environment updates
Agent 2 selects action
Environment updates
```

**Recommendation**: Simultaneous for Snake (more realistic)

### Reward Normalization

Normalize rewards for stable multi-agent training:
```
reward = clip(reward, -1, 1)
or
reward = reward / reward_scale
```

Important because agents may receive very different reward magnitudes.

---

## 8. Hyperparameters for Multi-Agent Training

### Independent Q-Learning (IQL)
```
learning_rate: 1e-4
batch_size: 64
buffer_size: 100,000
gamma: 0.99
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay: 0.9995
target_update_freq: 1000
```

### MADDPG
```
actor_lr: 1e-3
critic_lr: 1e-3
batch_size: 128
buffer_size: 100,000
gamma: 0.99
tau: 0.01 (soft update coefficient)
noise: 0.1
```

### QMIX
```
learning_rate: 5e-4
batch_size: 64
buffer_size: 100,000
gamma: 0.99
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay: 0.9995
target_update_freq: 200
```

---

## Summary

Multi-agent training for dual-snake adds complexity but enables rich behaviors:

### Algorithms
- **IQL**: Simplest, good for prototyping
- **MADDPG**: Centralized training, handles cooperation
- **QMIX**: Best for cooperative scenarios
- **Self-Play**: Natural for competitive scenarios

### Key Challenges
- Non-stationarity from changing opponent policies
- Credit assignment in multi-agent settings
- Coordination vs competition trade-offs

### Best Practices
- Start with IQL for quick prototyping
- Use curriculum learning for stable training
- Consider self-play for competitive scenarios
- QMIX for cooperative tasks
- Monitor both agents' performance separately

### For Dual-Snake
1. Competitive: Use self-play or IQL
2. Cooperative: Use QMIX
3. Mixed: Use IQL or MADDPG
4. Always use curriculum learning
5. Monitor for cyclic behaviors

Multi-agent RL is more challenging but enables emergent strategies and interesting gameplay dynamics.
