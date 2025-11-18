# Training Techniques for Reinforcement Learning

## 1. Experience Replay

Experience replay stores and reuses past experiences to improve sample efficiency and training stability.

### Mechanism

1. Store transitions (s, a, r, s') in a replay buffer D
2. Sample random mini-batches from D for training
3. Update network using sampled experiences

### Key Hyperparameters

#### Buffer Size
- Common values: 10,000 to 1,000,000
- Larger buffers: more stable, but slower to adapt
- Smaller buffers: more recent data, but higher correlation
- For Snake: Start with 50,000-100,000

#### Batch Size
- Common values: 32, 64, 128
- Larger batches: more stable gradients, better GPU utilization
- Smaller batches: faster updates, more stochastic
- For Snake: 32-64 typically works well

### Benefits

1. **Breaks correlations**: Random sampling removes sequential dependencies
2. **Sample efficiency**: Each experience used multiple times
3. **Stabilizes training**: Smooths out variance in updates
4. **Prevents catastrophic forgetting**: Retains older experiences

### Catastrophic Forgetting

Neural networks can forget previously learned information when learning new information. Experience replay mitigates this by:
- Maintaining a diverse set of experiences
- Regularly revisiting old situations
- Preventing overfitting to recent trajectories

### Buffer Size Trade-offs

- **Small buffer (< 10,000)**: Risk of forgetting, fast adaptation, high correlation
- **Medium buffer (10,000-100,000)**: Good balance for most applications
- **Large buffer (> 100,000)**: Very stable, slow adaptation, memory intensive

Research shows that buffer size should be tuned based on:
- Network architecture susceptibility to catastrophic forgetting
- Environment complexity and non-stationarity
- Available memory resources

---

## 2. Target Networks

Target networks stabilize DQN training by providing consistent targets during updates.

### Problem Without Target Networks

In naive Q-learning with function approximation:

```
Q(s, a; theta) <- Q(s, a; theta) + alpha[r + gamma max_{a'} Q(s', a'; theta) - Q(s, a; theta)]
```

Both the predicted Q-value and the target use the same parameters theta, creating a "moving target" problem that causes instability.

### Solution

Maintain two networks:
1. **Online network** Q(s, a; theta): Updated every step
2. **Target network** Q(s, a; theta^-): Updated periodically

### Update Rule

**Target value**: y = r + gamma max_{a'} Q(s', a'; theta^-)

**Loss**: L(theta) = (y - Q(s, a; theta))^2

Every C steps (e.g., C=1000): theta^- <- theta

### Soft Updates (Alternative)

Instead of hard copying, gradually blend:

```
theta^- <- tau * theta + (1 - tau) * theta^-
```

where tau << 1 (e.g., 0.001)

### Impact

- Experience replay has the largest performance improvement
- Target network improvement is significant but not as critical
- Together they enable stable deep RL

### Hyperparameters

- Update frequency C: 1,000 to 10,000 steps
- Higher C: More stable, slower adaptation
- Lower C: Faster learning, less stable

---

## 3. Epsilon-Greedy Exploration

Epsilon-greedy is the most common exploration strategy for DQN.

### Algorithm

```
Pseudocode:
if random() < epsilon:
    return random_action()  # Explore
else:
    return argmax(Q(s, a))  # Exploit
```

### Epsilon Decay Schedules

#### 1. Linear Decay

```
epsilon = max(epsilon_min, epsilon_start - (epsilon_start - epsilon_min) * step / total_steps)
```

Example: Start at 1.0, decay to 0.01 over 100,000 steps

#### 2. Exponential Decay

```
epsilon = max(epsilon_min, epsilon_start * (decay_rate ** episode))
```

Example: Start at 1.0, multiply by 0.995 each episode

#### 3. Step Decay

```
if episode % decay_interval == 0:
    epsilon *= decay_factor
epsilon = max(epsilon, epsilon_min)
```

Example: Multiply by 0.5 every 1000 episodes

### Recommended Values for Snake

```
epsilon_start = 1.0      # Start with full exploration
epsilon_min = 0.01       # Minimum 1% exploration
epsilon_decay = 0.995    # Decay factor per episode
# or
decay_steps = 100000     # Linear decay over 100k steps
```

### Best Practices

- Start with high exploration (epsilon = 1.0) to gather diverse data
- Gradually reduce to low value (epsilon = 0.01-0.05) but don't eliminate entirely
- Longer decay for complex environments
- Monitor performance during decay period

---

## 4. Reward Normalization

Reward normalization is essential when using neural networks as function approximators.

### Why Normalize

- Neural networks are not invariant to input scale
- Rewards can vary over many orders of magnitude
- Large rewards lead to large gradients, causing instability
- Small rewards lead to small gradients, causing slow learning

### Technique 1: Reward Clipping

Used in original Atari DQN paper:

```
reward_clipped = clip(reward, -1, 1)
```

All positive rewards become +1, negative rewards become -1

**Advantages**:
- Simple to implement
- Limits gradient magnitude
- Same hyperparameters across different games

**Disadvantages**:
- Loses reward magnitude information
- May not distinguish between small and large rewards

### Technique 2: Reward Scaling

Scale rewards to reasonable range:

```
reward_scaled = reward / scale_factor
```

Common scale factors: 10, 100, max_expected_reward

### Technique 3: Standardization (Z-score normalization)

Normalize to zero mean and unit variance:

```
reward_normalized = (reward - mean_reward) / (std_reward + epsilon)
```

For online learning, use running statistics to track mean and variance.

### Technique 4: Return-Based Normalization

Normalize returns (cumulative rewards) rather than immediate rewards:

```
returns = compute_returns(rewards, gamma)
normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

### For Snake, Recommended Approach

```
Pseudocode for simple scaling:
if died:
    return -1.0
elif ate_food:
    return 1.0
else:
    return reward / 10.0  # Scale small intermediate rewards
```

Or use full standardization for more stable training.

---

## 5. Curriculum Learning

Curriculum learning structures the learning process by gradually increasing task difficulty.

### Core Concept

Instead of training on the full problem from the start, begin with easier versions and progressively increase complexity.

### For Snake, Curriculum Strategies

#### 1. Grid Size Progression

```
Curriculum stages:
Stage 1: 5x5 grid, 10k episodes
Stage 2: 10x10 grid, 20k episodes
Stage 3: 15x15 grid, 30k episodes
Stage 4: 20x20 grid, 50k episodes
```

Start with tiny grids where food is easy to find, then expand.

#### 2. Initial Snake Position

```
# Easy: Start very close to food
initial_distance = max_distance - episode / 1000

# Start far from food as training progresses
```

#### 3. Snake Length Progression

```
Phase 1: Max length = 5 (learn basic food seeking)
Phase 2: Max length = 10 (learn intermediate navigation)
Phase 3: No limit (learn full game)
```

#### 4. Obstacle Density

```
# Start with no obstacles
# Gradually add walls or barriers
# Finally add full collision detection
```

#### 5. Reward Shaping Schedule

```
# Early training: Heavy reward shaping
# Late training: Sparse rewards only

shaping_weight = max(0, 1 - episode / total_episodes)
reward = base_reward + shaping_weight * shaped_reward
```

### Benefits

- Faster convergence
- More stable learning
- Better final performance
- Avoids getting stuck in local optima

### Challenges

- Requires careful design of progression
- Risk of too-easy curriculum (undertraining)
- Risk of too-hard jumps (catastrophic forgetting)
- Additional hyperparameters to tune

### Best Practice for Snake

Start with a simple curriculum (grid size progression) and add complexity only if needed.

---

## Summary

Key training techniques for RL:

### Experience Replay
- Store and reuse experiences
- Buffer size: 50k-100k for Snake
- Batch size: 32-64

### Target Networks
- Stabilize training with fixed targets
- Update every 1k-10k steps
- Consider soft updates for continuous adaptation

### Epsilon-Greedy
- Balance exploration and exploitation
- Start at 1.0, decay to 0.01-0.05
- Use exponential or linear decay

### Reward Normalization
- Essential for stable neural network training
- Clip, scale, or standardize rewards
- Prevents gradient instability

### Curriculum Learning
- Gradually increase difficulty
- Start with small grids for Snake
- Progress to full complexity

These techniques work together to enable effective training of deep RL agents.
