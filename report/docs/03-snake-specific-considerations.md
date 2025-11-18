# Snake-Specific Considerations for RL

## 1. State Representation

The state representation is crucial for learning performance. There are several approaches for representing the Snake game state.

### Approach 1: Feature-Based (Compact) Representation

A minimal state vector containing essential information.

**Example 11-dimensional boolean vector**:
```
state = [
    danger_straight, danger_right, danger_left,      # 3 bits: danger indicators
    moving_up, moving_down, moving_left, moving_right, # 4 bits: current direction
    food_up, food_down, food_left, food_right          # 4 bits: food location
]
```

**Advantages**:
- Very compact and efficient
- Fast training
- Easy to interpret
- Works well with small neural networks

**Disadvantages**:
- Loses spatial information
- Cannot see entire snake body
- May miss complex patterns

### Approach 2: Grid-Based Representation

Represent the full game grid as a 2D or 3D tensor.

**Example multi-channel representation**:
- Channel 1: Snake head position
- Channel 2: Snake body positions
- Channel 3: Food position
- Channel 4 (optional): Walls/boundaries

For a 20x20 grid with 4 channels: shape = (20, 20, 4)

**Advantages**:
- Preserves all spatial information
- Can learn complex patterns
- Generalizes to larger grids

**Disadvantages**:
- Requires convolutional networks
- Slower training
- Needs more data

### Approach 3: Hybrid Representation

Combines features with partial spatial information.

**Components**:
- Manhattan distance to food
- Snake length
- Proximity to walls (4 directions)
- Local grid around head (e.g., 5x5 patch)
- Body segment positions

### Approach 4: Ray-Casting / Sensor-Based

Cast rays in multiple directions from snake head.

**For each direction, measure**:
- Distance to wall
- Distance to body
- Distance to food

Example with 8 directions: 24-dimensional vector (3 measurements x 8 directions)

**Advantages**:
- Rotation/translation invariant
- Compact yet informative
- Mimics vision systems

**Best Practices**:
- Start with simple feature-based representation
- Add complexity only if needed
- Normalize all features to [0, 1] or [-1, 1]
- Test multiple representations empirically

---

## 2. Action Space Design

### Approach 1: Absolute Actions (4 actions)

Actions: {UP, DOWN, LEFT, RIGHT}

**Advantages**:
- Intuitive and simple
- Matches game mechanics directly

**Disadvantages**:
- Invalid actions possible (moving directly backward into body)
- Need to handle or prevent invalid moves

### Approach 2: Relative Actions (3 actions)

Actions: {STRAIGHT, TURN_LEFT, TURN_RIGHT}

**Advantages**:
- No invalid actions
- Smaller action space
- More natural decision-making

**Disadvantages**:
- Requires tracking current direction
- Slightly more complex to implement

### Recommendation

Relative actions (3 actions) are generally preferred for Snake because:
- Eliminates the possibility of immediate death from invalid moves
- Reduces action space by 25%
- Aligns with how the problem is naturally framed

### Action Masking

If using absolute actions, implement action masking:
- Prevent selection of opposite direction
- Set Q-value of invalid action to -infinity
- Only consider valid actions during argmax

---

## 3. Reward Shaping Strategies

Reward design is critical for Snake. Sparse rewards (only +1 for food, -1 for death) make learning very slow.

### Basic Reward Structure

```
reward = {
    +10     if ate food
    -10     if died (hit wall or body)
    0       otherwise
}
```

### Problem with Basic Rewards

In a 20x20 grid, the probability of randomly eating food is extremely low. The agent spends 99% of time receiving 0 reward, making learning very difficult.

### Improved Reward Shaping

#### 1. Distance-Based Rewards

Give small rewards for moving closer to food:

```
delta_distance = old_distance - new_distance
reward = {
    +10                  if ate food
    -10                  if died
    +1 * delta_distance  if moved closer/farther
}
```

**Warning**: Pure distance-based rewards can cause oscillating behavior (snake moves back and forth). Use cautiously or combine with penalties.

#### 2. Step Penalty

Encourage efficiency:

```
reward = {
    +10     if ate food
    -10     if died
    -0.01   for each step (small penalty)
}
```

This encourages the snake to eat food quickly rather than wandering.

#### 3. Survival Bonus

Reward staying alive:

```
reward = {
    +10     if ate food
    -10     if died
    +0.1    for each step survived
}
```

#### 4. Combined Reward Structure (Recommended)

```
Pseudocode:
if died:
    return -100  # Large penalty for death
elif ate_food:
    return +50   # Large reward for food
else:
    # Small reward for approaching food
    distance_reward = (prev_distance - current_distance) * 2
    survival_reward = 0.1  # Small survival bonus
    step_penalty = -0.01   # Tiny penalty to encourage efficiency
    return distance_reward + survival_reward + step_penalty
```

#### 5. Length-Based Rewards

Scale rewards by snake length:

```
reward_scale = 1 + (snake_length / 10)
# Increase reward magnitude as snake grows
```

### Key Principles

- Food reward >> death penalty > intermediate rewards
- Avoid rewards that encourage local optima
- Test different reward functions empirically
- Consider normalizing rewards
- Balance exploration vs. exploitation

### Common Pitfalls

- **Too large distance rewards**: Leads to oscillation behavior
- **Too small food rewards**: Agent doesn't learn to prioritize food
- **Too harsh death penalty**: Agent becomes overly conservative
- **No intermediate rewards**: Extremely slow learning

---

## 4. Challenges Specific to Snake

### 4.1 Sparse Rewards

**Problem**:
- Food appears infrequently, especially for untrained agents
- Most actions yield zero immediate reward

**Solutions**:
- Reward shaping (distance-based, survival bonus)
- Curiosity-driven exploration
- Curriculum learning

### 4.2 Dynamic Environment

**Problem**:
- State space changes as snake grows
- Longer snake makes navigation harder

**Solutions**:
- Train on progressive difficulty
- Length-dependent rewards
- Adaptive policies

### 4.3 Varying Episode Length

**Problem**:
- Episodes can be very short (immediate death) or very long
- Creates high variance in returns

**Solutions**:
- Normalize returns
- Use advantage estimation
- Clip episode length during training

### 4.4 Self-Collision

**Problem**:
- Snake body creates moving obstacles
- Becomes harder as snake grows

**Solutions**:
- Include body positions in state
- Use path planning heuristics
- Train with increasing snake lengths

### 4.5 Credit Assignment

**Problem**:
- A death might be caused by decisions made several steps earlier
- Hard to assign blame correctly

**Solutions**:
- Multi-step returns
- Eligibility traces
- TD(lambda)

### 4.6 Exploration Difficulty

**Problem**:
- Random exploration rarely leads to food in large grids

**Solutions**:
- Curriculum learning (start with small grids)
- Guided exploration
- Imitation learning from baseline algorithms

### 4.7 Non-Stationarity

**Problem**:
- Optimal policy changes as snake grows
- Early game: aggressive food seeking
- Late game: careful navigation

**Solutions**:
- Use recurrent networks (LSTM) to capture state history
- Separate policies for different lengths
- Adaptive learning rates

---

## Summary

For successful Snake RL implementation:

### State Representation
- Start with feature-based (11-dimensional) for quick prototyping
- Use grid-based with CNN for better spatial understanding
- Normalize all inputs

### Action Space
- Prefer relative actions (3) over absolute (4)
- Implement action masking if using absolute actions

### Reward Design
- Use dense rewards with distance-based shaping
- Balance food reward, death penalty, and intermediate rewards
- Avoid oscillation by careful tuning

### Address Challenges
- Sparse rewards: Use reward shaping
- Dynamic environment: Progressive training
- Credit assignment: Multi-step learning
- Exploration: Curriculum learning

These considerations are critical for training effective Snake RL agents.
