# Implementation Guide and Best Practices

## 1. Recommended Hyperparameters

### 1.1 DQN for Single Snake

```
Network Architecture:
- Input: State representation (11-dim feature or grid)
- Hidden layers: [128, 128] or [256, 256]
- Activation: ReLU
- Output: Number of actions (3 or 4)
- Learning rate: 0.001 or 0.0005

Training:
- Batch size: 64
- Gamma (discount): 0.99
- Buffer size: 100,000
- Min buffer size: 1,000 (start training)
- Target update frequency: 1,000 steps

Exploration:
- Epsilon start: 1.0
- Epsilon min: 0.01
- Epsilon decay: 0.995 per episode
  OR
- Epsilon decay steps: 100,000 (linear)

Rewards:
- Food: +10
- Death: -10
- Step penalty: -0.01
- Distance reward: +1 if closer, -1 if farther

Training:
- Episodes: 10,000
- Max steps per episode: 1,000
```

### 1.2 Double DQN (Recommended)

Same as DQN, but with Double Q-learning target:
```
Target: y = r + gamma Q(s', argmax_a Q(s', a; theta), theta^-)
```

No additional hyperparameters needed.

### 1.3 PPO for Single Snake

```
Network Architecture:
- Actor hidden: [128, 128]
- Critic hidden: [128, 128]
- Learning rate: 0.0003

Training:
- Batch size: 64
- N epochs: 10 (PPO update epochs)
- N steps: 2,048 (steps before update)
- Gamma: 0.99
- GAE lambda: 0.95
- Clip epsilon: 0.2
- VF coefficient: 0.5
- Entropy coefficient: 0.01

Training:
- Total timesteps: 1,000,000
```

### 1.4 Multi-Agent (Dual Snake)

**IQL (Independent Q-Learning)**:
```
Same as DQN for each agent
Separate networks and buffers
Potentially different learning rates
```

**QMIX (Cooperative)**:
```
Agent Q-networks: Same as DQN
Mixer network hidden: 128
Learning rate: 0.0005 (shared)
Buffer size: 100,000
Batch size: 64
```

---

## 2. Monitoring and Debugging

### 2.1 Key Metrics to Track

**Performance Metrics**:
1. Average episode reward (smoothed over 100 episodes)
2. Average episode length
3. Success rate (% episodes where food was eaten)
4. Maximum snake length achieved
5. Survival time

**Learning Metrics**:
1. Loss value (Q-learning loss or policy loss)
2. Q-value estimates (mean, max, min)
3. TD error magnitude
4. Gradient norms
5. Learning rate (if adaptive)

**Exploration Metrics**:
1. Current epsilon value
2. Action distribution (are all actions being tried?)
3. State visitation frequency
4. Entropy of policy (for policy gradient methods)

### 2.2 Visualization

**During Training**:
- Plot average reward per 100 episodes
- Plot episode length
- Plot loss curve
- Show current epsilon

**After Training**:
- Visualize trained agent gameplay
- Heatmap of state visitations
- Q-value visualization
- Action selection patterns

### 2.3 Logging

Recommended logging structure:
```
Episode 100, Avg Reward: 5.23, Avg Length: 42.1, Epsilon: 0.90, Loss: 0.123
Episode 200, Avg Reward: 8.45, Avg Length: 67.3, Epsilon: 0.81, Loss: 0.089
...
```

Log to file for later analysis:
```
CSV format:
episode,avg_reward,avg_length,epsilon,loss,max_length
100,5.23,42.1,0.90,0.123,15
200,8.45,67.3,0.81,0.089,23
```

---

## 3. Common Pitfalls and Solutions

### 3.1 Agent Doesn't Learn Anything

**Symptoms**:
- Reward stays near zero or very negative
- Agent moves randomly
- No improvement after thousands of episodes

**Possible Causes and Solutions**:

1. **Reward scale too small**
   - Solution: Increase reward magnitudes
   - Try: Food = +50, Death = -50

2. **Network not updating**
   - Solution: Check gradients are being computed
   - Verify optimizer is stepping
   - Print loss values

3. **Exploration insufficient**
   - Solution: Ensure epsilon is high initially (1.0)
   - Slow down epsilon decay

4. **State representation not informative**
   - Solution: Verify state contains necessary information
   - Check normalization
   - Try different representation

### 3.2 Agent Learns Then Gets Worse

**Symptoms**:
- Performance improves for a while
- Then suddenly degrades
- May recover or continue degrading

**Possible Causes and Solutions**:

1. **Catastrophic forgetting**
   - Solution: Increase replay buffer size
   - Try: 200,000 instead of 50,000

2. **Learning rate too high**
   - Solution: Reduce learning rate
   - Try: 0.0001 instead of 0.001

3. **Epsilon decayed too fast**
   - Solution: Slow down epsilon decay
   - Maintain minimum exploration (epsilon_min = 0.05)

4. **Overfitting to recent experiences**
   - Solution: Increase buffer size
   - Sample more diverse batches

### 3.3 Agent Learns Suboptimal Policy

**Symptoms**:
- Agent converges to consistent but poor strategy
- Gets stuck in local optimum
- Repeats same mistakes

**Possible Causes and Solutions**:

1. **Reward function encourages wrong behavior**
   - Solution: Redesign rewards
   - Ensure food reward >> other rewards
   - Check for unintended incentives

2. **Insufficient exploration**
   - Solution: Increase epsilon_min to 0.05
   - Extend exploration phase
   - Try noisy networks

3. **Local optimum**
   - Solution: Restart training with different seed
   - Use curriculum learning
   - Increase exploration

4. **Network capacity too small**
   - Solution: Increase hidden layer sizes
   - Add more layers
   - Try: [256, 256] instead of [128, 128]

### 3.4 Training is Very Slow

**Symptoms**:
- Low FPS (frames per second)
- Training takes hours/days
- Slow convergence

**Possible Causes and Solutions**:

1. **Not using GPU**
   - Solution: Move model and data to GPU
   - Use vectorized environments
   - Check CUDA availability

2. **Batch size too small**
   - Solution: Increase batch size to 64-128
   - Better GPU utilization

3. **Network too large**
   - Solution: Reduce network size if not needed
   - Start with [128, 128]

4. **Environment is slow**
   - Solution: Optimize game logic
   - Use vectorized operations
   - Implement parallel environments

### 3.5 High Variance in Performance

**Symptoms**:
- Some episodes very good, others very bad
- Inconsistent behavior
- High standard deviation in rewards

**Possible Causes and Solutions**:

1. **Insufficient training**
   - Solution: Train for more episodes
   - Increase total timesteps

2. **Replay buffer too small**
   - Solution: Increase buffer size
   - Use larger batches

3. **Reward normalization missing**
   - Solution: Normalize or clip rewards
   - Standardize returns

4. **Target network update too frequent**
   - Solution: Increase update frequency
   - Try: 10,000 instead of 1,000

---

## 4. Advanced Techniques

### 4.1 Prioritized Experience Replay

Sample important transitions more frequently:
```
Priority = |TD error| + epsilon
Probability prop to priority^alpha

Importance weights: w = (N * P(i))^{-beta}
```

**Benefits**:
- Faster learning
- Better sample efficiency
- Particularly useful for Snake (sparse rewards)

**Cost**:
- More complex implementation
- Additional hyperparameters (alpha, beta)

### 4.2 Hindsight Experience Replay (HER)

Relabel failed trajectories with alternative goals:
```
Original: Failed to reach food at (10, 10)
Relabeled: Successfully reached position (5, 7) where we died
```

**Benefits**:
- Learn from failures
- Significantly speeds up learning in sparse reward environments

**Application to Snake**:
- Relabel "reached position X" as a success
- Learn navigation even when not reaching food

### 4.3 Imitation Learning / Behavioral Cloning

Pre-train on expert demonstrations:
```
1. Generate expert trajectories (A* algorithm, human play)
2. Train network to predict expert actions
3. Fine-tune with RL
```

**Benefits**:
- Warm-start the policy
- Faster initial learning
- Better exploration

**For Snake**:
- Use A* pathfinding as expert
- Pre-train on 10,000 expert steps
- Then switch to RL

### 4.4 Curriculum Learning

Already covered in training techniques, but worth emphasizing:
```
Stage 1: 5x5 grid, 10k episodes
Stage 2: 10x10 grid, 20k episodes
Stage 3: 15x15 grid, 30k episodes
Stage 4: 20x20 grid, 50k episodes
```

### 4.5 Frame Stacking

Stack last N frames as input:
```
Instead of: Current grid state
Use: Stack of 4 most recent states
```

**Benefits**:
- Infer velocity and direction
- Handle partial observability
- Better for image-based representations

---

## 5. Debugging Checklist

When training isn't working, check:

- [ ] State representation contains necessary information
- [ ] State values are normalized to [0, 1] or [-1, 1]
- [ ] Rewards are appropriately scaled
- [ ] Learning rate is reasonable (0.0001 - 0.001)
- [ ] Batch size is adequate (32-128)
- [ ] Replay buffer has minimum samples before training
- [ ] Target network is being updated
- [ ] Epsilon is decaying properly
- [ ] Loss is being computed correctly
- [ ] Gradients are flowing (check with gradient norms)
- [ ] Actions are being executed correctly in environment
- [ ] Rewards match expected values
- [ ] Episode termination is working correctly
- [ ] GPU is being utilized (if available)

---

## 6. Performance Optimization

### 6.1 Code-Level Optimizations

1. **Vectorize operations**: Use NumPy/PyTorch instead of Python loops
2. **Batch processing**: Process multiple states simultaneously
3. **GPU utilization**: Move tensors to GPU, minimize transfers
4. **Pre-allocate memory**: Avoid repeated allocations
5. **Profile code**: Identify bottlenecks with profilers

### 6.2 Algorithm-Level Optimizations

1. **Double DQN**: Reduces overestimation, faster convergence
2. **Prioritized replay**: Learn from important transitions
3. **Dueling architecture**: Better value estimation
4. **Noisy networks**: Better exploration without epsilon tuning
5. **Multi-step returns**: Better credit assignment

### 6.3 Training-Level Optimizations

1. **Parallel environments**: 64-256 simultaneous games
2. **Larger batches**: Better GPU utilization
3. **Mixed precision**: 2x speedup with FP16
4. **Gradient accumulation**: Simulate larger batches
5. **Curriculum learning**: Faster convergence

---

## 7. Testing and Evaluation

### 7.1 Evaluation Protocol

```
For each trained model:
1. Run 100 test episodes with epsilon=0 (no exploration)
2. Record:
   - Average reward
   - Average length
   - Max length achieved
   - Success rate
   - Standard deviation
3. Compare across models
```

### 7.2 Generalization Testing

Test on variations:
- Different grid sizes
- Different initial positions
- Different food spawn patterns
- Against different opponents (multi-agent)

### 7.3 Ablation Studies

Test importance of components:
- With vs without reward shaping
- Different network architectures
- Different hyperparameters
- Single vs multi-agent

---

## 8. Deployment Considerations

### 8.1 Model Export

```
Save trained model:
- Network weights
- Hyperparameters
- State normalization parameters
- Action selection logic
```

### 8.2 Inference Optimization

For deployment:
- Set epsilon = 0 (deterministic)
- Use CPU for inference if needed
- Simplify network if possible
- Remove training-specific code

### 8.3 Real-Time Performance

For interactive gameplay:
- Optimize inference speed
- Pre-allocate memory
- Use smaller networks if latency critical
- Consider model quantization

---

## Summary

### Quick Start Recommendations

1. **Start Simple**:
   - Feature-based state (11-dim)
   - DQN algorithm
   - Small network [128, 128]
   - Basic reward shaping

2. **If Not Working**:
   - Check debugging checklist
   - Verify state/rewards
   - Increase reward magnitudes
   - Slow down epsilon decay

3. **If Working But Slow**:
   - Increase batch size
   - Use GPU
   - Vectorize environments
   - Try Double DQN

4. **For Better Performance**:
   - Curriculum learning
   - Prioritized replay
   - Larger networks
   - Fine-tune hyperparameters

5. **For Multi-Agent**:
   - Start with IQL
   - Use curriculum
   - Monitor both agents
   - Try self-play for competitive

This guide provides a practical roadmap for implementing and debugging RL agents for Snake.
