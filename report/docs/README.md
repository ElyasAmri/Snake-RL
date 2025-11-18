# Reinforcement Learning Theory Documentation

This directory contains comprehensive theoretical documentation for implementing Reinforcement Learning agents for the Snake game, including GPU utilization and multi-agent training.

## Documentation Structure

### Core RL Theory

1. **[01-core-rl-concepts.md](01-core-rl-concepts.md)**
   - Markov Decision Processes (MDPs)
   - Value functions (state-value and action-value)
   - Policy types (deterministic vs stochastic)
   - Bellman equations
   - Exploration vs exploitation

2. **[02-rl-algorithms.md](02-rl-algorithms.md)**
   - Q-Learning fundamentals
   - Deep Q-Networks (DQN)
   - DQN variants (Double DQN, Dueling DQN, Prioritized Replay, Rainbow)
   - Policy Gradient methods
   - Actor-Critic methods
   - Proximal Policy Optimization (PPO)

3. **[05-mathematical-formulations.md](05-mathematical-formulations.md)**
   - Temporal Difference (TD) learning
   - Policy gradient theorem
   - Advantage functions and GAE
   - Q-learning derivations
   - Bellman operators

### Application to Snake

4. **[03-snake-specific-considerations.md](03-snake-specific-considerations.md)**
   - State representation options (feature-based, grid-based, hybrid, ray-casting)
   - Action space design (absolute vs relative)
   - Reward shaping strategies
   - Snake-specific challenges and solutions

5. **[04-training-techniques.md](04-training-techniques.md)**
   - Experience replay
   - Target networks
   - Epsilon-greedy exploration
   - Reward normalization
   - Curriculum learning

### Advanced Topics

6. **[06-gpu-utilization.md](06-gpu-utilization.md)**
   - GPU acceleration for RL
   - Vectorized environments
   - PyTorch CUDA setup
   - Memory optimization techniques
   - Performance benchmarks
   - Best practices

7. **[07-multi-agent-training.md](07-multi-agent-training.md)**
   - Multi-agent RL theory (stochastic games, Nash equilibrium)
   - Dual-snake scenarios (competitive, cooperative, mixed)
   - Algorithms: IQL, MADDPG, QMIX, Self-Play
   - Handling non-stationarity
   - Curriculum learning for multi-agent
   - Training strategies

### Practical Implementation

8. **[08-implementation-guide.md](08-implementation-guide.md)**
   - Recommended hyperparameters (DQN, PPO, multi-agent)
   - Monitoring and debugging
   - Common pitfalls and solutions
   - Advanced techniques
   - Testing and evaluation
   - Deployment considerations

9. **[references.md](references.md)**
   - Key academic papers
   - Snake-specific research
   - Textbooks
   - Online resources
   - Software libraries
   - Community resources

## Reading Guide

### For Beginners

Start here to build foundational knowledge:
1. [01-core-rl-concepts.md](01-core-rl-concepts.md) - Understand MDP, value functions, policies
2. [02-rl-algorithms.md](02-rl-algorithms.md) - Learn DQN (recommended for Snake)
3. [03-snake-specific-considerations.md](03-snake-specific-considerations.md) - Apply to Snake
4. [04-training-techniques.md](04-training-techniques.md) - Training best practices
5. [08-implementation-guide.md](08-implementation-guide.md) - Start implementing

### For Intermediate Users

Already familiar with basic RL:
1. [02-rl-algorithms.md](02-rl-algorithms.md) - Review DQN variants and PPO
2. [05-mathematical-formulations.md](05-mathematical-formulations.md) - Deep dive into math
3. [06-gpu-utilization.md](06-gpu-utilization.md) - Speed up training
4. [08-implementation-guide.md](08-implementation-guide.md) - Optimize implementation

### For Advanced Users

Ready for multi-agent and advanced techniques:
1. [07-multi-agent-training.md](07-multi-agent-training.md) - Dual-snake training
2. [06-gpu-utilization.md](06-gpu-utilization.md) - Parallel environments
3. [05-mathematical-formulations.md](05-mathematical-formulations.md) - Theoretical foundations
4. [references.md](references.md) - Research papers

## Quick Reference

### Algorithm Selection for Snake

- **Recommended**: DQN or Double DQN
  - Discrete actions (3-4)
  - Off-policy learning
  - Well-established for games

- **Alternative**: PPO
  - Stochastic policies
  - Complex reward structures

- **Multi-Agent**:
  - Competitive: Self-Play or IQL
  - Cooperative: QMIX
  - Mixed: MADDPG or IQL

### Key Hyperparameters (DQN)

```
Learning rate: 0.001
Batch size: 64
Buffer size: 100,000
Gamma: 0.99
Epsilon: 1.0 -> 0.01 (decay 0.995)
Network: [128, 128] hidden layers
```

### State Representation

- **Start with**: Feature-based (11-dimensional)
- **Upgrade to**: Grid-based with CNN
- Always normalize inputs

### Reward Structure

```
Food: +10
Death: -10
Step penalty: -0.01
Distance reward: +1 (closer), -1 (farther)
```

## File Formats

All documentation is in Markdown format (.md) and can be viewed:
- In any text editor
- On GitHub with formatting
- Using Markdown viewers
- In IDEs with Markdown support

## Contributing

When adding new documentation:
1. Follow existing structure and formatting
2. Include mathematical formulations where appropriate
3. Provide practical examples
4. Reference sources in references.md
5. Use ASCII-only characters (no Unicode symbols)

## Usage Notes

- These documents focus on **theory** with minimal code
- Small code snippets are for illustration only
- Refer to implementation notebooks/scripts for full code
- Mathematical notation uses ASCII-compatible symbols

## Document Interconnections

```
Core Concepts (01)
    |
    v
Algorithms (02) <-> Math Formulations (05)
    |
    v
Snake Specific (03)
    |
    v
Training Techniques (04) <-> GPU Utilization (06)
    |
    v
Implementation Guide (08) <-> Multi-Agent (07)
    |
    v
References (09)
```

## Total Coverage

- **Core RL**: MDPs, value functions, policies, Bellman equations
- **Algorithms**: Q-Learning, DQN variants, Policy Gradient, PPO
- **Snake-Specific**: State/action spaces, rewards, challenges
- **Training**: Experience replay, target networks, exploration
- **Math**: TD learning, policy gradients, advantage functions
- **GPU**: Acceleration, vectorization, optimization
- **Multi-Agent**: MARL theory, dual-snake, algorithms
- **Implementation**: Hyperparameters, debugging, best practices
- **References**: Papers, books, courses, tools

## Next Steps

After reading the theory:
1. Set up development environment
2. Implement basic Snake environment
3. Start with simple DQN
4. Use recommended hyperparameters
5. Monitor training metrics
6. Iterate and improve

For questions or clarifications, refer to [references.md](references.md) for additional learning resources.
