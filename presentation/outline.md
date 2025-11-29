# Presentation Outline: Reinforcement Learning & Snake Game

**Duration:** 20-30 minutes
**Format:** PowerPoint (.pptx)
**Presenters:** 3 (alternating)
**Demo:** Pre-recorded GIF/video

---

## Presenter A (~7-8 min)

### Slide 1: Cover Page
- Title: "Reinforcement Learning Applied to the Snake Game"
- Course: ECEN 446
- Team member names
- Date

### Slide 2-3: Introduction
- What is Reinforcement Learning?
- Agent, Environment, State, Action, Reward
- The RL loop diagram
- Why Snake game?

### Slide 4-5: History of RL
- Timeline: Bellman (1950s) to Deep RL (2013+)
- Key milestones:
  - Dynamic Programming (Bellman)
  - TD Learning (Sutton, 1988)
  - Q-Learning (Watkins, 1989)
  - Deep Q-Network (Mnih et al., 2013)
  - AlphaGo (2016), OpenAI Five (2019)

### Slide 6-7: Playground - Snake as MDP
- Snake game rules
- MDP formulation:
  - State: Grid, snake position, food location
  - Actions: Up, Down, Left, Right
  - Rewards: +10 food, -10 death, step penalty
  - Terminal: Collision with wall/self

---

## Presenter B (~8-10 min)

### Slide 8-9: Value-Based Methods - DQN Family
- DQN: Q-function with neural networks, experience replay, target network
- Double DQN: Reduces overestimation bias
- Dueling DQN: Separates value and advantage streams
- Noisy DQN: Learned exploration
- PER DQN: Prioritized experience replay

### Slide 10-11: Policy Gradient Methods
- REINFORCE: Direct policy optimization, Monte Carlo returns
- A2C: Actor-Critic, advantage function
- PPO: Clipped surrogate objective, most stable

### Slide 12-13: Implementation Details
- State representations:
  - Basic (11 dims): Direction, food position, danger sensors
  - Flood-fill (14 dims): + reachable space
  - Grid-based (CNN): 3-channel image
- Network architectures: MLP (128x128), CNN
- Training setup: Gymnasium, PyTorch

---

## Presenter C (~7-8 min)

### Slide 14-15: Results & Comparisons
- Performance comparison chart
- Key findings:
  - PPO + flood-fill best (~37+ avg score)
  - Flood-fill adds +10-15 improvement
  - DQN variants competitive
- Learning curves

### Slide 16: GPU Utilization
- Vectorized environments: 256 parallel games
- 100-300x speedup
- PyTorch CUDA operations
- Training time: ~30-60 min for 10K episodes

### Slide 17-18: Two-Agent Competitive
- Competitive two-snake environment
- Curriculum learning stages:
  1. Static opponent
  2. Random opponent
  3. Greedy (BFS) opponent
  4. Defensive (flood-fill) opponent
  5. Co-evolution (self-play)

### Slide 19: Demo Video
- Single snake: PPO agent playing
- Two-snake: Competitive match

### Slide 20: Real-World Applications
- Robotics
- Game AI (AlphaGo, OpenAI Five)
- Autonomous vehicles
- Resource management
- Recommendation systems

### Slide 21: Conclusion & Q&A
- Summary of key learnings
- What worked best
- Questions?

---

## Time Allocation

| Section | Presenter | Duration |
|---------|-----------|----------|
| Intro + History + Playground | A | ~7-8 min |
| Algorithms + Implementation | B | ~8-10 min |
| Results + GPU + Two-Agent + Applications | C | ~7-8 min |
| Q&A | All | ~3-5 min |
| **Total** | | **25-30 min** |
