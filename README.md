# Snake-RL

**ECEN 446 Course Project: Reinforcement Learning Applied to the Snake Game**

A comprehensive implementation of reinforcement learning algorithms trained to play the classic Snake game, featuring both single-player and competitive two-snake modes with GPU-accelerated vectorized environments.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithms Implemented](#algorithms-implemented)
- [Training Notebooks](#training-notebooks)
- [Visualizers](#visualizers)
- [Baseline Agents](#baseline-agents)
- [State Representations](#state-representations)
- [Results](#results)
- [Report](#report)
- [Running Tests](#running-tests)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project implements and compares multiple reinforcement learning algorithms for playing Snake:

**Key Features:**
- 8 single-snake algorithms (DQN variants + policy gradient methods)
- Competitive two-snake mode with curriculum learning
- GPU-accelerated vectorized environments (256+ parallel games)
- Multiple state representations (feature-based and CNN grid-based)
- Real-time Pygame visualizers for both game modes
- Comprehensive evaluation and comparison framework

**Best Results:**
- Single-snake: PPO with flood-fill features achieves 37+ average score
- Two-snake: Curriculum-trained agents learn competitive strategies

---

## Project Structure

```
Snake-RL/
+-- core/                 # Core RL modules
|   +-- environment.py              # Single-snake Gymnasium environment
|   +-- environment_vectorized.py   # Vectorized single-snake (GPU)
|   +-- environment_two_snake_vectorized.py  # Two-snake competitive
|   +-- networks.py                 # Neural network architectures
|   +-- state_representations.py    # Feature encoders
|   +-- utils.py                    # Replay buffers, schedulers, metrics
|
+-- notebooks/            # Jupyter notebooks
|   +-- 01_baseline_testing.ipynb   # Baseline agent evaluation
|   +-- 02_model_evaluation.ipynb   # Full model comparison
|   +-- training/                   # Algorithm-specific training notebooks
|
+-- scripts/              # Python scripts
|   +-- baselines/        # Baseline agents (random, A*, scripted)
|   +-- training/         # Training scripts
|   +-- visualizer/       # Pygame visualizers
|
+-- results/              # Training outputs
|   +-- weights/          # Model checkpoints (.pt files)
|   +-- figures/          # Generated plots
|   +-- data/             # Metrics and logs
|
+-- report/               # LaTeX report and documentation
|   +-- report.pdf        # Final compiled report
|   +-- report.tex        # LaTeX source
|
+-- tests/                # Unit tests
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for training)
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Snake-RL
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   ```bash
   # Windows
   .\venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation:**
   ```bash
   ./venv/Scripts/python.exe -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

---

## Quick Start

### Train a DQN Agent

```bash
./venv/Scripts/python.exe -m papermill notebooks/training/train_dqn.ipynb output.ipynb \
  -p NUM_EPISODES 1000 -p USE_FLOOD_FILL True
```

### Visualize a Trained Agent

```bash
./venv/Scripts/python.exe scripts/visualizer/visualize_snake.py \
  --mode trained \
  --weights results/weights/dqn_mlp_floodfill_128x128_5000ep_20251121_025811.pt \
  --network dqn \
  --fps 15
```

### Run Model Evaluation

```bash
./venv/Scripts/python.exe -m papermill notebooks/02_model_evaluation.ipynb output.ipynb
```

---

## Algorithms Implemented

### Single-Snake Algorithms

| Algorithm | Type | Key Features |
|-----------|------|--------------|
| **DQN** | Value-based | Experience replay, target networks, epsilon-greedy |
| **Double DQN** | Value-based | Reduces Q-value overestimation bias |
| **Dueling DQN** | Value-based | Separate value and advantage streams |
| **Noisy DQN** | Value-based | Parametric noise for exploration (no epsilon) |
| **PER DQN** | Value-based | Prioritized experience replay |
| **PPO** | Policy gradient | Clipped surrogate objective, stable training |
| **A2C** | Policy gradient | Advantage actor-critic with N-step returns |
| **REINFORCE** | Policy gradient | Monte Carlo policy gradient baseline |

### Two-Snake Competitive Algorithms

| Algorithm | Training Method | Description |
|-----------|-----------------|-------------|
| **Classic DQN** | Co-evolution | Both agents learn simultaneously |
| **PPO Curriculum** | 5-stage curriculum | Progressive difficulty with scripted opponents |
| **DQN Curriculum** | 5-stage curriculum | DQN variant with curriculum learning |
| **PPO Co-evolution** | Direct competition | Both PPO agents trained together |

### Curriculum Learning Stages

1. **Stage 0 (Static)**: Opponent always goes straight
2. **Stage 1 (Random)**: Opponent takes random actions
3. **Stage 2 (Greedy)**: Opponent uses BFS pathfinding to food
4. **Stage 3 (Defensive)**: Opponent uses flood-fill for space control
5. **Stage 4 (Co-evolution)**: Train against the other learning agent

---

## Training Notebooks

All training notebooks support parameterization via [papermill](https://papermill.readthedocs.io/).

### Available Notebooks

| Notebook | Algorithm | Description |
|----------|-----------|-------------|
| `train_dqn.ipynb` | DQN | Vanilla Deep Q-Network |
| `train_double_dqn.ipynb` | Double DQN | Reduced overestimation |
| `train_dueling_dqn.ipynb` | Dueling DQN | Value/advantage separation |
| `train_noisy_dqn.ipynb` | Noisy DQN | Parametric exploration |
| `train_per_dqn.ipynb` | PER DQN | Prioritized replay |
| `train_ppo.ipynb` | PPO | Proximal Policy Optimization |
| `train_a2c.ipynb` | A2C | Advantage Actor-Critic |
| `train_reinforce.ipynb` | REINFORCE | Policy gradient |
| `train_two_snake_classic.ipynb` | Classic DQN | Two-snake co-evolution |
| `train_ppo_two_snake_mlp.ipynb` | PPO Two-Snake | Direct co-evolution |
| `train_ppo_two_snake_mlp_curriculum.ipynb` | PPO Curriculum | 5-stage curriculum |
| `train_dqn_two_snake_curriculum.ipynb` | DQN Curriculum | 5-stage curriculum |

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GRID_SIZE` | 10 | Game grid dimensions |
| `NUM_ENVS` | 256 | Parallel environments (GPU) |
| `NUM_EPISODES` | 500 | Training episodes |
| `LEARNING_RATE` | 0.001 | Optimizer learning rate |
| `BATCH_SIZE` | 64 | Mini-batch size |
| `GAMMA` | 0.99 | Discount factor |
| `USE_FLOOD_FILL` | False | Enable flood-fill features |

### Training Examples

```bash
# DQN with flood-fill features (5000 episodes)
./venv/Scripts/python.exe -m papermill notebooks/training/train_dqn.ipynb output.ipynb \
  -p NUM_EPISODES 5000 \
  -p USE_FLOOD_FILL True \
  -p NUM_ENVS 256

# PPO with custom learning rate
./venv/Scripts/python.exe -m papermill notebooks/training/train_ppo.ipynb output.ipynb \
  -p NUM_EPISODES 10000 \
  -p LEARNING_RATE 0.0003 \
  -p USE_FLOOD_FILL True

# Two-snake curriculum training
./venv/Scripts/python.exe -m papermill notebooks/training/train_ppo_two_snake_mlp_curriculum.ipynb output.ipynb \
  -p STAGE0_MIN_STEPS 10000 \
  -p STAGE4_MIN_STEPS 50000
```

---

## Visualizers

### Single Snake Visualizer

Real-time visualization of single-snake agents using Pygame.

```bash
./venv/Scripts/python.exe scripts/visualizer/visualize_snake.py [OPTIONS]
```

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--mode` | `random`, `trained`, `training` | required | Visualization mode |
| `--weights` | path | None | Model weights file (trained mode) |
| `--network` | `dqn`, `dueling`, `noisy` | `dueling` | Network architecture |
| `--algorithm` | `dqn`, `dueling`, `noisy` | `dqn` | Algorithm (training mode) |
| `--grid-size` | int | 10 | Grid dimensions |
| `--fps` | 1-60 | 10 | Frames per second |
| `--episodes` | int | 100 | Number of episodes |

**Examples:**

```bash
# Watch random agent
./venv/Scripts/python.exe scripts/visualizer/visualize_snake.py --mode random --fps 15

# Watch trained Dueling DQN
./venv/Scripts/python.exe scripts/visualizer/visualize_snake.py \
  --mode trained \
  --weights results/weights/dueling_dqn_mlp_floodfill_128x128_5000ep_20251121_115531.pt \
  --network dueling \
  --fps 10

# Watch real-time training
./venv/Scripts/python.exe scripts/visualizer/visualize_snake.py \
  --mode training \
  --algorithm dqn \
  --episodes 500 \
  --num-envs 128
```

### Two-Snake Visualizer

Real-time visualization of competitive two-snake matches.

```bash
./venv/Scripts/python.exe scripts/visualizer/visualize_two_snake.py [OPTIONS]
```

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--mode` | `random`, `trained`, `scripted` | `random` | Visualization mode |
| `--weights1` | path | None | Agent 1 weights (trained/scripted) |
| `--weights2` | path | None | Agent 2 weights (trained mode) |
| `--opponent` | `static`, `random`, `greedy`, `defensive` | `greedy` | Scripted opponent type |
| `--episodes` | int | 100 | Number of rounds |
| `--target-food` | int | 10 | Food to win |

**Examples:**

```bash
# Random vs random
./venv/Scripts/python.exe scripts/visualizer/visualize_two_snake.py --mode random --fps 10

# Trained agents compete
./venv/Scripts/python.exe scripts/visualizer/visualize_two_snake.py \
  --mode trained \
  --weights1 results/weights/ppo_two_snake_mlp/big_256x256_final_20251124_055135.pt \
  --weights2 results/weights/ppo_two_snake_mlp/small_128x128_final_20251124_055135.pt \
  --episodes 20

# Trained agent vs greedy scripted opponent
./venv/Scripts/python.exe scripts/visualizer/visualize_two_snake.py \
  --mode scripted \
  --weights1 results/weights/competitive/Stage2_Greedy/big_256x256_latest.pt \
  --opponent greedy
```

---

## Baseline Agents

### Random Agent

Selects actions uniformly at random. Used as a lower-bound baseline.

```python
from scripts.baselines import RandomAgent
agent = RandomAgent(action_space_type='relative', seed=42)
action = agent.get_action(env)
```

### Shortest Path Agent (A*)

Deterministic pathfinding using A* algorithm with Manhattan distance heuristic.

```python
from scripts.baselines import ShortestPathAgent
agent = ShortestPathAgent(action_space_type='relative')
action = agent.get_action(env)
```

### Scripted Opponents (Two-Snake)

Used in curriculum learning for progressive difficulty:

| Agent | Strategy | Difficulty |
|-------|----------|------------|
| **Static** | Always goes straight | Easiest |
| **Random** | Random valid actions | Easy |
| **Greedy** | BFS pathfinding to food | Medium |
| **Defensive** | Flood-fill space control | Hard |

```python
from scripts.baselines.scripted_opponents import get_scripted_agent
opponent = get_scripted_agent('greedy', device=torch.device('cuda'))
```

---

## State Representations

### Feature-Based (MLP Networks)

| Type | Dimensions | Features Included |
|------|------------|-------------------|
| **Basic** | 11 | Direction one-hot, relative food position, danger sensors |
| **Flood-fill** | 14 | Basic + reachable space in each direction |
| **Selective** | 19 | Flood-fill + enhanced spatial features |
| **Enhanced** | 24 | Full feature set with body awareness |

### Grid-Based (CNN Networks)

- **Input Shape:** `(3, grid_size, grid_size)`
- **Channels:** Snake head, snake body, food
- **Spatial:** Full board representation

---

## Results

### Evaluation Notebook

Run the comprehensive evaluation:

```bash
./venv/Scripts/python.exe -m papermill notebooks/02_model_evaluation.ipynb output.ipynb
```

### Generated Outputs

**Figures** (`results/figures/`):
- `single_snake_comparison.png` - Algorithm performance comparison
- `algorithm_family_comparison.png` - DQN variants vs policy gradient
- `flood_fill_impact.png` - Effect of flood-fill features

**Data** (`results/data/`):
- `evaluation_results_latest.json` - Full metrics
- `single_snake_summary_latest.csv` - Summary table

### Key Findings

1. **PPO with flood-fill** achieves highest single-snake scores
2. **Curriculum learning** enables competitive two-snake strategies
3. **Flood-fill features** significantly improve performance (+10-15 avg score)
4. **MLP networks** outperform CNN for feature-based representations

---

## Report

The full project report is available at `report/report.pdf`.

**Topics Covered:**
- Reinforcement learning theory (MDPs, value functions, policy gradient)
- Algorithm implementations (DQN variants, PPO, A2C, REINFORCE)
- Snake-specific considerations (state representations, reward shaping)
- GPU-accelerated training with vectorized environments
- Multi-agent competitive learning with curriculum
- Experimental results and analysis

---

## Running Tests

```bash
# Run all tests
./venv/Scripts/python.exe -m pytest tests/

# Run specific test file
./venv/Scripts/python.exe -m pytest tests/test_environment.py

# Run with verbose output
./venv/Scripts/python.exe -m pytest tests/ -v
```

---

## Acknowledgments

This project was developed for **ECEN 446** at Texas A&M University.

**Key Libraries:**
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Gymnasium](https://gymnasium.farama.org/) - RL environment interface
- [Pygame](https://www.pygame.org/) - Game visualization
- [Papermill](https://papermill.readthedocs.io/) - Notebook parameterization

---

## License

This project is for educational purposes as part of ECEN 446 coursework.
