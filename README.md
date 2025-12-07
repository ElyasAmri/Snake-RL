# Snake-RL

**ECEN 446 Course Project: Reinforcement Learning Applied to the Snake Game**

A comprehensive implementation of reinforcement learning algorithms trained to play the classic Snake game, featuring both single-player and competitive two-snake modes with GPU-accelerated vectorized environments.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Pre-computed Results](#pre-computed-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithms Implemented](#algorithms-implemented)
- [Training Scripts](#training-scripts)
- [Visualizers](#visualizers)
- [Baseline Agents](#baseline-agents)
- [State Representations](#state-representations)
- [Results](#results)
- [Report](#report)
- [Reproducibility](#reproducibility)
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

## Pre-computed Results

If you prefer not to run simulations yourself, pre-computed results (trained weights, evaluation metrics, and generated figures) are available as a release download:

**📦 Download:** [https://github.com/ElyasAmri/Snake-RL/releases/tag/latest](https://github.com/ElyasAmri/Snake-RL/releases/tag/latest)

This release includes:
- All trained model weights (`.pt` files)
- Evaluation results and metrics (JSON/CSV)
- Generated comparison plots and figures
- Complete `results/` folder ready to use

Simply extract the release assets into the project root to use the pre-trained models with the visualizers or review the evaluation outputs directly.

---

## Project Structure

```
Snake-RL/
├── core/                 # Core RL modules
│   ├── environment.py              # Single-snake Gymnasium environment
│   ├── environment_vectorized.py   # Vectorized single-snake (GPU)
│   ├── environment_two_snake_vectorized.py  # Two-snake competitive
│   ├── networks.py                 # Neural network architectures
│   ├── state_representations.py    # Feature encoders
│   └── utils.py                    # Replay buffers, schedulers, metrics
│
├── scripts/              # Python scripts
│   ├── baselines/        # Baseline agents (random, A*, scripted)
│   ├── training/         # Training scripts for all algorithms
│   ├── visualizer/       # Pygame visualizers
│   └── evaluation/       # Evaluation and plotting scripts
│
├── results/              # Training outputs
│   ├── weights/          # Model checkpoints (.pt files)
│   ├── figures/          # Generated plots
│   └── data/             # Metrics and logs
│
├── report/               # LaTeX report and documentation
│   ├── report.pdf        # Final compiled report
│   └── report.tex        # LaTeX source
│
└── tests/                # Unit tests
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
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

---

## Quick Start

### Train a DQN Agent

```bash
python scripts/training/train_dqn.py
```

### Visualize a Trained Agent

```bash
python scripts/visualizer/visualize_snake.py \
  --mode trained \
  --weights results/weights/dqn_floodfill.pt \
  --network dqn \
  --fps 15
```

### Run Two-Snake Competition

```bash
python scripts/run_two_snake_experiments.py
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

## Training Scripts

All training scripts are located in `scripts/training/`.

### Available Scripts

| Script | Algorithm | Description |
|--------|-----------|-------------|
| `train_dqn.py` | DQN | Vanilla Deep Q-Network |
| `train_ppo.py` | PPO | Proximal Policy Optimization |
| `train_a2c.py` | A2C | Advantage Actor-Critic |
| `train_reinforce.py` | REINFORCE | Policy gradient |
| `train_rainbow.py` | Rainbow | Combined DQN improvements |
| `train_all_models.py` | All | Train all single-snake algorithms |
| `train_two_snake_classic.py` | DQN | Two-snake co-evolution |
| `train_ppo_two_snake_mlp.py` | PPO | Two-snake direct co-evolution |
| `train_ppo_two_snake_mlp_curriculum.py` | PPO | 5-stage curriculum learning |

### Training Examples

```bash
# Train DQN agent
python scripts/training/train_dqn.py

# Train PPO agent
python scripts/training/train_ppo.py

# Train all single-snake models
python scripts/training/train_all_models.py

# Two-snake curriculum training
python scripts/training/train_ppo_two_snake_mlp_curriculum.py
```

---

## Visualizers

### Single Snake Visualizer

Real-time visualization of single-snake agents using Pygame.

```bash
python scripts/visualizer/visualize_snake.py [OPTIONS]
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
python scripts/visualizer/visualize_snake.py --mode random --fps 15

# Watch trained Dueling DQN
python scripts/visualizer/visualize_snake.py \
  --mode trained \
  --weights results/weights/dueling_dqn_floodfill.pt \
  --network dueling \
  --fps 10

# Watch real-time training
python scripts/visualizer/visualize_snake.py \
  --mode training \
  --algorithm dqn \
  --episodes 500 \
  --num-envs 128
```

### Two-Snake Visualizer

Real-time visualization of competitive two-snake matches.

```bash
python scripts/visualizer/visualize_two_snake.py [OPTIONS]
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
python scripts/visualizer/visualize_two_snake.py --mode random --fps 10

# Trained agents compete
python scripts/visualizer/visualize_two_snake.py \
  --mode trained \
  --weights1 results/weights/ppo_coevolution/256x256_coevo.pt \
  --weights2 results/weights/ppo_coevolution/128x128_coevo.pt \
  --episodes 20

# Trained agent vs greedy scripted opponent
python scripts/visualizer/visualize_two_snake.py \
  --mode scripted \
  --weights1 results/weights/ppo_curriculum_256x256/stage3_final_256x256.pt \
  --opponent greedy
```

---

## Baseline Agents

### Random Agent

Selects actions uniformly at random. Used as a lower-bound baseline.

```python
from scripts.baselines import RandomAgent
agent = RandomAgent(action_space_type='relative', seed=67)
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
| **Basic** | 10 | Direction one-hot, relative food position, danger sensors |
| **Flood-fill** | 13 | Basic + reachable space in each direction |

### Grid-Based (CNN Networks)

- **Input Shape:** `(3, grid_size, grid_size)`
- **Channels:** Snake head, snake body, food
- **Spatial:** Full board representation

---

## Results

### Running Evaluation

```bash
python scripts/run_two_snake_experiments.py
python scripts/generate_score_plots.py
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

## Reproducibility

This section provides instructions to reproduce all results from scratch.

### Hardware Requirements

- **GPU:** NVIDIA GPU with CUDA support (tested with CUDA 13.0)
- **RAM:** 16GB+ recommended for vectorized training with 256 environments
- **Training Time:** ~2-4 hours per single-snake algorithm, ~6-8 hours for two-snake curriculum

### Step 1: Environment Setup

```bash
# Clone and setup
git clone https://github.com/elyashium/Snake-RL.git
cd Snake-RL
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Train Single-Snake Agents

Train all single-snake algorithms:

```bash
# Train all models at once
python scripts/training/train_all_models.py

# Or train individually
python scripts/training/train_dqn.py
python scripts/training/train_ppo.py
python scripts/training/train_a2c.py
python scripts/training/train_reinforce.py
python scripts/training/train_rainbow.py
```

Model weights are saved to `results/weights/`.

### Step 3: Train Two-Snake Agents

```bash
# PPO with curriculum learning (recommended)
python scripts/training/train_ppo_two_snake_mlp_curriculum.py

# DQN co-evolution
python scripts/training/train_two_snake_classic.py

# PPO direct co-evolution
python scripts/training/train_ppo_two_snake_mlp.py
```

### Step 4: Evaluate Models

Run evaluation scripts to generate metrics and figures:

```bash
python scripts/run_two_snake_experiments.py
python scripts/generate_score_plots.py
```

This generates:
- `results/figures/` - All comparison plots
- `results/data/evaluation_results_latest.json` - Full metrics
- `results/data/single_snake_summary_latest.csv` - Summary table

### Step 5: Verify Results

Run the test suite to verify environment and training correctness:

```bash
python -m pytest tests/ -v
```

### Pretrained Weights

Pretrained weights are included in `results/weights/` for immediate evaluation:

| Model | Path |
|-------|------|
| DQN Co-evolution | `results/weights/dqn_coevolution/` |
| DQN Direct Co-evolution | `results/weights/dqn_direct_coevolution/` |
| PPO Curriculum (128) | `results/weights/ppo_curriculum_128x128/` |
| PPO Curriculum (256) | `results/weights/ppo_curriculum_256x256/` |
| PPO Direct Co-evolution | `results/weights/ppo_direct_coevolution/` |
| PPO Co-evolution | `results/weights/ppo_coevolution/` |

### Reproducing Specific Experiments

#### Grid Size Comparison

```bash
python scripts/evaluate_grid_sizes.py
```

#### Extended Training (50k episodes)

```bash
python scripts/generate_extended_training_plot.py
```

#### Two-Snake Competition Evaluation

```bash
python scripts/run_two_snake_experiments.py
```

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_environment.py

# Run with verbose output
python -m pytest tests/ -v
```

---

## Acknowledgments

This project was developed for **ECEN 446** at Texas A&M University.

**Key Libraries:**
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Gymnasium](https://gymnasium.farama.org/) - RL environment interface
- [Pygame](https://www.pygame.org/) - Game visualization

---

## License

This project is for educational purposes as part of ECEN 446 coursework.
