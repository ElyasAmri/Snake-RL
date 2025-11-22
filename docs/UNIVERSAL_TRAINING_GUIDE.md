# Universal Training Script Guide

## Overview

The universal training script (`scripts/training/train.py`) is a single, comprehensive script that replaces all 28+ specialized training scripts. It supports all algorithms, network types, and feature configurations through command-line arguments.

## Quick Start

```bash
# Basic DQN with MLP
./venv/Scripts/python.exe scripts/training/train.py --algorithm dqn --network mlp --features basic --episodes 1000

# PPO with flood-fill features
./venv/Scripts/python.exe scripts/training/train.py --algorithm ppo --network mlp --features flood --episodes 1000

# Double Dueling DQN with selective features
./venv/Scripts/python.exe scripts/training/train.py --algorithm dqn --network mlp --features selective --use-double-dqn --use-dueling --episodes 2000
```

## Available Options

### List Features
```bash
./venv/Scripts/python.exe scripts/training/train.py --list-algorithms
./venv/Scripts/python.exe scripts/training/train.py --list-features
```

### Algorithms
- **dqn**: Deep Q-Network (supports Double, Dueling, Noisy, PER variants)
- **ppo**: Proximal Policy Optimization
- **reinforce**: REINFORCE policy gradient
- **a2c**: Advantage Actor-Critic

### Network Types
- **mlp**: Multi-Layer Perceptron (feature-based state representation)
- **cnn**: Convolutional Neural Network (grid-based state representation)

### Feature Sets
- **basic** (11 features): Danger detection + food direction + current direction
- **flood** (14 features): Basic + flood-fill free space calculation
- **selective** (19 features): Flood-fill + tail direction + tail reachability
- **enhanced** (24 features): All available features (most comprehensive)

## Common Arguments

```bash
--algorithm {dqn,ppo,reinforce,a2c}  # Required: Which algorithm to use
--network {mlp,cnn}                   # Network architecture (default: mlp)
--features {basic,flood,selective,enhanced}  # Feature set (default: basic)
--episodes INT                        # Number of training episodes (default: 1000)
--envs INT                            # Parallel environments (default: 256)
--max-steps INT                       # Max steps per episode (default: 1000)
--hidden-dims INT [INT ...]           # Hidden layer sizes (default: 128 128)
--gamma FLOAT                         # Discount factor (default: 0.99)
--seed INT                            # Random seed (default: 42)
--log-interval INT                    # Episodes between logs (default: 100)
--save-dir STR                        # Where to save weights (default: results/weights)
```

## DQN-Specific Arguments

```bash
--dqn-lr FLOAT                        # Learning rate (default: 0.001)
--batch-size INT                      # Batch size (default: 64)
--buffer-size INT                     # Replay buffer size (default: 100000)
--epsilon-start FLOAT                 # Initial epsilon (default: 1.0)
--epsilon-end FLOAT                   # Final epsilon (default: 0.01)
--epsilon-decay FLOAT                 # Epsilon decay (default: 0.995)
--target-update-freq INT              # Target network update freq (default: 1000)
--train-steps-ratio FLOAT             # Training steps per transition (default: 0.03125)

# DQN Variants
--use-double-dqn                      # Enable Double DQN
--use-dueling                         # Enable Dueling architecture (MLP only)
--use-noisy                           # Enable Noisy Networks (MLP only)
--use-per                             # Enable Prioritized Experience Replay
--per-alpha FLOAT                     # PER alpha (default: 0.6)
--per-beta-start FLOAT                # PER beta start (default: 0.4)
```

## PPO-Specific Arguments

```bash
--actor-lr FLOAT                      # Actor learning rate (default: 0.0003)
--critic-lr FLOAT                     # Critic learning rate (default: 0.001)
--rollout-steps INT                   # Steps per rollout (default: 128)
--epochs-per-rollout INT              # Training epochs per rollout (default: 4)
--gae-lambda FLOAT                    # GAE lambda (default: 0.95)
--clip-epsilon FLOAT                  # PPO clip epsilon (default: 0.2)
--value-loss-coef FLOAT               # Value loss coefficient (default: 0.5)
--entropy-coef FLOAT                  # Entropy coefficient (default: 0.01)
```

## REINFORCE-Specific Arguments

```bash
--reinforce-lr FLOAT                  # Learning rate (default: 0.001)
--entropy-coef FLOAT                  # Entropy coefficient (default: 0.01)
```

## A2C-Specific Arguments

```bash
--actor-lr FLOAT                      # Actor learning rate (default: 0.0003)
--critic-lr FLOAT                     # Critic learning rate (default: 0.001)
--a2c-rollout-steps INT               # Rollout steps (default: 5)
--value-coef FLOAT                    # Value loss coefficient (default: 0.5)
--entropy-coef FLOAT                  # Entropy coefficient (default: 0.01)
```

## Usage Examples

### Basic Training

```bash
# Train basic DQN for 1000 episodes
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network mlp --features basic --episodes 1000

# Train PPO with flood-fill features
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm ppo --network mlp --features flood --episodes 1000

# Train REINFORCE with enhanced features
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm reinforce --network mlp --features enhanced --episodes 500
```

### DQN Variants

```bash
# Double DQN
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network mlp --features basic \
    --use-double-dqn --episodes 1000

# Dueling DQN (MLP only)
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network mlp --features selective \
    --use-dueling --episodes 1000

# Double Dueling DQN with PER
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network mlp --features selective \
    --use-double-dqn --use-dueling --use-per --episodes 2000

# Noisy DQN (MLP only, no epsilon-greedy needed)
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network mlp --features enhanced \
    --use-noisy --episodes 1000
```

### Custom Network Architecture

```bash
# Large 4-layer network
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network mlp --features enhanced \
    --hidden-dims 256 256 128 128 --episodes 2000

# Small fast network for quick iteration
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm ppo --network mlp --features basic \
    --hidden-dims 64 64 --episodes 500 --envs 128
```

### Speed Control

```bash
# Fast training (good for iteration)
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network mlp --features basic \
    --train-steps-ratio 0.03125 --episodes 10000

# Quality training (better convergence)
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network mlp --features basic \
    --train-steps-ratio 0.25 --episodes 10000
```

### CNN Networks

```bash
# DQN with CNN
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network cnn --features basic --episodes 1000

# PPO with CNN and flood-fill features
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm ppo --network cnn --features flood --episodes 1000
```

## Output Files

The script generates two files per training run:

1. **Model weights**: `{algorithm}_{variants}_{network}_{features}_{dims}_{episodes}ep_{timestamp}.pt`
   - Example: `dqn_double_dueling_mlp_selective_128x128_10000ep_20251122_183045.pt`

2. **Configuration JSON**: Same name with `_config.json` suffix
   - Contains all command-line arguments for reproducibility
   - Example: `dqn_double_dueling_mlp_selective_128x128_10000ep_20251122_183045_config.json`

## Validation & Error Handling

The script validates configurations before training:

```bash
# ERROR: Noisy DQN requires MLP
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network cnn --use-noisy
# ValueError: Noisy DQN only supports MLP networks. Use --network mlp or remove --use-noisy

# ERROR: Dueling DQN requires MLP
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network cnn --use-dueling
# ValueError: Dueling DQN only supports MLP networks. Use --network mlp or remove --use-dueling
```

## Dry Run Mode

Validate configuration without training:

```bash
./venv/Scripts/python.exe scripts/training/train.py \
    --algorithm dqn --network mlp --features selective \
    --use-double-dqn --use-dueling --dry-run
# Validates configuration and exits without training
```

## Compatibility Matrix

| Feature | DQN | DQN+Double | DQN+Dueling | DQN+Noisy | DQN+PER | PPO | REINFORCE | A2C |
|---------|-----|------------|-------------|-----------|---------|-----|-----------|-----|
| MLP + Basic | YES | YES | YES | YES | YES | YES | YES | YES |
| MLP + Flood | YES | YES | YES | YES | YES | YES | YES | YES |
| MLP + Selective | YES | YES | YES | YES | YES | YES | YES | YES |
| MLP + Enhanced | YES | YES | YES | YES | YES | YES | YES | YES |
| CNN + Basic | YES | YES | NO | NO | YES | YES | YES | YES |
| CNN + Flood | YES | YES | NO | NO | YES | YES | YES | YES |
| CNN + Selective | YES | YES | NO | NO | YES | YES | YES | YES |
| CNN + Enhanced | YES | YES | NO | NO | YES | YES | YES | YES |

**Key:**
- YES = Supported
- NO = Not supported (Dueling and Noisy DQN require MLP)

## Tips & Best Practices

1. **Start with basic features** for quick iteration, then upgrade to selective/enhanced
2. **Use --dry-run** to validate complex configurations
3. **Check generated config.json** to verify all parameters are as expected
4. **Use descriptive filenames** - they're auto-generated based on your configuration
5. **Experiment with train-steps-ratio** - lower = faster, higher = better quality
6. **Save important configurations** - the config.json makes experiments reproducible

## Troubleshooting

**Problem**: Training is too slow
**Solution**: Reduce `--train-steps-ratio` or use fewer `--envs`

**Problem**: Want to reproduce a previous run
**Solution**: Load the saved `_config.json` and use those exact arguments

**Problem**: Noisy DQN not working with CNN
**Solution**: Noisy DQN is only implemented for MLP networks. Use `--network mlp`

**Problem**: Selective features not improving performance
**Solution**: Selective features require more training time. Increase `--episodes` or `--train-steps-ratio`

## Migration from Old Scripts

Old specialized scripts can be replaced with universal script:

```bash
# OLD: scripts/training/train_dqn_mlp.py --episodes 1000
# NEW:
./venv/Scripts/python.exe scripts/training/train.py --algorithm dqn --network mlp --features basic --episodes 1000

# OLD: scripts/training/train_double_dqn_mlp_floodfill.py --episodes 2000
# NEW:
./venv/Scripts/python.exe scripts/training/train.py --algorithm dqn --network mlp --features flood --use-double-dqn --episodes 2000

# OLD: scripts/training/train_ppo_mlp.py --episodes 1000
# NEW:
./venv/Scripts/python.exe scripts/training/train.py --algorithm ppo --network mlp --features basic --episodes 1000
```

## Summary

The universal training script provides:
- Single source of truth for all training
- Consistent interface across algorithms
- Automatic filename generation
- Configuration tracking via JSON
- Comprehensive validation
- Easy experimentation
- Better maintainability

**Replaces**: 28+ specialized training scripts
**Lines of code**: ~550 (vs ~10,000+ across all old scripts)
**Maintenance**: Fix bugs in one place
**User experience**: Simpler, more consistent
