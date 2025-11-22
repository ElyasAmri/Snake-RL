"""
Universal Training Script for Snake RL

A single script that supports all algorithms, network types, and feature configurations.

Usage Examples:
    # Basic DQN with MLP
    python train.py --algorithm dqn --network mlp --features basic --episodes 1000

    # PPO with flood-fill features
    python train.py --algorithm ppo --network mlp --features flood --episodes 1000

    # Double Dueling DQN with PER and selective features
    python train.py --algorithm dqn --network mlp --features selective \\
        --use-double-dqn --use-dueling --use-per --episodes 2000

    # List available options
    python train.py --list-algorithms
    python train.py --list-features

Supported Algorithms:
    - dqn: Deep Q-Network (with optional Double, Dueling, Noisy, PER variants)
    - ppo: Proximal Policy Optimization
    - reinforce: REINFORCE policy gradient
    - a2c: Advantage Actor-Critic

Supported Features:
    - basic (11 features): Danger detection, food direction, current direction
    - flood (14 features): Basic + flood-fill free space
    - selective (19 features): Flood-fill + tail features (best balance)
    - enhanced (24 features): All features (most comprehensive)

Supported Networks:
    - mlp: Multi-Layer Perceptron (feature-based state)
    - cnn: Convolutional Neural Network (grid-based state)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any

from scripts.training.train_dqn import DQNTrainer
from scripts.training.train_ppo import PPOTrainer
from scripts.training.train_reinforce import REINFORCETrainer
from scripts.training.train_a2c import A2CTrainer


# ============================================================================
# CONFIGURATION REGISTRIES
# ============================================================================

ALGORITHM_REGISTRY = {
    'dqn': DQNTrainer,
    'ppo': PPOTrainer,
    'reinforce': REINFORCETrainer,
    'a2c': A2CTrainer
}

FEATURE_CONFIG = {
    'basic': {
        'use_flood_fill': False,
        'use_selective_features': False,
        'use_enhanced_features': False,
        'input_dim': 11,
        'description': 'Danger detection + food direction + current direction'
    },
    'flood': {
        'use_flood_fill': True,
        'use_selective_features': False,
        'use_enhanced_features': False,
        'input_dim': 14,
        'description': 'Basic features + flood-fill free space'
    },
    'selective': {
        'use_flood_fill': True,
        'use_selective_features': True,
        'use_enhanced_features': False,
        'input_dim': 19,
        'description': 'Flood-fill + tail direction + tail reachability'
    },
    'enhanced': {
        'use_flood_fill': True,
        'use_selective_features': False,
        'use_enhanced_features': True,
        'input_dim': 24,
        'description': 'All available features (most comprehensive)'
    }
}

NETWORK_TO_STATE = {
    'mlp': 'feature',
    'cnn': 'grid'
}

ALGORITHM_DESCRIPTIONS = {
    'dqn': 'Deep Q-Network - Value-based RL with experience replay',
    'ppo': 'Proximal Policy Optimization - Policy gradient with clipping',
    'reinforce': 'REINFORCE - Monte Carlo policy gradient',
    'a2c': 'Advantage Actor-Critic - On-policy actor-critic'
}


# ============================================================================
# VALIDATION
# ============================================================================

def validate_configuration(args):
    """Validate argument combinations and raise helpful errors"""

    # Check algorithm exists
    if args.algorithm not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{args.algorithm}'. "
            f"Available: {', '.join(ALGORITHM_REGISTRY.keys())}"
        )

    # Check feature exists
    if args.features not in FEATURE_CONFIG:
        raise ValueError(
            f"Unknown feature set '{args.features}'. "
            f"Available: {', '.join(FEATURE_CONFIG.keys())}"
        )

    # Check network type
    if args.network not in NETWORK_TO_STATE:
        raise ValueError(
            f"Unknown network type '{args.network}'. "
            f"Available: {', '.join(NETWORK_TO_STATE.keys())}"
        )

    # DQN-specific validations
    if args.algorithm == 'dqn':
        if args.use_noisy and args.network == 'cnn':
            raise ValueError(
                "Noisy DQN only supports MLP networks. "
                "Use --network mlp or remove --use-noisy"
            )

        if args.use_dueling and args.network == 'cnn':
            raise ValueError(
                "Dueling DQN only supports MLP networks. "
                "Use --network mlp or remove --use-dueling"
            )

    # Warn about ignored parameters
    if args.algorithm != 'dqn':
        dqn_only_flags = []
        if args.use_double_dqn:
            dqn_only_flags.append('--use-double-dqn')
        if args.use_dueling:
            dqn_only_flags.append('--use-dueling')
        if args.use_noisy:
            dqn_only_flags.append('--use-noisy')
        if args.use_per:
            dqn_only_flags.append('--use-per')

        if dqn_only_flags:
            print(f"WARNING: {', '.join(dqn_only_flags)} are DQN-specific and will be ignored", flush=True)


# ============================================================================
# CONFIGURATION BUILDER
# ============================================================================

def build_config(args) -> Dict[str, Any]:
    """Build trainer configuration from CLI arguments"""

    # Common configuration
    config = {
        'num_envs': args.envs,
        'grid_size': 10,
        'action_space_type': 'relative',
        'state_representation': NETWORK_TO_STATE[args.network],
        'num_episodes': args.episodes,
        'max_steps': args.max_steps,
        'seed': args.seed,
        'hidden_dims': tuple(args.hidden_dims)
    }

    # Add feature configuration
    feature_config = FEATURE_CONFIG[args.features]
    config['use_flood_fill'] = feature_config['use_flood_fill']
    config['use_selective_features'] = feature_config['use_selective_features']
    config['use_enhanced_features'] = feature_config['use_enhanced_features']

    # Algorithm-specific parameters
    if args.algorithm == 'dqn':
        config.update({
            'learning_rate': args.dqn_lr,
            'batch_size': args.batch_size,
            'buffer_size': args.buffer_size,
            'gamma': args.gamma,
            'epsilon_start': args.epsilon_start,
            'epsilon_end': args.epsilon_end,
            'epsilon_decay': args.epsilon_decay,
            'target_update_freq': args.target_update_freq,
            'min_buffer_size': args.min_buffer_size,
            'train_steps_ratio': args.train_steps_ratio,
            'use_double_dqn': args.use_double_dqn,
            'use_dueling': args.use_dueling,
            'use_noisy': args.use_noisy,
            'use_prioritized_replay': args.use_per
        })

        if args.use_per:
            config['per_alpha'] = args.per_alpha
            config['per_beta_start'] = args.per_beta_start

    elif args.algorithm == 'ppo':
        config.update({
            'actor_lr': args.actor_lr,
            'critic_lr': args.critic_lr,
            'gamma': args.gamma,
            'rollout_steps': args.rollout_steps,
            'batch_size': args.batch_size,
            'epochs_per_rollout': args.epochs_per_rollout,
            'gae_lambda': args.gae_lambda,
            'clip_epsilon': args.clip_epsilon,
            'value_loss_coef': args.value_loss_coef,
            'entropy_coef': args.entropy_coef
        })

    elif args.algorithm == 'reinforce':
        config.update({
            'learning_rate': args.reinforce_lr,
            'gamma': args.gamma,
            'entropy_coef': args.entropy_coef
        })

    elif args.algorithm == 'a2c':
        config.update({
            'actor_lr': args.actor_lr,
            'critic_lr': args.critic_lr,
            'gamma': args.gamma,
            'rollout_steps': args.a2c_rollout_steps,
            'entropy_coef': args.entropy_coef,
            'value_coef': args.value_coef
        })

    return config


# ============================================================================
# FILENAME GENERATION
# ============================================================================

def generate_filename(args, timestamp: str) -> str:
    """Generate descriptive filename from configuration"""
    parts = []

    # Algorithm
    parts.append(args.algorithm)

    # DQN variants
    if args.algorithm == 'dqn':
        if args.use_double_dqn:
            parts.append('double')
        if args.use_dueling:
            parts.append('dueling')
        if args.use_noisy:
            parts.append('noisy')
        if args.use_per:
            parts.append('per')

    # Network type
    parts.append(args.network)

    # Features
    parts.append(args.features)

    # Hidden dimensions
    dims_str = 'x'.join(map(str, args.hidden_dims))
    parts.append(dims_str)

    # Episodes
    parts.append(f"{args.episodes}ep")

    # Timestamp
    parts.append(timestamp)

    return '_'.join(parts) + '.pt'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def list_algorithms():
    """Print available algorithms"""
    print("\n" + "="*80)
    print("AVAILABLE ALGORITHMS")
    print("="*80)
    for name, description in ALGORITHM_DESCRIPTIONS.items():
        print(f"  {name:<12} - {description}")
    print("="*80 + "\n")


def list_features():
    """Print available feature sets"""
    print("\n" + "="*80)
    print("AVAILABLE FEATURE SETS")
    print("="*80)
    for name, config in FEATURE_CONFIG.items():
        print(f"  {name:<12} ({config['input_dim']:2d} features) - {config['description']}")
    print("="*80 + "\n")


def print_configuration(args, config):
    """Print training configuration banner"""
    print("\n" + "="*80, flush=True)
    print("TRAINING CONFIGURATION", flush=True)
    print("="*80, flush=True)
    print(f"Algorithm: {args.algorithm.upper()}", flush=True)
    print(f"Network: {args.network.upper()}", flush=True)
    print(f"Features: {args.features} ({FEATURE_CONFIG[args.features]['input_dim']} dimensions)", flush=True)
    print(f"Hidden Layers: {args.hidden_dims}", flush=True)
    print(f"Episodes: {args.episodes}", flush=True)
    print(f"Parallel Environments: {args.envs}", flush=True)
    print(f"Max Steps per Episode: {args.max_steps}", flush=True)

    # Algorithm-specific details
    if args.algorithm == 'dqn':
        variants = []
        if args.use_double_dqn:
            variants.append("Double DQN")
        if args.use_dueling:
            variants.append("Dueling")
        if args.use_noisy:
            variants.append("Noisy")
        if args.use_per:
            variants.append("PER")

        if variants:
            print(f"DQN Variants: {', '.join(variants)}", flush=True)
        print(f"Learning Rate: {args.dqn_lr}", flush=True)
        print(f"Batch Size: {args.batch_size}", flush=True)
        print(f"Train Steps Ratio: {args.train_steps_ratio}", flush=True)

    elif args.algorithm == 'ppo':
        print(f"Actor LR: {args.actor_lr}, Critic LR: {args.critic_lr}", flush=True)
        print(f"Rollout Steps: {args.rollout_steps}", flush=True)
        print(f"Clip Epsilon: {args.clip_epsilon}", flush=True)

    elif args.algorithm == 'reinforce':
        print(f"Learning Rate: {args.reinforce_lr}", flush=True)

    elif args.algorithm == 'a2c':
        print(f"Actor LR: {args.actor_lr}, Critic LR: {args.critic_lr}", flush=True)
        print(f"Rollout Steps: {args.a2c_rollout_steps}", flush=True)

    print("="*80, flush=True)
    print(flush=True)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def create_parser():
    """Create comprehensive argument parser"""
    parser = argparse.ArgumentParser(
        description='Universal Training Script for Snake RL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Helper flags
    parser.add_argument('--list-algorithms', action='store_true',
                       help='List available algorithms and exit')
    parser.add_argument('--list-features', action='store_true',
                       help='List available feature sets and exit')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without training')

    # Core arguments
    parser.add_argument('--algorithm', type=str, required='--list-algorithms' not in sys.argv and '--list-features' not in sys.argv,
                       choices=['dqn', 'ppo', 'reinforce', 'a2c'],
                       help='Training algorithm')
    parser.add_argument('--network', type=str, default='mlp',
                       choices=['mlp', 'cnn'],
                       help='Network architecture type')
    parser.add_argument('--features', type=str, default='basic',
                       choices=['basic', 'flood', 'selective', 'enhanced'],
                       help='Feature set to use')

    # Training arguments
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train')
    parser.add_argument('--envs', type=int, default=256,
                       help='Number of parallel environments')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Episodes between logging')
    parser.add_argument('--save-dir', type=str, default='results/weights',
                       help='Directory to save weights')

    # Network arguments
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128],
                       help='Hidden layer dimensions')

    # DQN-specific arguments
    dqn_group = parser.add_argument_group('DQN Options')
    dqn_group.add_argument('--dqn-lr', type=float, default=0.001,
                          help='DQN learning rate')
    dqn_group.add_argument('--batch-size', type=int, default=64,
                          help='Batch size for training')
    dqn_group.add_argument('--buffer-size', type=int, default=100000,
                          help='Replay buffer size')
    dqn_group.add_argument('--epsilon-start', type=float, default=1.0,
                          help='Initial epsilon for exploration')
    dqn_group.add_argument('--epsilon-end', type=float, default=0.01,
                          help='Final epsilon')
    dqn_group.add_argument('--epsilon-decay', type=float, default=0.995,
                          help='Epsilon decay rate')
    dqn_group.add_argument('--target-update-freq', type=int, default=1000,
                          help='Target network update frequency')
    dqn_group.add_argument('--min-buffer-size', type=int, default=1000,
                          help='Minimum buffer size before training')
    dqn_group.add_argument('--train-steps-ratio', type=float, default=0.03125,
                          help='Training steps per collected transition')
    dqn_group.add_argument('--use-double-dqn', action='store_true',
                          help='Enable Double DQN')
    dqn_group.add_argument('--use-dueling', action='store_true',
                          help='Enable Dueling DQN architecture')
    dqn_group.add_argument('--use-noisy', action='store_true',
                          help='Enable Noisy Networks (MLP only)')
    dqn_group.add_argument('--use-per', action='store_true',
                          help='Enable Prioritized Experience Replay')
    dqn_group.add_argument('--per-alpha', type=float, default=0.6,
                          help='PER alpha parameter')
    dqn_group.add_argument('--per-beta-start', type=float, default=0.4,
                          help='PER beta start value')

    # PPO-specific arguments
    ppo_group = parser.add_argument_group('PPO Options')
    ppo_group.add_argument('--actor-lr', type=float, default=0.0003,
                          help='Actor learning rate')
    ppo_group.add_argument('--critic-lr', type=float, default=0.001,
                          help='Critic learning rate')
    ppo_group.add_argument('--rollout-steps', type=int, default=128,
                          help='Steps per rollout')
    ppo_group.add_argument('--epochs-per-rollout', type=int, default=4,
                          help='Training epochs per rollout')
    ppo_group.add_argument('--gae-lambda', type=float, default=0.95,
                          help='GAE lambda parameter')
    ppo_group.add_argument('--clip-epsilon', type=float, default=0.2,
                          help='PPO clip epsilon')
    ppo_group.add_argument('--value-loss-coef', type=float, default=0.5,
                          help='Value loss coefficient')
    ppo_group.add_argument('--entropy-coef', type=float, default=0.01,
                          help='Entropy coefficient')

    # REINFORCE-specific arguments
    reinforce_group = parser.add_argument_group('REINFORCE Options')
    reinforce_group.add_argument('--reinforce-lr', type=float, default=0.001,
                                help='REINFORCE learning rate')

    # A2C-specific arguments
    a2c_group = parser.add_argument_group('A2C Options')
    a2c_group.add_argument('--a2c-rollout-steps', type=int, default=5,
                          help='A2C rollout steps')
    a2c_group.add_argument('--value-coef', type=float, default=0.5,
                          help='Value loss coefficient')

    return parser


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training orchestration"""
    parser = create_parser()
    args = parser.parse_args()

    # Handle helper flags
    if args.list_algorithms:
        list_algorithms()
        return

    if args.list_features:
        list_features()
        return

    # Validate configuration
    validate_configuration(args)

    # Build configuration
    config = build_config(args)

    # Print configuration
    print_configuration(args, config)

    # Dry run - exit after validation
    if args.dry_run:
        print("DRY RUN - Configuration validated successfully. Exiting without training.", flush=True)
        return

    # Create trainer
    TrainerClass = ALGORITHM_REGISTRY[args.algorithm]
    trainer = TrainerClass(**config)

    # Train
    start_time = time.time()
    trainer.train(verbose=True, log_interval=args.log_interval)
    elapsed = time.time() - start_time

    # Generate filename and save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = generate_filename(args, timestamp)
    trainer.save(filename)

    # Save configuration JSON
    config_filename = filename.replace('.pt', '_config.json')
    config_path = Path(args.save_dir) / config_filename
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Print summary
    stats = trainer.metrics.get_recent_stats()
    print("\n" + "="*80, flush=True)
    print("TRAINING COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Total Time: {elapsed/60:.1f} minutes ({elapsed:.1f}s)", flush=True)
    print(f"Episodes: {args.episodes}", flush=True)
    print(f"Final Avg Score: {stats.get('avg_score', 0):.2f}", flush=True)
    print(f"Final Max Score: {stats.get('max_score', 0)}", flush=True)
    print(f"Model saved: {Path(args.save_dir) / filename}", flush=True)
    print(f"Config saved: {config_path}", flush=True)
    print("="*80, flush=True)


if __name__ == '__main__':
    main()
