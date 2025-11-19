"""
Controlled Comparison of Enhanced Features

Runs a fair comparison with:
- Equal network sizes (no auto-scaling)
- Longer training (500 episodes default)
- Same hyperparameters for both models
- Only difference: feature representation (14 vs 24 dims)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import time
import json
import torch
from scripts.training.train_dqn import DQNTrainer


def train_model(use_enhanced: bool, args, run_name: str = None):
    """Train a single model with controlled parameters"""
    feature_type = "enhanced" if use_enhanced else "baseline"
    feature_dim = 24 if use_enhanced else 14

    print('='*70)
    print(f'Training DQN-MLP with {feature_type.upper()} features ({feature_dim}-dim)')
    print('='*70)

    print(f'Configuration:')
    print(f'  Features: {feature_dim} ({feature_type})')
    print(f'  Episodes: {args.episodes}')
    print(f'  Parallel Environments: {args.envs}')
    print(f'  Max Steps: {args.max_steps}')
    print(f'  Hidden Dims: {args.hidden_dims} (SAME for both models)')
    print(f'  Learning Rate: {args.lr}')
    print(f'  Batch Size: {args.batch_size}')
    print(f'  Epsilon Decay: {args.epsilon_decay}')
    print('='*70)
    print()

    # Create trainer - NO AUTO-SCALING, same hyperparameters
    trainer = DQNTrainer(
        num_envs=args.envs,
        grid_size=10,
        action_space_type='relative',
        state_representation='feature',
        use_flood_fill=True,
        use_enhanced_features=use_enhanced,
        hidden_dims=tuple(args.hidden_dims),  # SAME for both
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        min_buffer_size=args.min_buffer_size,
        seed=args.seed
    )

    # Train
    print(f'Starting training for {feature_type} model...\n')
    start_time = time.time()
    trainer.train(verbose=True, log_interval=args.log_interval)
    end_time = time.time()

    # Save weights
    if run_name:
        save_name = f'dqn_{run_name}_{feature_type}.pt'
    else:
        save_name = f'dqn_mlp_{feature_type}_controlled.pt'

    trainer.save(save_name)
    actual_path = trainer.save_dir / save_name
    print(f'\nWeights saved to: {actual_path}')

    # Get training metrics
    duration = end_time - start_time
    stats = trainer.metrics.get_recent_stats()

    # Get learning curves (every 10 episodes)
    scores_curve = []
    rewards_curve = []
    for i in range(0, len(trainer.metrics.episode_scores), 10):
        end_idx = min(i + 10, len(trainer.metrics.episode_scores))
        scores_curve.append(sum(trainer.metrics.episode_scores[i:end_idx]) / (end_idx - i))
        rewards_curve.append(sum(trainer.metrics.episode_rewards[i:end_idx]) / (end_idx - i))

    results = {
        'feature_type': feature_type,
        'feature_dim': feature_dim,
        'hidden_dims': args.hidden_dims,
        'duration_seconds': duration,
        'final_avg_score': stats['avg_score'],
        'final_max_score': stats['max_score'],
        'final_avg_length': stats['avg_length'],
        'episodes': args.episodes,
        'episodes_per_second': args.episodes / duration,
        'scores_curve': scores_curve,
        'rewards_curve': rewards_curve,
        'final_epsilon': 1.0 * (args.epsilon_decay ** args.episodes)
    }

    return results, trainer


def print_comparison(baseline_results, enhanced_results):
    """Print detailed comparison"""
    print()
    print('='*70)
    print('CONTROLLED COMPARISON RESULTS')
    print('='*70)
    print()

    print('Baseline (Flood-fill only, 14 features):')
    print(f'  Final Avg Score: {baseline_results["final_avg_score"]:.2f}')
    print(f'  Final Max Score: {baseline_results["final_max_score"]}')
    print(f'  Final Avg Length: {baseline_results["final_avg_length"]:.2f}')
    print(f'  Training Time: {baseline_results["duration_seconds"]:.2f}s')
    print(f'  Hidden Dims: {baseline_results["hidden_dims"]}')
    print()

    print('Enhanced (Flood-fill + Enhanced, 24 features):')
    print(f'  Final Avg Score: {enhanced_results["final_avg_score"]:.2f}')
    print(f'  Final Max Score: {enhanced_results["final_max_score"]}')
    print(f'  Final Avg Length: {enhanced_results["final_avg_length"]:.2f}')
    print(f'  Training Time: {enhanced_results["duration_seconds"]:.2f}s')
    print(f'  Hidden Dims: {enhanced_results["hidden_dims"]}')
    print()

    print('Improvements:')
    score_improvement = enhanced_results["final_avg_score"] - baseline_results["final_avg_score"]
    score_pct = (score_improvement / max(baseline_results["final_avg_score"], 0.01)) * 100
    max_score_improvement = enhanced_results["final_max_score"] - baseline_results["final_max_score"]
    length_improvement = enhanced_results["final_avg_length"] - baseline_results["final_avg_length"]
    time_diff = enhanced_results["duration_seconds"] - baseline_results["duration_seconds"]
    time_pct = (time_diff / baseline_results["duration_seconds"]) * 100

    print(f'  Avg Score: {score_improvement:+.2f} ({score_pct:+.1f}%)')
    print(f'  Max Score: {max_score_improvement:+d}')
    print(f'  Avg Length: {length_improvement:+.2f}')
    print(f'  Training Time: {time_diff:+.2f}s ({time_pct:+.1f}%)')
    print()

    # Determine winner
    if score_improvement > 0.5:
        print('VERDICT: Enhanced features show SIGNIFICANT improvement!')
        print('The additional features (escape routes, tail-chasing, body awareness)')
        print('provide meaningful strategic advantages to the agent.')
    elif score_improvement > 0.2:
        print('VERDICT: Enhanced features show moderate improvement.')
        print('The additional features provide some benefit but may need more training.')
    elif score_improvement > -0.2:
        print('VERDICT: Performance is similar (no significant difference).')
        print('Both models perform comparably. More training may reveal differences.')
    else:
        print('VERDICT: Baseline performs better.')
        print('The enhanced features may need different hyperparameters or more training.')

    print('='*70)


def main():
    parser = argparse.ArgumentParser(description='Controlled comparison of enhanced features')

    # Training config
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes to train (default: 500)')
    parser.add_argument('--envs', type=int, default=256, help='Number of parallel environments')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')

    # Network config - SAME for both models
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128],
                        help='Hidden layer dimensions (SAME for both models)')

    # DQN config
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--target-update-freq', type=int, default=1000,
                        help='Target network update frequency')
    parser.add_argument('--min-buffer-size', type=int, default=1000,
                        help='Minimum buffer size before training')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=50, help='Logging interval')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Run name for saving results')

    args = parser.parse_args()

    print('='*70)
    print('CONTROLLED ENHANCED FEATURES COMPARISON')
    print('='*70)
    print(f'Comparing baseline (14-dim) vs enhanced (24-dim) features')
    print(f'CONTROLLED CONDITIONS:')
    print(f'  - Same network size: {args.hidden_dims}')
    print(f'  - Same hyperparameters')
    print(f'  - Same seed: {args.seed}')
    print(f'  - Episodes per model: {args.episodes}')
    print(f'  - Only difference: Feature representation')
    print('='*70)
    print()

    results = {}

    # Train baseline
    print('TRAINING BASELINE MODEL (14-dim flood-fill only)...\n')
    baseline_results, baseline_trainer = train_model(False, args, args.run_name)
    results['baseline'] = baseline_results

    print()
    print('='*70)
    print()

    # Train enhanced
    print('TRAINING ENHANCED MODEL (24-dim with all features)...\n')
    enhanced_results, enhanced_trainer = train_model(True, args, args.run_name)
    results['enhanced'] = enhanced_results

    # Compare results
    print_comparison(baseline_results, enhanced_results)

    # Save comparison results
    if args.run_name:
        comparison_file = Path(f'results/data/controlled_comparison_{args.run_name}.json')
    else:
        comparison_file = Path('results/data/controlled_comparison.json')

    comparison_file.parent.mkdir(parents=True, exist_ok=True)

    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nComparison results saved to: {comparison_file}')


if __name__ == '__main__':
    main()
