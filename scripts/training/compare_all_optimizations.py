"""
Comprehensive Comparison of All Optimizations

Tests 4 configurations:
1. Baseline: Flood-fill only (14-dim, [128, 128])
2. Selective: Flood-fill + tail features (19-dim, [128, 128])
3. Enhanced: All features (24-dim, [128, 128])
4. Enhanced-Large: All features (24-dim, [256, 256])

All with optimized BFS performance.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import time
import json
import torch
from scripts.training.train_dqn import DQNTrainer


def train_model(config_name, use_selective, use_enhanced, hidden_dims, args):
    """Train a single model with specified configuration"""

    if use_enhanced:
        feature_dim = 24
        feature_desc = "All enhanced features"
    elif use_selective:
        feature_dim = 19
        feature_desc = "Selective (tail features)"
    else:
        feature_dim = 14
        feature_desc = "Baseline (flood-fill only)"

    print('='*70)
    print(f'Training: {config_name}')
    print('='*70)
    print(f'  Features: {feature_dim}-dim ({feature_desc})')
    print(f'  Hidden Dims: {hidden_dims}')
    print(f'  Episodes: {args.episodes}')
    print(f'  Optimizations: BFS with deque + early termination')
    print('='*70)
    print()

    # Create trainer
    trainer = DQNTrainer(
        num_envs=args.envs,
        grid_size=10,
        action_space_type='relative',
        state_representation='feature',
        use_flood_fill=True,
        use_selective_features=use_selective,
        use_enhanced_features=use_enhanced,
        hidden_dims=tuple(hidden_dims),
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
    print(f'Starting training for {config_name}...\n')
    start_time = time.time()
    trainer.train(verbose=True, log_interval=args.log_interval)
    end_time = time.time()

    # Save weights
    save_name = f'dqn_{args.run_name}_{config_name}.pt'
    trainer.save(save_name)
    actual_path = trainer.save_dir / save_name
    print(f'\nWeights saved to: {actual_path}')

    # Get metrics
    duration = end_time - start_time
    stats = trainer.metrics.get_recent_stats()

    # Get learning curves
    scores_curve = []
    for i in range(0, len(trainer.metrics.episode_scores), 10):
        end_idx = min(i + 10, len(trainer.metrics.episode_scores))
        scores_curve.append(sum(trainer.metrics.episode_scores[i:end_idx]) / (end_idx - i))

    results = {
        'config_name': config_name,
        'feature_dim': feature_dim,
        'feature_desc': feature_desc,
        'hidden_dims': hidden_dims,
        'duration_seconds': duration,
        'final_avg_score': stats['avg_score'],
        'final_max_score': stats['max_score'],
        'final_avg_length': stats['avg_length'],
        'episodes': args.episodes,
        'scores_curve': scores_curve
    }

    return results


def print_comparison(results_dict):
    """Print detailed comparison of all models"""
    print()
    print('='*70)
    print('COMPREHENSIVE COMPARISON RESULTS')
    print('='*70)
    print()

    # Sort by final score (descending)
    sorted_results = sorted(results_dict.items(), key=lambda x: x[1]['final_avg_score'], reverse=True)

    for rank, (name, res) in enumerate(sorted_results, 1):
        print(f'{rank}. {res["config_name"]}')
        print(f'   Features: {res["feature_dim"]}-dim ({res["feature_desc"]})')
        print(f'   Network: {res["hidden_dims"]}')
        print(f'   Final Avg Score: {res["final_avg_score"]:.2f}')
        print(f'   Final Max Score: {res["final_max_score"]}')
        print(f'   Training Time: {res["duration_seconds"]:.2f}s')
        print()

    # Compare to baseline
    baseline = results_dict['baseline']
    print('Improvements vs Baseline:')
    print()

    for name, res in results_dict.items():
        if name == 'baseline':
            continue

        score_diff = res['final_avg_score'] - baseline['final_avg_score']
        score_pct = (score_diff / max(baseline['final_avg_score'], 0.01)) * 100
        time_diff = res['duration_seconds'] - baseline['duration_seconds']
        time_pct = (time_diff / baseline['duration_seconds']) * 100

        print(f'{res["config_name"]}:')
        print(f'  Avg Score: {score_diff:+.2f} ({score_pct:+.1f}%)')
        print(f'  Max Score: {res["final_max_score"] - baseline["final_max_score"]:+d}')
        print(f'  Training Time: {time_diff:+.2f}s ({time_pct:+.1f}%)')
        print()

    # Winner
    winner = sorted_results[0][1]
    print('='*70)
    print(f'WINNER: {winner["config_name"]}')
    print(f'  Score: {winner["final_avg_score"]:.2f}')
    print(f'  Configuration: {winner["feature_dim"]}-dim, {winner["hidden_dims"]}')
    print('='*70)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive comparison of all optimizations')

    # Training config
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes (default: 1000 for thorough test)')
    parser.add_argument('--envs', type=int, default=256)
    parser.add_argument('--max-steps', type=int, default=500)

    # DQN config
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-end', type=float, default=0.01)
    parser.add_argument('--epsilon-decay', type=float, default=0.995)
    parser.add_argument('--target-update-freq', type=int, default=1000)
    parser.add_argument('--min-buffer-size', type=int, default=1000)

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--run-name', type=str, default='optimized',
                        help='Run name for saving results')

    args = parser.parse_args()

    print('='*70)
    print('COMPREHENSIVE OPTIMIZATION COMPARISON')
    print('='*70)
    print(f'Testing 4 configurations with {args.episodes} episodes each:')
    print(f'  1. Baseline: 14-dim, [128, 128]')
    print(f'  2. Selective: 19-dim, [128, 128] (tail features only)')
    print(f'  3. Enhanced: 24-dim, [128, 128] (all features)')
    print(f'  4. Enhanced-Large: 24-dim, [256, 256] (all features, larger network)')
    print(f'All with optimized BFS (deque + early termination)')
    print('='*70)
    print()

    results = {}

    # 1. Baseline
    print('\n[1/4] TRAINING BASELINE...\n')
    results['baseline'] = train_model('baseline', False, False, [128, 128], args)

    print('\n' + '='*70 + '\n')

    # 2. Selective
    print('\n[2/4] TRAINING SELECTIVE...\n')
    results['selective'] = train_model('selective', True, False, [128, 128], args)

    print('\n' + '='*70 + '\n')

    # 3. Enhanced
    print('\n[3/4] TRAINING ENHANCED...\n')
    results['enhanced'] = train_model('enhanced', False, True, [128, 128], args)

    print('\n' + '='*70 + '\n')

    # 4. Enhanced-Large
    print('\n[4/4] TRAINING ENHANCED-LARGE...\n')
    results['enhanced_large'] = train_model('enhanced_large', False, True, [256, 256], args)

    # Print comparison
    print_comparison(results)

    # Save results
    comparison_file = Path(f'results/data/optimization_comparison_{args.run_name}.json')
    comparison_file.parent.mkdir(parents=True, exist_ok=True)

    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nResults saved to: {comparison_file}')


if __name__ == '__main__':
    main()
