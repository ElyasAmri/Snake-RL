"""
Train REINFORCE with CNN (Grid-based State)

Standalone script for training REINFORCE with 10x10x3 grid state representation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import time
from scripts.training.train_reinforce import REINFORCETrainer


def main():
    parser = argparse.ArgumentParser(description='Train REINFORCE-CNN on Snake game')

    # Training config
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--envs', type=int, default=256, help='Number of parallel environments')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')

    # REINFORCE config
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=50, help='Logging interval')
    parser.add_argument('--save-path', type=str, default='results/weights/reinforce_cnn.pt', help='Path to save weights')

    args = parser.parse_args()

    print('='*70)
    print('Training REINFORCE-CNN (Grid-based State)')
    print('='*70)
    print(f'Configuration:')
    print(f'  Episodes: {args.episodes}')
    print(f'  Parallel Environments: {args.envs}')
    print(f'  Max Steps: {args.max_steps}')
    print(f'  Grid Size: 10x10x3')
    print(f'  Learning Rate: {args.lr}')
    print(f'  Gamma: {args.gamma}')
    print(f'  Entropy Coef: {args.entropy_coef}')
    print(f'  Save Path: {args.save_path}')
    print('='*70)
    print()

    # Create trainer
    trainer = REINFORCETrainer(
        num_envs=args.envs,
        grid_size=10,
        action_space_type='relative',
        state_representation='grid',  # CNN uses grid state
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        seed=args.seed
    )

    # Train
    start_time = time.time()
    trainer.train(verbose=True, log_interval=args.log_interval)
    end_time = time.time()

    # Save weights
    filename = Path(args.save_path).name
    trainer.save(filename)
    actual_path = trainer.save_dir / filename
    print(f'\nWeights saved to: {actual_path}')

    # Print summary
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = duration % 60

    print()
    print('='*70)
    print('Training Summary')
    print('='*70)
    print(f'Total Time: {minutes}m {seconds:.2f}s')
    print(f'Episodes: {args.episodes}')
    print(f'Episodes/second: {args.episodes / duration:.2f}')
    print(f'Time/episode: {duration / args.episodes:.3f}s')
    stats = trainer.metrics.get_recent_stats()
    print(f'Final Avg Score: {stats["avg_score"]:.2f}')
    print(f'Final Max Score: {stats["max_score"]}')
    print(f'Final Avg Length: {stats["avg_length"]:.2f}')
    print('='*70)


if __name__ == '__main__':
    main()
