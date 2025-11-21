"""
Train DQN with MLP (Basic Features)

Standalone script for training DQN with 11-dimensional basic feature state representation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import time
from datetime import datetime
from scripts.training.train_dqn import DQNTrainer


def main():
    parser = argparse.ArgumentParser(description='Train DQN-MLP on Snake game')

    # Training config
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--envs', type=int, default=256, help='Number of parallel environments')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')

    # Network config
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128], help='Hidden layer dimensions')

    # DQN config
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--target-update-freq', type=int, default=1000, help='Target network update frequency')
    parser.add_argument('--min-buffer-size', type=int, default=1000, help='Minimum buffer size before training')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=50, help='Logging interval')
    parser.add_argument('--save-path', type=str, default='results/weights/dqn_mlp_128x128.pt', help='Path to save weights')

    args = parser.parse_args()

    print('='*70)
    print('Training DQN-MLP with Basic Features (11-feature State)')
    print('='*70)
    print(f'Configuration:')
    print(f'  Episodes: {args.episodes}')
    print(f'  Parallel Environments: {args.envs}')
    print(f'  Max Steps: {args.max_steps}')
    print(f'  Hidden Dims: {args.hidden_dims}')
    print(f'  Learning Rate: {args.lr}')
    print(f'  Batch Size: {args.batch_size}')
    print(f'  Buffer Size: {args.buffer_size}')
    print(f'  Gamma: {args.gamma}')
    print(f'  Epsilon: {args.epsilon_start} -> {args.epsilon_end} (decay={args.epsilon_decay})')
    print(f'  Target Update Freq: {args.target_update_freq}')
    print(f'  Save Path: {args.save_path}')
    print(f'  FEATURES: Basic 11 features (no flood-fill, selective, or enhanced)')
    print('='*70)
    print()

    # Create trainer with basic features only
    trainer = DQNTrainer(
        num_envs=args.envs,
        grid_size=10,
        action_space_type='relative',
        state_representation='feature',
        use_flood_fill=False,  # Disable flood-fill
        use_selective_features=False,  # Disable selective features
        use_enhanced_features=False,  # Disable enhanced features
        hidden_dims=tuple(args.hidden_dims),
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
    start_time = time.time()
    trainer.train(verbose=True, log_interval=args.log_interval)
    end_time = time.time()

    # Save weights with episode count and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = Path(args.save_path).stem
    ext = Path(args.save_path).suffix
    filename = f"{base_filename}_{args.episodes}ep_{timestamp}{ext}"
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
