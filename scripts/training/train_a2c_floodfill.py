"""
Train A2C with Flood-Fill Features

Standalone script for training A2C with 14-dimensional feature state representation including flood-fill.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import time
from scripts.training.train_a2c import A2CTrainer


def main():
    parser = argparse.ArgumentParser(description='Train A2C with Flood-Fill on Snake game')

    # Training config
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--envs', type=int, default=256, help='Number of parallel environments')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')

    # Network config
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[128, 128], help='Hidden layer dimensions')

    # A2C config
    parser.add_argument('--actor-lr', type=float, default=0.0003, help='Actor learning rate')
    parser.add_argument('--critic-lr', type=float, default=0.001, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--value-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--rollout-steps', type=int, default=5, help='Rollout steps')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=50, help='Logging interval')
    parser.add_argument('--save-path', type=str, default='results/weights/a2c_floodfill.pt', help='Path to save weights')

    args = parser.parse_args()

    print('='*70)
    print('Training A2C with Flood-Fill (14-feature State)')
    print('='*70)
    print(f'Configuration:')
    print(f'  Episodes: {args.episodes}')
    print(f'  Parallel Environments: {args.envs}')
    print(f'  Max Steps: {args.max_steps}')
    print(f'  Hidden Dims: {args.hidden_dims}')
    print(f'  Actor LR: {args.actor_lr}')
    print(f'  Critic LR: {args.critic_lr}')
    print(f'  Gamma: {args.gamma}')
    print(f'  Entropy Coef: {args.entropy_coef}')
    print(f'  Value Coef: {args.value_coef}')
    print(f'  Rollout Steps: {args.rollout_steps}')
    print(f'  Save Path: {args.save_path}')
    print(f'  FLOOD-FILL FEATURES: ENABLED (14 features)')
    print('='*70)
    print()

    # Create trainer with flood-fill enabled
    trainer = A2CTrainer(
        num_envs=args.envs,
        grid_size=10,
        action_space_type='relative',
        state_representation='feature',
        use_flood_fill=True,  # Enable flood-fill features
        hidden_dims=tuple(args.hidden_dims),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        rollout_steps=args.rollout_steps,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
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
