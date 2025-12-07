"""
Train All Single-Snake Models

Trains all 8 single-snake RL algorithms with all 4 feature variants
(32 total configurations) for a specified number of episodes and generates
reward/score plots for each.

Feature Variants:
- basic: 11 features (danger, food direction, current direction)
- flood-fill: 14 features (basic + flood-fill free space)
- selective: 21 features (basic + enhanced features: escape routes, tail info)
- enhanced: 24 features (basic + flood-fill + enhanced)

Usage:
    python scripts/training/train_all_models.py [--episodes 5000] [--num-envs 256]
    python scripts/training/train_all_models.py --feature-variant basic  # Single variant
    python scripts/training/train_all_models.py --algorithm DQN  # Single algorithm
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import time
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch

from core.utils import set_seed, get_device


def smooth(data, window=50):
    """Smooth data using moving average"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_training_results(name: str, rewards: list, scores: list, metrics, save_dir: Path, timestamp: str):
    """Generate and save 2-panel plot for training results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Episode Scores (Food Eaten)
    ax1 = axes[0]
    ax1.plot(scores, alpha=0.3, color='green', label='Raw')
    if len(scores) >= 50:
        smoothed = smooth(scores, 50)
        ax1.plot(range(49, 49 + len(smoothed)), smoothed, color='green', label='Smoothed (50)')
    # Max score line
    max_score = max(scores) if scores else 0
    ax1.axhline(y=max_score, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.text(len(scores) * 0.02, max_score + 0.5, f'Max: {max_score}',
             color='darkgreen', fontsize=9, fontweight='bold')
    # Average score line
    avg_score = np.mean(scores) if scores else 0
    ax1.axhline(y=avg_score, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.text(len(scores) * 0.02, avg_score + 0.5, f'Avg: {avg_score:.1f}',
             color='blue', fontsize=9, fontweight='bold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score (Food Eaten)')
    ax1.set_xlim(0, 3000)
    ax1.set_ylim(0, 100)
    ax1.set_title(f'{name} - Episode Scores')
    ax1.legend(['Raw', 'Smoothed (50)'], loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Cumulative Deaths
    ax2 = axes[1]
    episodes = range(len(metrics.wall_deaths_per_episode))
    ax2.plot(episodes, np.cumsum(metrics.wall_deaths_per_episode),
             label='Wall', color='red')
    ax2.plot(episodes, np.cumsum(metrics.self_deaths_per_episode),
             label='Self', color='orange')
    ax2.plot(episodes, np.cumsum(metrics.entrapments_per_episode),
             label='Entrapment', color='purple')
    ax2.plot(episodes, np.cumsum(metrics.timeouts_per_episode),
             label='Timeout', color='gray')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Count')
    ax2.set_title(f'{name} - Death Causes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    filename = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{timestamp}.png"
    filepath = save_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved: {filepath}")
    return filepath


def train_dqn(num_episodes: int, num_envs: int, device, **kwargs):
    """Train vanilla DQN"""
    from scripts.training.train_dqn import DQNTrainer

    trainer = DQNTrainer(
        num_envs=num_envs,
        num_episodes=num_episodes,
        use_double_dqn=False,
        use_dueling=False,
        use_noisy=False,
        use_prioritized_replay=False,
        device=device,
        **kwargs
    )
    trainer.train(verbose=True, log_interval=100)
    return trainer.metrics.episode_rewards, trainer.metrics.episode_scores, trainer


def train_double_dqn(num_episodes: int, num_envs: int, device, **kwargs):
    """Train Double DQN"""
    from scripts.training.train_dqn import DQNTrainer

    trainer = DQNTrainer(
        num_envs=num_envs,
        num_episodes=num_episodes,
        use_double_dqn=True,
        use_dueling=False,
        use_noisy=False,
        use_prioritized_replay=False,
        device=device,
        **kwargs
    )
    trainer.train(verbose=True, log_interval=100)
    return trainer.metrics.episode_rewards, trainer.metrics.episode_scores, trainer


def train_dueling_dqn(num_episodes: int, num_envs: int, device, **kwargs):
    """Train Dueling DQN"""
    from scripts.training.train_dqn import DQNTrainer

    trainer = DQNTrainer(
        num_envs=num_envs,
        num_episodes=num_episodes,
        use_double_dqn=False,
        use_dueling=True,
        use_noisy=False,
        use_prioritized_replay=False,
        device=device,
        **kwargs
    )
    trainer.train(verbose=True, log_interval=100)
    return trainer.metrics.episode_rewards, trainer.metrics.episode_scores, trainer


def train_noisy_dqn(num_episodes: int, num_envs: int, device, **kwargs):
    """Train Noisy DQN"""
    from scripts.training.train_dqn import DQNTrainer

    trainer = DQNTrainer(
        num_envs=num_envs,
        num_episodes=num_episodes,
        use_double_dqn=False,
        use_dueling=False,
        use_noisy=True,
        use_prioritized_replay=False,
        device=device,
        **kwargs
    )
    trainer.train(verbose=True, log_interval=100)
    return trainer.metrics.episode_rewards, trainer.metrics.episode_scores, trainer


def train_per_dqn(num_episodes: int, num_envs: int, device, **kwargs):
    """Train Prioritized Experience Replay DQN"""
    from scripts.training.train_dqn import DQNTrainer

    trainer = DQNTrainer(
        num_envs=num_envs,
        num_episodes=num_episodes,
        use_double_dqn=False,
        use_dueling=False,
        use_noisy=False,
        use_prioritized_replay=True,
        device=device,
        **kwargs
    )
    trainer.train(verbose=True, log_interval=100)
    return trainer.metrics.episode_rewards, trainer.metrics.episode_scores, trainer


def train_ppo(num_episodes: int, num_envs: int, device, **kwargs):
    """Train PPO"""
    from scripts.training.train_ppo import PPOTrainer

    # PPO only supports use_flood_fill, not use_enhanced_features
    ppo_kwargs = {k: v for k, v in kwargs.items() if k != 'use_enhanced_features'}
    trainer = PPOTrainer(
        num_envs=num_envs,
        num_episodes=num_episodes,
        device=device,
        **ppo_kwargs
    )
    trainer.train(verbose=True, log_interval=100)
    return trainer.metrics.episode_rewards, trainer.metrics.episode_scores, trainer


def train_a2c(num_episodes: int, num_envs: int, device, **kwargs):
    """Train A2C"""
    from scripts.training.train_a2c import A2CTrainer

    # A2C only supports use_flood_fill, not use_enhanced_features
    a2c_kwargs = {k: v for k, v in kwargs.items() if k != 'use_enhanced_features'}
    trainer = A2CTrainer(
        num_envs=num_envs,
        num_episodes=num_episodes,
        device=device,
        **a2c_kwargs
    )
    trainer.train(verbose=True, log_interval=100)
    return trainer.metrics.episode_rewards, trainer.metrics.episode_scores, trainer


def train_reinforce(num_episodes: int, num_envs: int, device, **kwargs):
    """Train REINFORCE"""
    from scripts.training.train_reinforce import REINFORCETrainer

    # REINFORCE only supports use_flood_fill, not use_enhanced_features
    reinforce_kwargs = {k: v for k, v in kwargs.items() if k != 'use_enhanced_features'}
    trainer = REINFORCETrainer(
        num_envs=num_envs,
        num_episodes=num_episodes,
        device=device,
        **reinforce_kwargs
    )
    trainer.train(verbose=True, log_interval=100)
    return trainer.metrics.episode_rewards, trainer.metrics.episode_scores, trainer


# All algorithms to train
ALGORITHMS = [
    ("DQN", train_dqn),
    ("Double DQN", train_double_dqn),
    ("Dueling DQN", train_dueling_dqn),
    ("Noisy DQN", train_noisy_dqn),
    ("PER DQN", train_per_dqn),
    ("PPO", train_ppo),
    ("A2C", train_a2c),
    ("REINFORCE", train_reinforce),
]

# Feature variants: (name, use_flood_fill, use_enhanced_features)
# Note: PPO, A2C, REINFORCE only support basic and flood-fill (not enhanced features)
FEATURE_VARIANTS = [
    ("basic", False, False),           # 11 features
    ("flood-fill", True, False),       # 14 features
    ("selective", False, True),        # 21 features (DQN variants only)
    ("enhanced", True, True),          # 24 features (DQN variants only)
]

# Algorithms that support enhanced features (use_enhanced_features parameter)
DQN_ALGORITHMS = ["DQN", "Double DQN", "Dueling DQN", "Noisy DQN", "PER DQN"]
# Algorithms that only support flood-fill (not enhanced)
POLICY_GRADIENT_ALGORITHMS = ["PPO", "A2C", "REINFORCE"]


def main():
    parser = argparse.ArgumentParser(description="Train all single-snake RL models")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes per model")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=67, help="Random seed")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size")
    parser.add_argument("--feature-variant", type=str, default=None,
                        choices=["basic", "flood-fill", "selective", "enhanced"],
                        help="Train only specific feature variant (default: all)")
    parser.add_argument("--algorithm", type=str, default=None,
                        help="Train only specific algorithm (e.g., 'DQN', 'PPO')")
    parser.add_argument("--reward-death", type=float, default=-10.0,
                        help="Death penalty reward (default: -10.0)")
    args = parser.parse_args()

    # Filter feature variants if specified
    if args.feature_variant:
        feature_variants = [fv for fv in FEATURE_VARIANTS if fv[0] == args.feature_variant]
    else:
        feature_variants = FEATURE_VARIANTS

    # Filter algorithms if specified
    if args.algorithm:
        algorithms = [(n, f) for n, f in ALGORITHMS if n.lower() == args.algorithm.lower()]
        if not algorithms:
            print(f"Unknown algorithm: {args.algorithm}")
            print(f"Available: {[n for n, _ in ALGORITHMS]}")
            return
    else:
        algorithms = ALGORITHMS

    # Setup
    device = get_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output directories
    fig_dir = Path("results/figures")
    weights_dir = Path("results/weights")
    fig_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total configurations
    # DQN variants: 5 algos x 4 variants = 20
    # Policy gradient: 3 algos x 2 variants = 6
    # Total = 26 (not 32)
    dqn_algos = [(n, f) for n, f in algorithms if n in DQN_ALGORITHMS]
    pg_algos = [(n, f) for n, f in algorithms if n in POLICY_GRADIENT_ALGORITHMS]
    total_configs = len(dqn_algos) * len(feature_variants) + len(pg_algos) * 2  # PG only has basic + flood-fill

    print("=" * 70)
    print("TRAINING ALL SINGLE-SNAKE MODELS WITH FEATURE VARIANTS")
    print("=" * 70)
    print(f"Episodes per model: {args.episodes}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Device: {device}")
    print(f"Grid size: {args.grid_size}")
    print(f"Seed: {args.seed}")
    print(f"DQN algorithms: {len(dqn_algos)} x 4 variants = {len(dqn_algos) * 4}")
    print(f"Policy gradient algorithms: {len(pg_algos)} x 2 variants = {len(pg_algos) * 2}")
    print(f"Total configurations: {total_configs}")
    print("=" * 70)
    print()

    results = {}
    total_start = time.time()
    config_num = 0

    for variant_name, use_flood_fill, use_enhanced in feature_variants:
        print(f"\n{'='*70}")
        print(f"FEATURE VARIANT: {variant_name.upper()}")
        print(f"  flood-fill: {use_flood_fill}, enhanced: {use_enhanced}")
        print(f"{'='*70}")

        for algo_name, train_fn in algorithms:
            # Skip enhanced variants for policy gradient algorithms
            if algo_name in POLICY_GRADIENT_ALGORITHMS and use_enhanced:
                print(f"\n  Skipping {algo_name} ({variant_name}) - enhanced features not supported")
                continue

            config_num += 1
            full_name = f"{algo_name} ({variant_name})"
            print(f"\n[{config_num}/{total_configs}] Training {full_name}...")
            print("-" * 50)

            # Set seed for reproducibility
            set_seed(args.seed)

            # Train
            start_time = time.time()
            try:
                rewards, scores, trainer = train_fn(
                    num_episodes=args.episodes,
                    num_envs=args.num_envs,
                    device=device,
                    grid_size=args.grid_size,
                    seed=args.seed,
                    use_flood_fill=use_flood_fill,
                    use_enhanced_features=use_enhanced,
                    reward_death=args.reward_death
                )
                elapsed = time.time() - start_time

                # Get death stats
                death_stats = trainer.metrics.get_death_stats()

                # Save results
                results[full_name] = {
                    "algorithm": algo_name,
                    "variant": variant_name,
                    "rewards": rewards,
                    "scores": scores,
                    "episode_lengths": trainer.metrics.episode_lengths,
                    "time": elapsed,
                    "avg_reward": np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
                    "avg_score": np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores),
                    "avg_length": np.mean(trainer.metrics.episode_lengths[-100:]) if len(trainer.metrics.episode_lengths) >= 100 else np.mean(trainer.metrics.episode_lengths),
                    "max_score": max(scores) if scores else 0,
                    "death_stats": death_stats
                }

                # Generate plot with variant in name
                plot_name = f"{algo_name}_{variant_name}"
                plot_training_results(plot_name, rewards, scores, trainer.metrics, fig_dir, timestamp)

                # Save weights with variant in name
                weight_filename = f"{algo_name.lower().replace(' ', '_')}_{variant_name}_{args.episodes}ep_{timestamp}.pt"
                trainer.save(weight_filename)

                print(f"\n  {full_name} completed in {elapsed:.1f}s")
                print(f"  Avg Reward (last 100): {results[full_name]['avg_reward']:.2f}")
                print(f"  Avg Score (last 100): {results[full_name]['avg_score']:.2f}")
                print(f"  Max Score: {results[full_name]['max_score']}")
                print(f"  Death Stats: Wall={death_stats['wall_deaths']}, Self={death_stats['self_deaths']}, Entrapment={death_stats['entrapments']}, Timeout={death_stats['timeouts']}")

            except Exception as e:
                print(f"  ERROR training {full_name}: {e}")
                import traceback
                traceback.print_exc()
                results[full_name] = {"error": str(e)}

    # Summary
    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print()

    # Group results by variant
    for variant_name, _, _ in feature_variants:
        print(f"\n--- {variant_name.upper()} ---")
        print(f"{'Algorithm':<15} {'Avg Reward':>12} {'Avg Score':>12} {'Max Score':>10} {'Time (s)':>10}")
        print("-" * 65)

        for algo_name, _ in algorithms:
            full_name = f"{algo_name} ({variant_name})"
            if full_name in results and "error" not in results[full_name]:
                r = results[full_name]
                print(f"{algo_name:<15} {r['avg_reward']:>12.2f} {r['avg_score']:>12.2f} {r['max_score']:>10} {r['time']:>10.1f}")
            elif full_name in results:
                print(f"{algo_name:<15} {'ERROR':>12}")

    # Best performers per variant
    print("\n" + "=" * 80)
    print("BEST PERFORMERS BY VARIANT")
    print("=" * 80)
    for variant_name, _, _ in feature_variants:
        variant_results = {k: v for k, v in results.items()
                         if variant_name in k and "error" not in v}
        if variant_results:
            best = max(variant_results.items(), key=lambda x: x[1]['avg_score'])
            print(f"  {variant_name:<12}: {best[0]:<30} (Avg Score: {best[1]['avg_score']:.2f})")

    print("=" * 80)
    print(f"\nPlots saved to: {fig_dir}")
    print(f"Weights saved to: {weights_dir}")

    # Save stats to JSON (without large lists)
    data_dir = Path("results/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    json_results = {}
    for name, data in results.items():
        if "error" in data:
            json_results[name] = {"error": data["error"]}
        else:
            json_results[name] = {
                "algorithm": data["algorithm"],
                "variant": data["variant"],
                "training_time_seconds": data["time"],
                "episodes": len(data["rewards"]),
                "avg_reward_last_100": data["avg_reward"],
                "avg_score_last_100": data["avg_score"],
                "avg_length_last_100": data["avg_length"],
                "max_score": data["max_score"],
                "death_stats": data["death_stats"]
            }
    json_path = data_dir / f"training_stats_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Statistics saved to: {json_path}")


if __name__ == "__main__":
    main()
