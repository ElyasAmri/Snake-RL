"""
Generate score and death cause plots for the report.

Trains models and creates:
- Score plots for all configurations
- Death cause plots for basic feature configurations (for death investigation section)

Configurations:
- DQN basic
- PPO basic
- DQN flood-fill
- PPO flood-fill
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from core.utils import set_seed, get_device


def smooth(data, window=50):
    """Smooth data using moving average"""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_score_only(name: str, scores: list, output_path: Path):
    """Generate and save score-only plot"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Raw scores
    ax.plot(scores, alpha=0.3, color='green', label='Raw')

    # Smoothed scores
    if len(scores) >= 50:
        smoothed = smooth(scores, 50)
        ax.plot(range(49, 49 + len(smoothed)), smoothed, color='green', linewidth=2, label='Smoothed (50)')

    # Max score line
    max_score = max(scores) if scores else 0
    ax.axhline(y=max_score, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(len(scores) * 0.02, max_score + 1, f'Max: {max_score}',
            color='darkgreen', fontsize=10, fontweight='bold')

    # Average score line
    avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    ax.axhline(y=avg_score, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(len(scores) * 0.02, avg_score + 1, f'Avg (last 100): {avg_score:.1f}',
            color='blue', fontsize=10, fontweight='bold')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score (Food Eaten)', fontsize=12)
    ax.set_title(f'{name} - Training Scores', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set reasonable y-limit
    ax.set_ylim(0, max(max_score + 10, 50))
    ax.set_xlim(0, len(scores))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_cumulative_deaths(name: str, metrics, output_path: Path):
    """Generate cumulative death cause plot with separate lines"""
    fig, ax = plt.subplots(figsize=(8, 5))

    wall_cumsum = np.cumsum(metrics.wall_deaths_per_episode)
    self_cumsum = np.cumsum(metrics.self_deaths_per_episode)
    entrap_cumsum = np.cumsum(metrics.entrapments_per_episode)

    episodes = range(1, len(wall_cumsum) + 1)

    ax.plot(episodes, wall_cumsum, linewidth=2, label='Wall', color='red')
    ax.plot(episodes, self_cumsum, linewidth=2, label='Self', color='blue')
    ax.plot(episodes, entrap_cumsum, linewidth=2, label='Entrapment', color='orange')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Deaths', fontsize=12)
    ax.set_title(f'{name} - Cumulative Death Causes', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(episodes))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")

    total = len(metrics.wall_deaths_per_episode)
    return {
        'wall': sum(metrics.wall_deaths_per_episode) / total * 100,
        'self': sum(metrics.self_deaths_per_episode) / total * 100,
        'entrapment': sum(metrics.entrapments_per_episode) / total * 100,
        'timeout': sum(metrics.timeouts_per_episode) / total * 100
    }


def train_and_plot(algorithm: str, use_flood_fill: bool, num_episodes: int, output_dir: Path):
    """Train a model and generate score-only plot"""

    device = get_device()
    set_seed(42)

    feature_name = "flood-fill" if use_flood_fill else "basic"
    name = f"{algorithm} ({feature_name.title()} Features)"
    output_file = f"{algorithm.lower()}_{feature_name}_scores.png"

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    if algorithm.upper() == 'DQN':
        from scripts.training.train_dqn import DQNTrainer
        trainer = DQNTrainer(
            num_envs=256,
            num_episodes=num_episodes,
            grid_size=10,
            use_flood_fill=use_flood_fill,
            action_space_type='relative',
            use_double_dqn=False,
            use_dueling=False,
            use_noisy=False,
            use_prioritized_replay=False,
            learning_rate=0.001,
            batch_size=64,
            gamma=0.99,
            buffer_size=100000,
            target_update_freq=1000,
            reward_death=-10.0,
            seed=42,
            device=device,
        )
    elif algorithm.upper() == 'PPO':
        from scripts.training.train_ppo import PPOTrainer
        trainer = PPOTrainer(
            num_envs=256,
            num_episodes=num_episodes,
            grid_size=10,
            use_flood_fill=use_flood_fill,
            action_space_type='relative',
            actor_lr=0.0003,
            critic_lr=0.001,
            gamma=0.99,
            reward_death=-10.0,
            seed=42,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    trainer.train(verbose=True, log_interval=500)

    # Get scores and plot
    scores = trainer.metrics.episode_scores
    plot_score_only(name, scores, output_dir / output_file)

    # Generate death cause plot for all configurations
    death_output = f"{algorithm.lower()}_{feature_name}_deaths.png"
    death_stats = plot_cumulative_deaths(name, trainer.metrics, output_dir / death_output)
    print(f"  Death breakdown: Wall {death_stats['wall']:.1f}%, Self {death_stats['self']:.1f}%, Entrap {death_stats['entrapment']:.1f}%")

    # Save data to JSON
    data_dir = Path('results/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    json_path = data_dir / f"{algorithm.lower()}_{feature_name}_scores.json"
    data = {
        'algorithm': algorithm,
        'features': feature_name,
        'num_episodes': num_episodes,
        'scores': scores,
        'max_score': max(scores),
        'avg_score_last_100': np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores),
        'death_stats': death_stats,
    }
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved data: {json_path}")

    return scores


def main():
    """Generate score-only plots for report"""

    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training configurations - 3000 episodes for all for fair comparison
    configs = [
        ('DQN', False, 3000),   # DQN basic
        ('PPO', False, 3000),   # PPO basic
        ('DQN', True, 3000),    # DQN flood-fill
        ('PPO', True, 3000),    # PPO flood-fill
    ]

    for algorithm, use_flood_fill, num_episodes in configs:
        train_and_plot(algorithm, use_flood_fill, num_episodes, output_dir)

    print("\n" + "="*60)
    print("All plots generated!")
    print("="*60)


if __name__ == '__main__':
    main()
