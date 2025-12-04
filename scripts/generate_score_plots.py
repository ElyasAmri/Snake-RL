"""
Generate score-only plots for the report.

Trains models and creates single-panel score plots (without death causes) for:
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

    # Save scores to JSON for future use
    data_dir = Path('results/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    json_path = data_dir / f"{algorithm.lower()}_{feature_name}_scores.json"
    with open(json_path, 'w') as f:
        json.dump({
            'algorithm': algorithm,
            'features': feature_name,
            'num_episodes': num_episodes,
            'scores': scores,
            'max_score': max(scores),
            'avg_score_last_100': np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores),
        }, f, indent=2)
    print(f"Saved data: {json_path}")

    return scores


def main():
    """Generate score-only plots for report"""

    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training configurations
    # Basic features: 3000 episodes
    # Flood-fill: 2000 episodes (converges faster)
    configs = [
        ('DQN', False, 3000),   # DQN basic
        ('PPO', False, 2000),   # PPO basic
        ('DQN', True, 5000),    # DQN flood-fill (needs more for good results)
        ('PPO', True, 2000),    # PPO flood-fill
    ]

    for algorithm, use_flood_fill, num_episodes in configs:
        train_and_plot(algorithm, use_flood_fill, num_episodes, output_dir)

    print("\n" + "="*60)
    print("All plots generated!")
    print("="*60)


if __name__ == '__main__':
    main()
