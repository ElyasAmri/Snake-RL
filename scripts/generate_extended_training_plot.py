"""
Generate extended training comparison for A2C vs REINFORCE.

Demonstrates sample inefficiency of policy gradient methods by
training for 10,000 episodes instead of the standard 3,000.

This generates a figure for the Limitations (sample inefficiency) section.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt

from core.utils import set_seed, get_device


def smooth(data, window=100):
    """Smooth data using moving average"""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def train_a2c(num_episodes: int, device):
    """Train A2C and return scores"""
    from scripts.training.train_a2c import A2CTrainer

    set_seed(42)
    trainer = A2CTrainer(
        num_envs=256,
        num_episodes=num_episodes,
        grid_size=10,
        use_flood_fill=False,
        action_space_type='relative',
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.99,
        seed=42,
    )
    trainer.train(verbose=True, log_interval=500)
    return trainer.metrics.episode_scores


def train_reinforce(num_episodes: int, device):
    """Train REINFORCE and return scores"""
    from scripts.training.train_reinforce import REINFORCETrainer

    set_seed(42)
    trainer = REINFORCETrainer(
        num_envs=256,
        num_episodes=num_episodes,
        grid_size=10,
        use_flood_fill=False,
        action_space_type='relative',
        learning_rate=0.001,
        gamma=0.99,
        seed=42,
    )
    trainer.train(verbose=True, log_interval=500)
    return trainer.metrics.episode_scores


def plot_extended_training(a2c_scores: list, reinforce_scores: list, output_path: Path):
    """Generate comparison plot matching project style"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors matching project palette
    a2c_color = '#7f7f7f'       # gray (from COLORS dict)
    reinforce_color = '#bcbd22' # olive (from COLORS dict)

    # Plot raw scores with low alpha
    ax.plot(a2c_scores, alpha=0.15, color=a2c_color)
    ax.plot(reinforce_scores, alpha=0.15, color=reinforce_color)

    # Plot smoothed curves
    window = 100
    if len(a2c_scores) >= window:
        a2c_smoothed = smooth(a2c_scores, window)
        ax.plot(range(window-1, window-1 + len(a2c_smoothed)), a2c_smoothed,
                color=a2c_color, linewidth=2.5, label='A2C')

    if len(reinforce_scores) >= window:
        reinforce_smoothed = smooth(reinforce_scores, window)
        ax.plot(range(window-1, window-1 + len(reinforce_smoothed)), reinforce_smoothed,
                color=reinforce_color, linewidth=2.5, label='REINFORCE')

    # Reference line for PPO performance at 3000 episodes (approx 20 avg score)
    ax.axhline(y=20, color='#e377c2', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(len(a2c_scores) * 0.75, 21, 'PPO (3,000 ep)', color='#e377c2', fontsize=10)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score (Food Eaten)', fontsize=12)
    ax.set_title('Extended Training: A2C vs REINFORCE (10,000 Episodes)', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set axis limits
    max_score = max(max(a2c_scores), max(reinforce_scores))
    ax.set_ylim(0, max(max_score + 5, 30))
    ax.set_xlim(0, len(a2c_scores))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    """Train A2C and REINFORCE for extended episodes and generate plot"""

    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path('results/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    num_episodes = 10000

    print("\n" + "="*60)
    print("EXTENDED TRAINING: A2C vs REINFORCE")
    print(f"Episodes: {num_episodes}")
    print("="*60)

    # Train A2C
    print("\n--- Training A2C ---")
    a2c_scores = train_a2c(num_episodes, device)

    # Train REINFORCE
    print("\n--- Training REINFORCE ---")
    reinforce_scores = train_reinforce(num_episodes, device)

    # Generate plot
    plot_extended_training(
        a2c_scores,
        reinforce_scores,
        output_dir / 'extended_training_a2c_reinforce.png'
    )

    # Save data
    data = {
        'num_episodes': num_episodes,
        'a2c': {
            'scores': a2c_scores,
            'avg_last_100': float(np.mean(a2c_scores[-100:])),
            'max_score': int(max(a2c_scores)),
        },
        'reinforce': {
            'scores': reinforce_scores,
            'avg_last_100': float(np.mean(reinforce_scores[-100:])),
            'max_score': int(max(reinforce_scores)),
        }
    }

    with open(data_dir / 'extended_training_a2c_reinforce.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("\n" + "="*60)
    print("EXTENDED TRAINING COMPLETE!")
    print("="*60)
    print(f"\nA2C:       Avg (last 100) = {data['a2c']['avg_last_100']:.2f}, Max = {data['a2c']['max_score']}")
    print(f"REINFORCE: Avg (last 100) = {data['reinforce']['avg_last_100']:.2f}, Max = {data['reinforce']['max_score']}")


if __name__ == '__main__':
    main()
