"""
Generate appendix plots for the report.

Trains all 9 algorithms with specified feature set,
generating multi-line comparison plots for:
- Training score curves (all algorithms overlaid)
- Death cause distribution (grouped bar chart)

Usage:
    python scripts/generate_appendix_plots.py --features basic
    python scripts/generate_appendix_plots.py --features flood-fill

Algorithms:
- DQN, Double DQN, Dueling DQN, Noisy DQN, PER DQN, Rainbow DQN
- PPO, A2C, REINFORCE
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

from core.utils import set_seed, get_device


def smooth(data, window=50):
    """Smooth data using moving average"""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


# Color palette for 9 algorithms
COLORS = {
    'DQN': '#1f77b4',           # blue
    'Double DQN': '#ff7f0e',    # orange
    'Dueling DQN': '#2ca02c',   # green
    'Noisy DQN': '#d62728',     # red
    'PER DQN': '#9467bd',       # purple
    'Rainbow DQN': '#8c564b',   # brown
    'PPO': '#e377c2',           # pink
    'A2C': '#7f7f7f',           # gray
    'REINFORCE': '#bcbd22',     # olive
}


def train_algorithm(algo_name: str, use_flood_fill: bool, num_episodes: int, device):
    """Train a single algorithm and return metrics"""

    # Common kwargs for all algorithms
    base_kwargs = {
        'num_envs': 256,
        'num_episodes': num_episodes,
        'grid_size': 10,
        'use_flood_fill': use_flood_fill,
        'action_space_type': 'relative',
        'seed': 42,
    }

    # DQN variants support reward_death, policy gradient algorithms don't
    common_kwargs = base_kwargs.copy()
    if algo_name in ['DQN', 'Double DQN', 'Dueling DQN', 'Noisy DQN', 'PER DQN', 'Rainbow DQN']:
        common_kwargs['reward_death'] = -10.0

    if algo_name == 'DQN':
        from scripts.training.train_dqn import DQNTrainer
        trainer = DQNTrainer(
            use_double_dqn=False,
            use_dueling=False,
            use_noisy=False,
            use_prioritized_replay=False,
            device=device,
            **common_kwargs
        )
    elif algo_name == 'Double DQN':
        from scripts.training.train_dqn import DQNTrainer
        trainer = DQNTrainer(
            use_double_dqn=True,
            use_dueling=False,
            use_noisy=False,
            use_prioritized_replay=False,
            device=device,
            **common_kwargs
        )
    elif algo_name == 'Dueling DQN':
        from scripts.training.train_dqn import DQNTrainer
        trainer = DQNTrainer(
            use_double_dqn=False,
            use_dueling=True,
            use_noisy=False,
            use_prioritized_replay=False,
            device=device,
            **common_kwargs
        )
    elif algo_name == 'Noisy DQN':
        from scripts.training.train_dqn import DQNTrainer
        trainer = DQNTrainer(
            use_double_dqn=False,
            use_dueling=False,
            use_noisy=True,
            use_prioritized_replay=False,
            device=device,
            **common_kwargs
        )
    elif algo_name == 'PER DQN':
        from scripts.training.train_dqn import DQNTrainer
        trainer = DQNTrainer(
            use_double_dqn=False,
            use_dueling=False,
            use_noisy=False,
            use_prioritized_replay=True,
            device=device,
            **common_kwargs
        )
    elif algo_name == 'Rainbow DQN':
        from scripts.training.train_rainbow import RainbowTrainer
        trainer = RainbowTrainer(
            device=device,
            **common_kwargs
        )
    elif algo_name == 'PPO':
        from scripts.training.train_ppo import PPOTrainer
        trainer = PPOTrainer(
            actor_lr=0.0003,
            critic_lr=0.001,
            gamma=0.99,
            **common_kwargs
        )
    elif algo_name == 'A2C':
        from scripts.training.train_a2c import A2CTrainer
        trainer = A2CTrainer(
            actor_lr=0.0003,
            critic_lr=0.001,
            gamma=0.99,
            **common_kwargs
        )
    elif algo_name == 'REINFORCE':
        from scripts.training.train_reinforce import REINFORCETrainer
        trainer = REINFORCETrainer(
            learning_rate=0.001,
            gamma=0.99,
            **common_kwargs
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    trainer.train(verbose=True, log_interval=500)

    return {
        'scores': trainer.metrics.episode_scores,
        'wall_deaths': trainer.metrics.wall_deaths_per_episode,
        'self_deaths': trainer.metrics.self_deaths_per_episode,
        'entrapments': trainer.metrics.entrapments_per_episode,
        'timeouts': trainer.metrics.timeouts_per_episode,
    }


def plot_multi_algorithm_scores(results: dict, feature_name: str, output_path: Path):
    """Generate multi-line overlay plot for all algorithms"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for algo_name, data in results.items():
        scores = data['scores']
        color = COLORS[algo_name]

        # Only plot smoothed for clarity
        if len(scores) >= 50:
            smoothed = smooth(scores, 50)
            ax.plot(range(49, 49 + len(smoothed)), smoothed,
                   color=color, linewidth=2, label=algo_name)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score (Food Eaten)', fontsize=12)
    ax.set_title(f'Training Score Comparison - {feature_name.title()} Features', fontsize=14)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 60)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_death_distribution(results: dict, feature_name: str, output_path: Path):
    """Generate grouped bar chart of death causes by algorithm"""
    algorithms = list(results.keys())
    n_algos = len(algorithms)

    # Calculate death percentages
    wall_pcts = []
    self_pcts = []
    entrap_pcts = []

    for algo in algorithms:
        data = results[algo]
        total = len(data['wall_deaths'])
        wall_pcts.append(sum(data['wall_deaths']) / total * 100)
        self_pcts.append(sum(data['self_deaths']) / total * 100)
        entrap_pcts.append(sum(data['entrapments']) / total * 100)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_algos)
    width = 0.25

    bars1 = ax.bar(x - width, wall_pcts, width, label='Wall', color='red', alpha=0.8)
    bars2 = ax.bar(x, self_pcts, width, label='Self', color='blue', alpha=0.8)
    bars3 = ax.bar(x + width, entrap_pcts, width, label='Entrapment', color='orange', alpha=0.8)

    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Death Percentage (%)', fontsize=12)
    ax.set_title(f'Death Cause Distribution - {feature_name.title()} Features', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 5:  # Only label if bar is tall enough
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    """Generate appendix plots for specified feature set"""

    parser = argparse.ArgumentParser(description="Generate appendix plots")
    parser.add_argument("--features", type=str, required=True,
                        choices=["basic", "flood-fill"],
                        help="Feature set to use (basic or flood-fill)")
    parser.add_argument("--episodes", type=int, default=3000,
                        help="Number of episodes per algorithm (default: 3000)")
    args = parser.parse_args()

    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path('results/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    num_episodes = args.episodes
    use_flood_fill = args.features == "flood-fill"
    feature_name = args.features

    algorithms = [
        'DQN', 'Double DQN', 'Dueling DQN', 'Noisy DQN', 'PER DQN',
        'Rainbow DQN', 'PPO', 'A2C', 'REINFORCE'
    ]

    print("\n" + "="*70)
    print(f"TRAINING WITH {feature_name.upper()} FEATURES")
    print("="*70)

    results = {}
    for algo in algorithms:
        print(f"\n--- Training {algo} ({feature_name}) ---")
        set_seed(42)
        results[algo] = train_algorithm(algo, use_flood_fill=use_flood_fill,
                                         num_episodes=num_episodes, device=device)

    # Generate plots
    suffix = "floodfill" if use_flood_fill else "basic"
    plot_multi_algorithm_scores(results, feature_name, output_dir / f'appendix_{suffix}_scores.png')
    plot_death_distribution(results, feature_name, output_dir / f'appendix_{suffix}_deaths.png')

    # Save summary data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {}

    for algo in algorithms:
        data = results[algo]
        summary[algo] = {
            'avg_score_last_100': float(np.mean(data['scores'][-100:])),
            'max_score': int(max(data['scores'])),
            'wall_pct': float(sum(data['wall_deaths']) / len(data['wall_deaths']) * 100),
            'self_pct': float(sum(data['self_deaths']) / len(data['self_deaths']) * 100),
            'entrap_pct': float(sum(data['entrapments']) / len(data['entrapments']) * 100),
        }

    with open(data_dir / f'appendix_{suffix}_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print(f"APPENDIX PLOTS FOR {feature_name.upper()} FEATURES GENERATED!")
    print("="*70)

    # Print summary table
    print(f"\n--- {feature_name.upper()} FEATURES ---")
    print(f"{'Algorithm':<15} {'Avg Score':>12} {'Max Score':>10} {'Wall%':>8} {'Self%':>8} {'Entrap%':>8}")
    print("-"*65)
    for algo in algorithms:
        d = summary[algo]
        print(f"{algo:<15} {d['avg_score_last_100']:>12.2f} {d['max_score']:>10} {d['wall_pct']:>8.1f} {d['self_pct']:>8.1f} {d['entrap_pct']:>8.1f}")


if __name__ == '__main__':
    main()
