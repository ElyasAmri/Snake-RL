"""
Appendix Training Script

Trains all model variants (DQN variants, PPO, A2C, REINFORCE) for the report appendix.
Generates:
- Individual score plot
- Individual death cause plot
- Combined subplot (for appendix)
- JSON with training times and final metrics
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Any

from scripts.training.train_dqn import DQNTrainer
from scripts.training.train_rainbow import RainbowTrainer
from scripts.training.train_ppo import PPOTrainer
from scripts.training.train_a2c import A2CTrainer
from scripts.training.train_reinforce import REINFORCETrainer


# Model configurations
MODEL_CONFIGS = {
    'DQN': {
        'trainer': 'dqn',
        'params': {
            'use_double_dqn': False,
            'use_dueling': False,
            'use_noisy': False,
            'use_prioritized_replay': False,
        }
    },
    'Double DQN': {
        'trainer': 'dqn',
        'params': {
            'use_double_dqn': True,
            'use_dueling': False,
            'use_noisy': False,
            'use_prioritized_replay': False,
        }
    },
    'Dueling DQN': {
        'trainer': 'dqn',
        'params': {
            'use_double_dqn': False,
            'use_dueling': True,
            'use_noisy': False,
            'use_prioritized_replay': False,
        }
    },
    'Noisy DQN': {
        'trainer': 'dqn',
        'params': {
            'use_double_dqn': False,
            'use_dueling': False,
            'use_noisy': True,
            'use_prioritized_replay': False,
        }
    },
    'PER DQN': {
        'trainer': 'dqn',
        'params': {
            'use_double_dqn': False,
            'use_dueling': False,
            'use_noisy': False,
            'use_prioritized_replay': True,
        }
    },
    'Rainbow DQN': {
        'trainer': 'rainbow',
        'params': {
            'v_min': -20.0,
            'v_max': 500.0,
            'n_atoms': 51,
            'n_step': 1,
        }
    },
    'PPO': {
        'trainer': 'ppo',
        'params': {}
    },
    'A2C': {
        'trainer': 'a2c',
        'params': {}
    },
    'REINFORCE': {
        'trainer': 'reinforce',
        'params': {}
    },
}


def smooth(data: List[float], window: int = 100) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(data) < window:
        return np.array(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    # Pad the beginning
    pad = data[:window-1]
    return np.concatenate([pad, smoothed])


def train_model(name: str, config: Dict, num_episodes: int, common_params: Dict) -> Dict[str, Any]:
    """Train a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    trainer_type = config['trainer']
    model_params = config['params'].copy()

    start_time = time.time()

    if trainer_type == 'dqn':
        trainer = DQNTrainer(
            num_envs=common_params['num_envs'],
            grid_size=common_params['grid_size'],
            action_space_type=common_params['action_space_type'],
            use_flood_fill=common_params['use_flood_fill'],
            num_episodes=num_episodes,
            learning_rate=common_params['learning_rate'],
            batch_size=common_params['batch_size'],
            gamma=common_params['gamma'],
            buffer_size=common_params['buffer_size'],
            target_update_freq=common_params['target_update_freq'],
            reward_death=common_params['reward_death'],
            seed=common_params['seed'],
            **model_params
        )
    elif trainer_type == 'rainbow':
        trainer = RainbowTrainer(
            num_envs=common_params['num_envs'],
            grid_size=common_params['grid_size'],
            action_space_type=common_params['action_space_type'],
            use_flood_fill=common_params['use_flood_fill'],
            num_episodes=num_episodes,
            learning_rate=common_params['learning_rate'],
            batch_size=common_params['batch_size'],
            gamma=common_params['gamma'],
            buffer_size=common_params['buffer_size'],
            target_update_freq=common_params['target_update_freq'],
            reward_death=common_params['reward_death'],
            seed=common_params['seed'],
            **model_params
        )
    elif trainer_type == 'ppo':
        trainer = PPOTrainer(
            num_envs=common_params['num_envs'],
            grid_size=common_params['grid_size'],
            action_space_type=common_params['action_space_type'],
            use_flood_fill=common_params['use_flood_fill'],
            num_episodes=num_episodes,
            actor_lr=0.0003,
            critic_lr=0.001,
            gamma=common_params['gamma'],
            reward_death=common_params['reward_death'],
            seed=common_params['seed'],
        )
    elif trainer_type == 'a2c':
        trainer = A2CTrainer(
            num_envs=common_params['num_envs'],
            grid_size=common_params['grid_size'],
            action_space_type=common_params['action_space_type'],
            use_flood_fill=common_params['use_flood_fill'],
            num_episodes=num_episodes,
            actor_lr=0.0003,
            critic_lr=0.001,
            gamma=common_params['gamma'],
            reward_death=common_params['reward_death'],
            seed=common_params['seed'],
        )
    elif trainer_type == 'reinforce':
        trainer = REINFORCETrainer(
            num_envs=common_params['num_envs'],
            grid_size=common_params['grid_size'],
            action_space_type=common_params['action_space_type'],
            use_flood_fill=common_params['use_flood_fill'],
            num_episodes=num_episodes,
            learning_rate=0.001,
            gamma=common_params['gamma'],
            reward_death=common_params['reward_death'],
            seed=common_params['seed'],
        )
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

    # Train
    trainer.train(verbose=True, log_interval=500)

    training_time = time.time() - start_time

    # Collect results
    stats = trainer.metrics.get_recent_stats()
    death_stats = trainer.metrics.get_death_stats()

    results = {
        'name': name,
        'training_time': training_time,
        'final_avg_score': stats['avg_score'],
        'final_max_score': stats['max_score'],
        'final_avg_reward': stats['avg_reward'],
        'scores': trainer.metrics.episode_scores.copy(),
        'death_stats': {
            'wall': death_stats['wall_death_rate'],
            'self': death_stats['self_death_rate'],
            'entrapment': death_stats['entrapment_rate'],
            'timeout': death_stats['timeout_rate'],
        }
    }

    print(f"\n{name} Results:")
    print(f"  Training Time: {training_time:.1f}s")
    print(f"  Final Avg Score: {stats['avg_score']:.2f}")
    print(f"  Final Max Score: {stats['max_score']}")

    return results


def plot_scores(all_results: Dict[str, Dict], output_path: Path, ylim: tuple = None):
    """Plot score curves for all models."""
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (name, results), color in zip(all_results.items(), colors):
        scores = results['scores']
        smoothed = smooth(scores, window=100)
        episodes = np.arange(len(smoothed))

        # Plot smoothed line
        plt.plot(episodes, smoothed, label=name, color=color, linewidth=2)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Training Performance Comparison', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_death_causes(all_results: Dict[str, Dict], output_path: Path):
    """Plot death cause breakdown for all models."""
    fig, ax = plt.subplots(figsize=(12, 8))

    names = list(all_results.keys())
    x = np.arange(len(names))
    width = 0.2

    wall_rates = [r['death_stats']['wall'] * 100 for r in all_results.values()]
    self_rates = [r['death_stats']['self'] * 100 for r in all_results.values()]
    entrapment_rates = [r['death_stats']['entrapment'] * 100 for r in all_results.values()]
    timeout_rates = [r['death_stats']['timeout'] * 100 for r in all_results.values()]

    bars1 = ax.bar(x - 1.5*width, wall_rates, width, label='Wall', color='#e74c3c')
    bars2 = ax.bar(x - 0.5*width, self_rates, width, label='Self', color='#3498db')
    bars3 = ax.bar(x + 0.5*width, entrapment_rates, width, label='Entrapment', color='#f39c12')
    bars4 = ax.bar(x + 1.5*width, timeout_rates, width, label='Timeout', color='#27ae60')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Death Rate (%)', fontsize=12)
    ax.set_title('Death Cause Distribution by Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined(all_results: Dict[str, Dict], output_path: Path, ylim: tuple = None):
    """Create combined subplot with scores and death causes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    # Left plot: Score curves
    for (name, results), color in zip(all_results.items(), colors):
        scores = results['scores']
        smoothed = smooth(scores, window=100)
        episodes = np.arange(len(smoothed))
        ax1.plot(episodes, smoothed, label=name, color=color, linewidth=2)

    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Training Performance Comparison', fontsize=14)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    if ylim:
        ax1.set_ylim(ylim)

    # Right plot: Death causes
    names = list(all_results.keys())
    x = np.arange(len(names))
    width = 0.2

    wall_rates = [r['death_stats']['wall'] * 100 for r in all_results.values()]
    self_rates = [r['death_stats']['self'] * 100 for r in all_results.values()]
    entrapment_rates = [r['death_stats']['entrapment'] * 100 for r in all_results.values()]
    timeout_rates = [r['death_stats']['timeout'] * 100 for r in all_results.values()]

    ax2.bar(x - 1.5*width, wall_rates, width, label='Wall', color='#e74c3c')
    ax2.bar(x - 0.5*width, self_rates, width, label='Self', color='#3498db')
    ax2.bar(x + 0.5*width, entrapment_rates, width, label='Entrapment', color='#f39c12')
    ax2.bar(x + 1.5*width, timeout_rates, width, label='Timeout', color='#27ae60')

    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Death Rate (%)', fontsize=12)
    ax2.set_title('Death Cause Distribution', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main(num_episodes: int = 3000, models: List[str] = None):
    """Main training and plotting function."""

    # Common parameters
    common_params = {
        'num_envs': 256,
        'grid_size': 10,
        'action_space_type': 'relative',
        'use_flood_fill': True,
        'learning_rate': 0.001,
        'batch_size': 64,
        'gamma': 0.99,
        'buffer_size': 100000,
        'target_update_freq': 1000,
        'reward_death': -10.0,
        'seed': 67,
    }

    # Output directories
    figures_dir = Path('results/figures')
    data_dir = Path('results/data')
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to train
    if models is None:
        models_to_train = MODEL_CONFIGS
    else:
        models_to_train = {k: v for k, v in MODEL_CONFIGS.items() if k in models}

    print("="*60)
    print("APPENDIX TRAINING - All Model Variants")
    print("="*60)
    print(f"Episodes per model: {num_episodes}")
    print(f"Models to train: {list(models_to_train.keys())}")
    print()

    # Train all models
    all_results = {}
    total_start = time.time()

    for name, config in models_to_train.items():
        try:
            results = train_model(name, config, num_episodes, common_params)
            all_results[name] = results
        except Exception as e:
            print(f"ERROR training {name}: {e}")
            continue

    total_time = time.time() - total_start

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total training time: {total_time/60:.1f} minutes")

    # Summary table
    print("\n" + "-"*60)
    print(f"{'Model':<15} {'Avg Score':>10} {'Max Score':>10} {'Time (s)':>10}")
    print("-"*60)
    for name, results in all_results.items():
        print(f"{name:<15} {results['final_avg_score']:>10.2f} {results['final_max_score']:>10} {results['training_time']:>10.1f}")
    print("-"*60)

    # Generate plots
    print("\nGenerating plots...")

    # Y-limit for score plots (based on max observed)
    max_score = max(max(r['scores']) for r in all_results.values())
    ylim = (0, min(max_score + 10, 70))

    # Individual plots
    plot_scores(all_results, figures_dir / 'appendix_scores.png', ylim=ylim)
    plot_death_causes(all_results, figures_dir / 'appendix_deaths.png')

    # Combined subplot
    plot_combined(all_results, figures_dir / 'appendix_combined.png', ylim=ylim)

    # Save results JSON
    json_results = {
        'num_episodes': num_episodes,
        'total_training_time': total_time,
        'common_params': common_params,
        'models': {}
    }

    for name, results in all_results.items():
        json_results['models'][name] = {
            'training_time': results['training_time'],
            'final_avg_score': results['final_avg_score'],
            'final_max_score': results['final_max_score'],
            'final_avg_reward': results['final_avg_reward'],
            'death_stats': results['death_stats'],
            'scores': results['scores'],  # Episode-by-episode scores for plotting
        }

    json_path = data_dir / 'appendix_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved: {json_path}")

    print("\nDone!")
    return all_results


def regenerate_plots(json_path: str = 'results/data/appendix_results.json'):
    """Regenerate plots from saved JSON data."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check if scores are present
    sample_model = list(data['models'].keys())[0]
    if 'scores' not in data['models'][sample_model]:
        print("ERROR: JSON does not contain score histories. Cannot regenerate plots.")
        print("Re-run training to generate new data with score histories.")
        return

    # Convert to expected format
    all_results = {}
    for name, model_data in data['models'].items():
        all_results[name] = {
            'scores': model_data['scores'],
            'death_stats': model_data['death_stats'],
            'final_avg_score': model_data['final_avg_score'],
            'final_max_score': model_data['final_max_score'],
        }

    figures_dir = Path('results/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    max_score = max(max(r['scores']) for r in all_results.values())
    ylim = (0, min(max_score + 10, 70))

    plot_scores(all_results, figures_dir / 'appendix_scores.png', ylim=ylim)
    plot_death_causes(all_results, figures_dir / 'appendix_deaths.png')
    plot_combined(all_results, figures_dir / 'appendix_combined.png', ylim=ylim)

    print("Plots regenerated successfully!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train all models for appendix')
    parser.add_argument('--episodes', type=int, default=3000, help='Episodes per model')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to train (default: all)')
    parser.add_argument('--regenerate-plots', action='store_true',
                        help='Regenerate plots from existing JSON data')

    args = parser.parse_args()

    if args.regenerate_plots:
        regenerate_plots()
    else:
        main(num_episodes=args.episodes, models=args.models)
