"""
Grid Size Generalization Evaluation

Trains a DQN model on 10x10 grid with flood-fill features,
then evaluates performance on 8x8, 10x10, 15x15, and 20x20 grids.

This demonstrates that feature-based state representation enables
generalization across different grid sizes without retraining.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from core.environment_vectorized import VectorizedSnakeEnv
from core.networks import DQN_MLP
from core.utils import set_seed, get_device
from scripts.training.train_dqn import DQNTrainer


def evaluate_model(policy_net, grid_size: int, num_episodes: int, device, use_flood_fill: bool = True):
    """Evaluate a trained model on a specific grid size"""

    env = VectorizedSnakeEnv(
        num_envs=256,
        grid_size=grid_size,
        action_space_type='relative',
        state_representation='feature',
        max_steps=1000,
        reward_death=-10.0,
        use_flood_fill=use_flood_fill,
        device=device
    )

    scores = []
    states = env.reset(seed=42)
    completed = 0

    policy_net.eval()

    while completed < num_episodes:
        with torch.no_grad():
            q_values = policy_net(states)
            actions = q_values.argmax(dim=1)

        next_states, rewards, dones, info = env.step(actions)

        if dones.any():
            done_indices = torch.where(dones)[0]
            for idx in done_indices:
                scores.append(info['scores'][idx].item())
                completed += 1
                if completed >= num_episodes:
                    break

        states = next_states

    return scores


def train_and_evaluate():
    """Train on 10x10 and evaluate on multiple grid sizes"""

    device = get_device()
    set_seed(42)

    print("=" * 60)
    print("Grid Size Generalization Experiment")
    print("=" * 60)

    # Training configuration
    train_grid_size = 10
    train_episodes = 5000
    eval_episodes = 500
    eval_grid_sizes = [8, 10, 15, 20]

    # Train on 10x10 with flood-fill features
    print(f"\nTraining DQN on {train_grid_size}x{train_grid_size} grid...")
    print(f"Episodes: {train_episodes}")
    print(f"Features: flood-fill (13 features)")
    print()

    trainer = DQNTrainer(
        num_envs=256,
        num_episodes=train_episodes,
        grid_size=train_grid_size,
        use_flood_fill=True,
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

    trainer.train(verbose=True, log_interval=500)

    # Save trained model
    weights_dir = Path('results/weights')
    weights_dir.mkdir(parents=True, exist_ok=True)
    trainer.save('dqn_flood_fill_10x10.pth')

    # Get reference score from training grid
    train_scores = trainer.metrics.episode_scores[-100:]
    train_avg = np.mean(train_scores)
    train_max = max(trainer.metrics.episode_scores)

    print(f"\nTraining complete on {train_grid_size}x{train_grid_size}")
    print(f"  Avg Score (last 100): {train_avg:.2f}")
    print(f"  Max Score: {train_max}")

    # Evaluate on all grid sizes
    print(f"\n{'=' * 60}")
    print("Evaluating on different grid sizes...")
    print(f"{'=' * 60}")

    results = {}

    for grid_size in eval_grid_sizes:
        print(f"\nEvaluating on {grid_size}x{grid_size} grid ({eval_episodes} episodes)...")

        scores = evaluate_model(
            trainer.policy_net,
            grid_size=grid_size,
            num_episodes=eval_episodes,
            device=device,
            use_flood_fill=True
        )

        avg_score = np.mean(scores)
        max_score = max(scores)
        std_score = np.std(scores)

        # Calculate theoretical max score for this grid size
        # Max food = grid_size^2 - 3 (initial snake length)
        theoretical_max = grid_size * grid_size - 3

        results[grid_size] = {
            'scores': scores,
            'avg_score': avg_score,
            'max_score': max_score,
            'std_score': std_score,
            'theoretical_max': theoretical_max,
            'avg_pct_of_max': (avg_score / theoretical_max) * 100
        }

        print(f"  Avg Score: {avg_score:.2f} +/- {std_score:.2f}")
        print(f"  Max Score: {max_score}")
        print(f"  Theoretical Max: {theoretical_max}")
        print(f"  Avg % of Max: {results[grid_size]['avg_pct_of_max']:.1f}%")

    # Generate comparison plot
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Average scores by grid size
    ax1 = axes[0]
    grid_labels = [f'{gs}x{gs}' for gs in eval_grid_sizes]
    avg_scores = [results[gs]['avg_score'] for gs in eval_grid_sizes]
    std_scores = [results[gs]['std_score'] for gs in eval_grid_sizes]

    bars = ax1.bar(grid_labels, avg_scores, yerr=std_scores, capsize=5,
                   color=['blue' if gs == train_grid_size else 'green' for gs in eval_grid_sizes],
                   alpha=0.7)

    ax1.set_xlabel('Grid Size', fontsize=12)
    ax1.set_ylabel('Average Score', fontsize=12)
    ax1.set_title('Average Score by Grid Size\n(Model trained on 10x10)', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Annotate bars
    for i, (bar, avg, gs) in enumerate(zip(bars, avg_scores, eval_grid_sizes)):
        label = f'{avg:.1f}'
        if gs == train_grid_size:
            label += '\n(trained)'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_scores[i] + 1,
                label, ha='center', va='bottom', fontsize=10)

    # Plot 2: Score as percentage of theoretical maximum
    ax2 = axes[1]
    pct_scores = [results[gs]['avg_pct_of_max'] for gs in eval_grid_sizes]

    bars2 = ax2.bar(grid_labels, pct_scores,
                    color=['blue' if gs == train_grid_size else 'green' for gs in eval_grid_sizes],
                    alpha=0.7)

    ax2.set_xlabel('Grid Size', fontsize=12)
    ax2.set_ylabel('% of Theoretical Maximum', fontsize=12)
    ax2.set_title('Performance Relative to Grid Capacity\n(Model trained on 10x10)', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 100)

    # Annotate bars
    for bar, pct, gs in zip(bars2, pct_scores, eval_grid_sizes):
        label = f'{pct:.1f}%'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                label, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'grid_size_generalization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {output_dir / 'grid_size_generalization.png'}")

    # Save results to JSON
    data_dir = Path('results/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    json_results = {
        'train_grid_size': train_grid_size,
        'train_episodes': train_episodes,
        'eval_episodes': eval_episodes,
        'results': {
            str(gs): {
                'avg_score': results[gs]['avg_score'],
                'max_score': results[gs]['max_score'],
                'std_score': results[gs]['std_score'],
                'theoretical_max': results[gs]['theoretical_max'],
                'avg_pct_of_max': results[gs]['avg_pct_of_max']
            }
            for gs in eval_grid_sizes
        }
    }

    with open(data_dir / 'grid_size_generalization.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Saved: {data_dir / 'grid_size_generalization.json'}")

    # Print summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY: Grid Size Generalization Results")
    print(f"{'=' * 60}")
    print(f"{'Grid':<10} {'Avg Score':<12} {'Max Score':<12} {'Theory Max':<12} {'% of Max':<10}")
    print("-" * 56)
    for gs in eval_grid_sizes:
        r = results[gs]
        trained = " (trained)" if gs == train_grid_size else ""
        print(f"{gs}x{gs}{trained:<10} {r['avg_score']:<12.2f} {r['max_score']:<12} {r['theoretical_max']:<12} {r['avg_pct_of_max']:<10.1f}%")

    return results


if __name__ == '__main__':
    train_and_evaluate()
