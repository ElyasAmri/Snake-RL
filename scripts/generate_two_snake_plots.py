"""
Generate two-snake competitive training plots for the report.

Trains and evaluates:
1. DQN direct co-evolution (both agents learning from start)
2. PPO direct co-evolution (both agents learning from start)
3. PPO with curriculum learning (5-stage progressive difficulty)

Generates figures:
- two_snake_training_curves.png - Win rate comparison over training
- two_snake_curriculum_stages.png - PPO curriculum stage progression
- two_snake_final_competition.png - Big vs Small head-to-head results

Also outputs:
- two_snake_results.json - All metrics for report tables
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from datetime import datetime

from core.utils import set_seed, get_device


def smooth(data, window=10):
    """Smooth data using moving average"""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def train_dqn_direct(total_steps: int = 1000000, max_time: int = None):
    """Train DQN direct co-evolution"""
    from scripts.training.train_dqn_two_snake_mlp import DQNConfig, TwoSnakeDQNTrainer

    print("\n" + "="*70)
    print("TRAINING: DQN Direct Co-evolution")
    print("="*70 + "\n")

    config = DQNConfig(
        total_steps=total_steps,
        save_dir='results/weights/dqn_two_snake_mlp',
        max_time=max_time,
        seed=67
    )

    trainer = TwoSnakeDQNTrainer(config)
    results = trainer.train()
    return results


def train_ppo_direct(total_steps: int = 1000000, max_time: int = None):
    """Train PPO direct co-evolution"""
    from scripts.training.train_ppo_two_snake_mlp import PPOConfig, TwoSnakePPOTrainer

    print("\n" + "="*70)
    print("TRAINING: PPO Direct Co-evolution")
    print("="*70 + "\n")

    config = PPOConfig(
        total_steps=total_steps,
        save_dir='results/weights/ppo_two_snake_mlp',
        max_time=max_time,
        seed=67
    )

    trainer = TwoSnakePPOTrainer(config)
    trainer.train()

    # Extract results
    return {
        'algorithm': 'PPO',
        'curriculum': False,
        'total_steps': trainer.total_steps,
        'total_rounds': trainer.total_rounds,
        'final_win_rate': trainer.calculate_win_rate(),
        'avg_score1': float(np.mean(trainer.scores1[-100:])) if trainer.scores1 else 0,
        'avg_score2': float(np.mean(trainer.scores2[-100:])) if trainer.scores2 else 0,
        'scores1': trainer.scores1,
        'scores2': trainer.scores2,
        'win_rate_history': getattr(trainer, 'win_rate_history', [])
    }


def train_ppo_curriculum(max_time: int = None):
    """Train PPO with curriculum learning"""
    from scripts.training.train_ppo_two_snake_mlp import PPOConfig
    from scripts.training.train_ppo_two_snake_mlp_curriculum import CurriculumPPOTrainer

    print("\n" + "="*70)
    print("TRAINING: PPO Curriculum Learning")
    print("="*70 + "\n")

    config = PPOConfig(
        save_dir='results/weights/ppo_two_snake_mlp_curriculum',
        max_time=max_time,
        seed=67
    )

    trainer = CurriculumPPOTrainer(config)
    trainer.train()

    # Extract curriculum stage results
    stage_results = []
    for stage in trainer.stages:
        stage_results.append({
            'stage_id': stage.stage_id,
            'name': stage.name,
            'opponent_type': stage.opponent_type,
            'target_food': stage.target_food,
            'min_steps': stage.min_steps,
            'win_rate_threshold': stage.win_rate_threshold,
        })

    return {
        'algorithm': 'PPO',
        'curriculum': True,
        'total_steps': trainer.total_steps,
        'total_rounds': trainer.total_rounds,
        'final_win_rate': trainer.calculate_win_rate(),
        'avg_score1': float(np.mean(trainer.scores1[-100:])) if trainer.scores1 else 0,
        'avg_score2': float(np.mean(trainer.scores2[-100:])) if trainer.scores2 else 0,
        'scores1': trainer.scores1,
        'scores2': trainer.scores2,
        'stage_results': stage_results,
        'win_rate_history': getattr(trainer, 'win_rate_history', [])
    }


def evaluate_competition(model_path1: str, model_path2: str, num_games: int = 100):
    """Evaluate head-to-head competition between two models"""
    from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv
    from core.networks import DQN_MLP, PPO_Actor_MLP

    device = get_device()

    # Determine if DQN or PPO based on path
    is_ppo = 'ppo' in model_path1.lower()

    # Load models
    if is_ppo:
        # PPO models
        model1 = PPO_Actor_MLP(33, 3, (256, 256)).to(device)
        model2 = PPO_Actor_MLP(33, 3, (128, 128)).to(device)

        checkpoint1 = torch.load(model_path1, map_location=device)
        checkpoint2 = torch.load(model_path2, map_location=device)

        model1.load_state_dict(checkpoint1['actor'])
        model2.load_state_dict(checkpoint2['actor'])
    else:
        # DQN models
        model1 = DQN_MLP(33, 3, (256, 256)).to(device)
        model2 = DQN_MLP(33, 3, (128, 128)).to(device)

        checkpoint1 = torch.load(model_path1, map_location=device)
        checkpoint2 = torch.load(model_path2, map_location=device)

        model1.load_state_dict(checkpoint1['policy_net'])
        model2.load_state_dict(checkpoint2['policy_net'])

    model1.eval()
    model2.eval()

    # Create environment
    env = VectorizedTwoSnakeEnv(
        num_envs=min(num_games, 128),
        grid_size=20,
        target_food=10,
        device=device
    )

    # Run evaluation
    winners = []
    scores1 = []
    scores2 = []
    games_played = 0

    obs1, obs2 = env.reset()

    while games_played < num_games:
        with torch.no_grad():
            if is_ppo:
                logits1 = model1(obs1)
                logits2 = model2(obs2)
                actions1 = logits1.argmax(dim=1)
                actions2 = logits2.argmax(dim=1)
            else:
                q1 = model1(obs1)
                q2 = model2(obs2)
                actions1 = q1.argmax(dim=1)
                actions2 = q2.argmax(dim=1)

        next_obs1, next_obs2, r1, r2, dones, info = env.step(actions1, actions2)

        if dones.any():
            num_done = len(info['done_envs'])
            for i in range(num_done):
                if games_played < num_games:
                    winners.append(int(info['winners'][i]))
                    scores1.append(info['food_counts1'][i])
                    scores2.append(info['food_counts2'][i])
                    games_played += 1

        obs1 = next_obs1
        obs2 = next_obs2

    big_wins = sum(1 for w in winners if w == 1)
    small_wins = sum(1 for w in winners if w == 2)
    draws = sum(1 for w in winners if w == 3)  # 3 = both_lose/stalemate, 0 = in_progress

    return {
        'big_wins': big_wins,
        'small_wins': small_wins,
        'draws': draws,
        'big_win_rate': big_wins / num_games,
        'small_win_rate': small_wins / num_games,
        'avg_score_big': np.mean(scores1),
        'avg_score_small': np.mean(scores2)
    }


def plot_training_curves(results_list, output_path: Path):
    """Plot win rate comparison for all methods"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'DQN': 'blue', 'PPO': 'green', 'PPO Curriculum': 'red'}

    for results in results_list:
        name = results['algorithm']
        if results.get('curriculum'):
            name += ' Curriculum'

        if results.get('win_rate_history'):
            steps = [h['step'] for h in results['win_rate_history']]
            rates = [h['win_rate'] for h in results['win_rate_history']]

            if len(rates) > 10:
                smoothed = smooth(rates, min(10, len(rates)//5))
                offset = len(rates) - len(smoothed)
                ax.plot(steps[offset:], smoothed, linewidth=2,
                        label=name, color=colors.get(name, 'gray'))
            else:
                ax.plot(steps, rates, linewidth=2,
                        label=name, color=colors.get(name, 'gray'))

    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='50% (Random)')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Big Agent (256x256) Win Rate', fontsize=12)
    ax.set_title('Two-Snake Competitive Training - Win Rate Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_curriculum_stages(results, output_path: Path):
    """Plot curriculum learning stage progression"""
    if not results.get('stage_results'):
        print("No curriculum stage data available")
        return

    stages = results['stage_results']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Stage parameters
    stage_names = [s['name'].replace('Stage', 'S').replace('_', '\n') for s in stages]
    target_foods = [s['target_food'] for s in stages]
    thresholds = [s['win_rate_threshold'] if s['win_rate_threshold'] else 0 for s in stages]

    x = np.arange(len(stages))
    width = 0.35

    # Plot 1: Target food and win rate thresholds
    bars1 = ax1.bar(x - width/2, target_foods, width, label='Target Food', color='steelblue')
    ax1.set_ylabel('Target Food Count', fontsize=11)
    ax1.set_xlabel('Curriculum Stage', fontsize=11)
    ax1.set_title('Progressive Difficulty Settings', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_names, fontsize=9)

    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, [t*100 for t in thresholds], width,
                         label='Win Rate Threshold', color='coral')
    ax1_twin.set_ylabel('Win Rate Threshold (%)', fontsize=11)

    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')

    # Plot 2: Opponent progression
    opponent_types = [s['opponent_type'] for s in stages]
    opponent_difficulty = {'static': 1, 'random': 2, 'greedy': 3, 'frozen': 4, 'learning': 5}
    difficulties = [opponent_difficulty.get(o, 0) for o in opponent_types]

    ax2.bar(x, difficulties, color='purple', alpha=0.7)
    ax2.set_ylabel('Opponent Difficulty', fontsize=11)
    ax2.set_xlabel('Curriculum Stage', fontsize=11)
    ax2.set_title('Opponent Progression', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stage_names, fontsize=9)
    ax2.set_yticks([1, 2, 3, 4, 5])
    ax2.set_yticklabels(['Static', 'Random', 'Greedy', 'Frozen', 'Co-evolve'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_competition(competition_results, output_path: Path):
    """Plot final competition results comparing all methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = list(competition_results.keys())
    big_wins = [competition_results[m]['big_win_rate'] * 100 for m in methods]
    small_wins = [competition_results[m]['small_win_rate'] * 100 for m in methods]
    draws = [100 - big_wins[i] - small_wins[i] for i in range(len(methods))]

    x = np.arange(len(methods))
    width = 0.25

    # Win rate comparison
    ax1.bar(x - width, big_wins, width, label='Big (256x256) Wins', color='steelblue')
    ax1.bar(x, small_wins, width, label='Small (128x128) Wins', color='coral')
    ax1.bar(x + width, draws, width, label='Draws', color='gray')

    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.set_xlabel('Training Method', fontsize=12)
    ax1.set_title('Final Competition: Big vs Small Network', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.legend(loc='upper right')
    ax1.axhline(y=50, color='black', linestyle='--', alpha=0.3)

    # Average scores
    avg_big = [competition_results[m]['avg_score_big'] for m in methods]
    avg_small = [competition_results[m]['avg_score_small'] for m in methods]

    ax2.bar(x - width/2, avg_big, width, label='Big (256x256)', color='steelblue')
    ax2.bar(x + width/2, avg_small, width, label='Small (128x128)', color='coral')

    ax2.set_ylabel('Average Score', fontsize=12)
    ax2.set_xlabel('Training Method', fontsize=12)
    ax2.set_title('Average Scores in Competition', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Run all two-snake training and generate plots"""
    set_seed(67)

    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path('results/data')
    data_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    start_time = time.time()

    # Train all configurations
    print("\n" + "="*70)
    print("TWO-SNAKE COMPETITIVE TRAINING - GENERATING ALL RESULTS")
    print("="*70)

    # 1. DQN Direct
    try:
        dqn_results = train_dqn_direct(total_steps=1000000)
        all_results['DQN'] = dqn_results
    except Exception as e:
        print(f"DQN training failed: {e}")

    # 2. PPO Direct
    try:
        ppo_results = train_ppo_direct(total_steps=1000000)
        all_results['PPO'] = ppo_results
    except Exception as e:
        print(f"PPO training failed: {e}")

    # 3. PPO Curriculum
    try:
        ppo_curriculum_results = train_ppo_curriculum()
        all_results['PPO Curriculum'] = ppo_curriculum_results
    except Exception as e:
        print(f"PPO Curriculum training failed: {e}")

    # Generate training curves plot
    print("\nGenerating training curves plot...")
    if all_results:
        plot_training_curves(list(all_results.values()), output_dir / 'two_snake_training_curves.png')

    # Generate curriculum stages plot
    if 'PPO Curriculum' in all_results:
        print("Generating curriculum stages plot...")
        plot_curriculum_stages(all_results['PPO Curriculum'],
                               output_dir / 'two_snake_curriculum_stages.png')

    # Run final competition evaluation
    print("\nRunning final competition evaluation...")
    competition_results = {}

    weights_dir = Path('results/weights')

    # Find latest checkpoints for each method
    for method, subdir in [('DQN', 'dqn_two_snake_mlp'),
                           ('PPO', 'ppo_two_snake_mlp'),
                           ('PPO Curriculum', 'ppo_two_snake_mlp_curriculum')]:
        method_dir = weights_dir / subdir
        if method_dir.exists():
            # Find big and small model files
            big_files = sorted(method_dir.glob('big_256x256_*.pt'))
            small_files = sorted(method_dir.glob('small_128x128_*.pt'))

            if big_files and small_files:
                try:
                    result = evaluate_competition(
                        str(big_files[-1]),
                        str(small_files[-1]),
                        num_games=100
                    )
                    competition_results[method] = result
                    print(f"  {method}: Big wins {result['big_win_rate']:.1%}, "
                          f"Small wins {result['small_win_rate']:.1%}")
                except Exception as e:
                    print(f"  {method} evaluation failed: {e}")

    # Generate competition plot
    if competition_results:
        plot_final_competition(competition_results, output_dir / 'two_snake_final_competition.png')

    # Save all results to JSON
    results_path = data_dir / 'two_snake_results.json'

    # Prepare serializable results
    serializable_results = {}
    for name, res in all_results.items():
        serializable_results[name] = {
            'algorithm': res.get('algorithm'),
            'curriculum': res.get('curriculum'),
            'total_steps': res.get('total_steps'),
            'total_rounds': res.get('total_rounds'),
            'final_win_rate': res.get('final_win_rate'),
            'avg_score1': res.get('avg_score1'),
            'avg_score2': res.get('avg_score2'),
        }
        if res.get('stage_results'):
            serializable_results[name]['stage_results'] = res['stage_results']

    serializable_results['competition'] = competition_results

    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {total_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
