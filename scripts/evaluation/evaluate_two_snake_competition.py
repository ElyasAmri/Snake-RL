"""
Final Head-to-Head Evaluation for Two-Snake Competition

Evaluates trained models in a head-to-head competition.
Supports evaluating models from different training methods:
- DQN Co-evolution
- PPO Direct Co-evolution
- PPO Curriculum + Co-evolution

Usage:
    ./venv/Scripts/python.exe scripts/evaluation/evaluate_two_snake_competition.py \
        --model-128 "results/weights/ppo_coevolution/*128x128*.pt" \
        --model-256 "results/weights/ppo_coevolution/*256x256*.pt" \
        --num-games 1000 \
        --output results/data/competition_results.json
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
from glob import glob
from typing import Optional, Dict, List

from scipy import stats

from core.environment_two_snake_vectorized import VectorizedTwoSnakeEnv
from core.networks import PPO_Actor_MLP, DQN_MLP
from core.utils import set_seed, get_device


def find_latest_checkpoint(pattern: str) -> Optional[str]:
    """Find the latest checkpoint matching a pattern"""
    matches = glob(pattern)
    if not matches:
        return None
    return max(matches, key=lambda x: Path(x).stat().st_mtime)


def detect_model_type(checkpoint_path: str) -> str:
    """Detect if model is PPO or DQN based on checkpoint keys"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'actor' in checkpoint:
        return 'ppo'
    elif 'policy_net' in checkpoint:
        return 'dqn'
    else:
        raise ValueError(f"Unknown model type in checkpoint: {checkpoint.keys()}")


def load_model(checkpoint_path: str, hidden_dims: tuple, device: torch.device):
    """Load model from checkpoint"""
    model_type = detect_model_type(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_type == 'ppo':
        model = PPO_Actor_MLP(33, 3, hidden_dims).to(device)
        model.load_state_dict(checkpoint['actor'])
    else:
        model = DQN_MLP(33, 3, hidden_dims).to(device)
        model.load_state_dict(checkpoint['policy_net'])

    model.eval()
    return model, model_type


def compute_confidence_interval(wins: int, total: int, confidence: float = 0.95) -> tuple:
    """Compute Wilson score confidence interval for binomial proportion"""
    if total == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = wins / total
    n = total

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

    return (max(0, center - margin), min(1, center + margin))


def evaluate_competition(
    model_128_path: str,
    model_256_path: str,
    num_games: int = 1000,
    num_envs: int = 128,
    grid_size: int = 20,
    target_food: int = 10,
    seed: int = 67
) -> Dict:
    """
    Run head-to-head competition between two models.

    Args:
        model_128_path: Path to 128x128 model checkpoint
        model_256_path: Path to 256x256 model checkpoint
        num_games: Number of games to play
        num_envs: Number of parallel environments
        grid_size: Grid size for environment
        target_food: Target food to win
        seed: Random seed

    Returns:
        Dictionary with competition results
    """
    set_seed(seed)
    device = get_device()

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPETITION EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Number of games: {num_games}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Target food: {target_food}")
    print(f"Model 128x128: {model_128_path}")
    print(f"Model 256x256: {model_256_path}")
    print("=" * 70 + "\n")

    # Load models
    model_128, type_128 = load_model(model_128_path, (128, 128), device)
    model_256, type_256 = load_model(model_256_path, (256, 256), device)

    print(f"Model 128x128 type: {type_128}")
    print(f"Model 256x256 type: {type_256}")

    # Create environment
    env = VectorizedTwoSnakeEnv(
        num_envs=min(num_envs, num_games),
        grid_size=grid_size,
        max_steps=1000,
        target_food=target_food,
        device=device
    )

    # Run evaluation
    winners = []
    scores_128 = []
    scores_256 = []
    games_played = 0

    obs_256, obs_128 = env.reset()

    while games_played < num_games:
        with torch.no_grad():
            # 256x256 is snake 1, 128x128 is snake 2
            if type_256 == 'ppo':
                logits_256 = model_256(obs_256)
                actions_256 = F.softmax(logits_256, dim=-1).argmax(dim=1)
            else:
                q_256 = model_256(obs_256)
                actions_256 = q_256.argmax(dim=1)

            if type_128 == 'ppo':
                logits_128 = model_128(obs_128)
                actions_128 = F.softmax(logits_128, dim=-1).argmax(dim=1)
            else:
                q_128 = model_128(obs_128)
                actions_128 = q_128.argmax(dim=1)

        next_obs_256, next_obs_128, r_256, r_128, dones, info = env.step(actions_256, actions_128)

        if dones.any():
            num_done = len(info['done_envs'])
            for i in range(num_done):
                if games_played < num_games:
                    winners.append(int(info['winners'][i]))
                    scores_256.append(info['food_counts1'][i])
                    scores_128.append(info['food_counts2'][i])
                    games_played += 1

        obs_256 = next_obs_256
        obs_128 = next_obs_128

        # Progress update
        if games_played % 100 == 0:
            print(f"Games played: {games_played}/{num_games}", flush=True)

    # Calculate statistics
    wins_256 = sum(1 for w in winners if w == 1)
    wins_128 = sum(1 for w in winners if w == 2)
    draws = sum(1 for w in winners if w == 3)

    win_rate_256 = wins_256 / num_games
    win_rate_128 = wins_128 / num_games
    draw_rate = draws / num_games

    # Confidence intervals
    ci_256 = compute_confidence_interval(wins_256, num_games)
    ci_128 = compute_confidence_interval(wins_128, num_games)
    ci_draw = compute_confidence_interval(draws, num_games)

    results = {
        'num_games': num_games,
        'grid_size': grid_size,
        'target_food': target_food,
        'model_128_path': model_128_path,
        'model_256_path': model_256_path,
        'model_128_type': type_128,
        'model_256_type': type_256,

        'wins_256': wins_256,
        'wins_128': wins_128,
        'draws': draws,

        'win_rate_256': win_rate_256,
        'win_rate_128': win_rate_128,
        'draw_rate': draw_rate,

        'confidence_interval_95': {
            'win_rate_256': list(ci_256),
            'win_rate_128': list(ci_128),
            'draw_rate': list(ci_draw)
        },

        'avg_score_256': float(np.mean(scores_256)),
        'avg_score_128': float(np.mean(scores_128)),
        'std_score_256': float(np.std(scores_256)),
        'std_score_128': float(np.std(scores_128)),
    }

    # Print results
    print("\n" + "=" * 70)
    print("COMPETITION RESULTS")
    print("=" * 70)
    print(f"Games played: {num_games}")
    print()
    print(f"256x256 Wins: {wins_256} ({win_rate_256:.1%}) [95% CI: {ci_256[0]:.1%} - {ci_256[1]:.1%}]")
    print(f"128x128 Wins: {wins_128} ({win_rate_128:.1%}) [95% CI: {ci_128[0]:.1%} - {ci_128[1]:.1%}]")
    print(f"Draws:        {draws} ({draw_rate:.1%}) [95% CI: {ci_draw[0]:.1%} - {ci_draw[1]:.1%}]")
    print()
    print(f"Avg Score 256x256: {np.mean(scores_256):.2f} +/- {np.std(scores_256):.2f}")
    print(f"Avg Score 128x128: {np.mean(scores_128):.2f} +/- {np.std(scores_128):.2f}")
    print("=" * 70 + "\n")

    return results


def evaluate_all_methods(
    methods: Dict[str, Dict[str, str]],
    num_games: int = 1000,
    output_path: Optional[str] = None
) -> Dict:
    """
    Evaluate all training methods and compile results.

    Args:
        methods: Dictionary mapping method name to model paths
            e.g., {'PPO Curriculum': {'128': 'path', '256': 'path'}}
        num_games: Number of games per evaluation
        output_path: Path to save combined results

    Returns:
        Dictionary with all results
    """
    all_results = {}

    for method_name, paths in methods.items():
        print(f"\n{'#' * 70}")
        print(f"# Evaluating: {method_name}")
        print(f"{'#' * 70}")

        if paths.get('128') and paths.get('256'):
            try:
                results = evaluate_competition(
                    model_128_path=paths['128'],
                    model_256_path=paths['256'],
                    num_games=num_games
                )
                all_results[method_name] = results
            except Exception as e:
                print(f"Error evaluating {method_name}: {e}")
                all_results[method_name] = {'error': str(e)}
        else:
            print(f"Missing model paths for {method_name}")

    # Save combined results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nSaved all results to: {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Two-Snake Competition Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model-128', type=str, default=None,
                        help='Path to 128x128 model (supports wildcards)')
    parser.add_argument('--model-256', type=str, default=None,
                        help='Path to 256x256 model (supports wildcards)')
    parser.add_argument('--num-games', type=int, default=1000,
                        help='Number of games to play')
    parser.add_argument('--output', type=str, default='results/data/competition_results.json',
                        help='Output JSON file')
    parser.add_argument('--eval-all', action='store_true',
                        help='Evaluate all available methods')
    parser.add_argument('--seed', type=int, default=67,
                        help='Random seed')

    args = parser.parse_args()

    if args.eval_all:
        # Find all available methods
        methods = {}

        # DQN Co-evolution
        dqn_128 = find_latest_checkpoint('results/weights/dqn_two_snake_mlp/*128x128*.pt')
        dqn_256 = find_latest_checkpoint('results/weights/dqn_two_snake_mlp/*256x256*.pt')
        if dqn_128 and dqn_256:
            methods['DQN Co-evolution'] = {'128': dqn_128, '256': dqn_256}

        # PPO Direct Co-evolution
        ppo_direct_128 = find_latest_checkpoint('results/weights/ppo_direct_coevolution*/*128x128*.pt')
        ppo_direct_256 = find_latest_checkpoint('results/weights/ppo_direct_coevolution*/*256x256*.pt')
        if ppo_direct_128 and ppo_direct_256:
            methods['PPO Direct'] = {'128': ppo_direct_128, '256': ppo_direct_256}

        # PPO Curriculum
        ppo_curriculum_128 = find_latest_checkpoint('results/weights/ppo_coevolution/*128x128*.pt')
        ppo_curriculum_256 = find_latest_checkpoint('results/weights/ppo_coevolution/*256x256*.pt')
        if ppo_curriculum_128 and ppo_curriculum_256:
            methods['PPO Curriculum'] = {'128': ppo_curriculum_128, '256': ppo_curriculum_256}

        if methods:
            evaluate_all_methods(methods, args.num_games, args.output)
        else:
            print("No trained models found. Please train models first.")

    else:
        # Single evaluation
        model_128 = args.model_128
        model_256 = args.model_256

        # Handle wildcards
        if model_128 and '*' in model_128:
            model_128 = find_latest_checkpoint(model_128)
            if model_128:
                print(f"Found 128x128 model: {model_128}")
            else:
                print("No 128x128 model found matching pattern")
                return

        if model_256 and '*' in model_256:
            model_256 = find_latest_checkpoint(model_256)
            if model_256:
                print(f"Found 256x256 model: {model_256}")
            else:
                print("No 256x256 model found matching pattern")
                return

        if not model_128 or not model_256:
            print("Please provide both --model-128 and --model-256 paths")
            return

        results = evaluate_competition(
            model_128_path=model_128,
            model_256_path=model_256,
            num_games=args.num_games,
            seed=args.seed
        )

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
