"""
Comprehensive Evaluation of All Two-Snake Training Methods

Runs head-to-head competition (1000 games) for all trained methods:
1. DQN Direct Co-evolution (2M steps)
2. PPO Direct Co-evolution (2M steps)
3. PPO Direct Co-evolution (14M steps)
4. PPO Curriculum + Co-evolution (14M total)

Saves comprehensive results to JSON for plotting and report generation.

Usage:
    # Evaluate all methods
    ./venv/Scripts/python.exe scripts/evaluation/evaluate_all_two_snake.py

    # Quick test with fewer games
    ./venv/Scripts/python.exe scripts/evaluation/evaluate_all_two_snake.py --num-games 100
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
import argparse
from glob import glob
from datetime import datetime
from typing import Optional, Dict

from scripts.evaluation.evaluate_two_snake_competition import (
    evaluate_competition,
    find_latest_checkpoint
)


# Define all training methods and their checkpoint locations
METHODS = {
    'DQN Direct 2M': {
        'weights_dir': 'results/weights/dqn_direct_coevolution',
        'alt_dirs': ['results/weights/dqn_two_snake_mlp'],
        'pattern_128': '*128x128*final*.pt',
        'pattern_256': '*256x256*final*.pt',
        'alt_pattern_128': '*small_128x128*final*.pt',
        'alt_pattern_256': '*big_256x256*final*.pt',
    },
    'PPO Direct 2M': {
        'weights_dir': 'results/weights/ppo_direct_coevolution_2M',
        'pattern_128': '*128x128*coevo*.pt',
        'pattern_256': '*256x256*coevo*.pt',
    },
    'PPO Direct 14M': {
        'weights_dir': 'results/weights/ppo_direct_coevolution',
        'alt_dirs': ['results/weights/ppo_direct_coevolution_14M'],
        'pattern_128': '*128x128*coevo*.pt',
        'pattern_256': '*256x256*coevo*.pt',
    },
    'PPO Curriculum': {
        'weights_dir': 'results/weights/ppo_coevolution',
        'pattern_128': '*128x128*coevo*.pt',
        'pattern_256': '*256x256*coevo*.pt',
    }
}


def find_model_checkpoint(weights_dir: str, pattern: str, alt_dirs: list = None, alt_pattern: str = None) -> Optional[str]:
    """Find the latest checkpoint matching the pattern"""
    # Try primary location
    full_pattern = str(Path(weights_dir) / pattern)
    match = find_latest_checkpoint(full_pattern)
    if match:
        return match

    # Try alternative pattern in primary location
    if alt_pattern:
        full_pattern = str(Path(weights_dir) / alt_pattern)
        match = find_latest_checkpoint(full_pattern)
        if match:
            return match

    # Try alternative directories
    if alt_dirs:
        for alt_dir in alt_dirs:
            full_pattern = str(Path(alt_dir) / pattern)
            match = find_latest_checkpoint(full_pattern)
            if match:
                return match

            if alt_pattern:
                full_pattern = str(Path(alt_dir) / alt_pattern)
                match = find_latest_checkpoint(full_pattern)
                if match:
                    return match

    return None


def evaluate_all_methods(num_games: int = 1000, output_path: str = 'results/data/two_snake_competition_results.json'):
    """Evaluate all trained methods and save comprehensive results"""

    print("\n" + "=" * 70)
    print("COMPREHENSIVE TWO-SNAKE EVALUATION")
    print("=" * 70)
    print(f"Games per method: {num_games}")
    print(f"Output: {output_path}")
    print("=" * 70 + "\n")

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_config': {
            'num_games': num_games,
            'grid_size': 20,
            'target_food': 10
        },
        'methods': {}
    }

    for method_name, method_config in METHODS.items():
        print(f"\n{'#' * 70}")
        print(f"# Method: {method_name}")
        print(f"{'#' * 70}")

        # Find checkpoints
        model_128 = find_model_checkpoint(
            method_config['weights_dir'],
            method_config['pattern_128'],
            method_config.get('alt_dirs'),
            method_config.get('alt_pattern_128')
        )
        model_256 = find_model_checkpoint(
            method_config['weights_dir'],
            method_config['pattern_256'],
            method_config.get('alt_dirs'),
            method_config.get('alt_pattern_256')
        )

        if not model_128:
            print(f"  128x128 model not found in {method_config['weights_dir']}")
            all_results['methods'][method_name] = {'error': '128x128 model not found'}
            continue

        if not model_256:
            print(f"  256x256 model not found in {method_config['weights_dir']}")
            all_results['methods'][method_name] = {'error': '256x256 model not found'}
            continue

        print(f"  128x128: {model_128}")
        print(f"  256x256: {model_256}")

        try:
            results = evaluate_competition(
                model_128_path=model_128,
                model_256_path=model_256,
                num_games=num_games
            )
            all_results['methods'][method_name] = results
        except Exception as e:
            print(f"  Error: {e}")
            all_results['methods'][method_name] = {'error': str(e)}

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'256 Win':>10} {'128 Win':>10} {'Draw':>10}")
    print("-" * 70)

    for method_name, results in all_results['methods'].items():
        if 'error' in results:
            print(f"{method_name:<20} {'ERROR':>10} {'':<10} {'':<10}")
        else:
            wr_256 = f"{results['win_rate_256']:.1%}"
            wr_128 = f"{results['win_rate_128']:.1%}"
            dr = f"{results['draw_rate']:.1%}"
            print(f"{method_name:<20} {wr_256:>10} {wr_128:>10} {dr:>10}")

    print("=" * 70)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate all two-snake training methods',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--num-games', type=int, default=1000,
                        help='Number of games per evaluation')
    parser.add_argument('--output', type=str,
                        default='results/data/two_snake_competition_results.json',
                        help='Output JSON file')

    args = parser.parse_args()

    evaluate_all_methods(args.num_games, args.output)


if __name__ == '__main__':
    main()
