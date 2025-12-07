"""
Master Script for Two-Snake Competitive Training Experiments

Runs all training experiments for the two-snake competitive environment section:
1. DQN Direct Co-evolution (2M steps)
2. PPO Direct Co-evolution (2M steps - short baseline)
3. PPO Curriculum 128x128 (6M steps total: 0.5M + 0.5M + 3M + 2M)
4. PPO Curriculum 256x256 (6M steps total: 0.5M + 0.5M + 3M + 2M)
5. PPO Co-evolution with curriculum checkpoints (8M steps)
6. PPO Direct Co-evolution (14M steps - full comparison)

Usage:
    # Full training run
    ./venv/Scripts/python.exe scripts/run_two_snake_experiments.py

    # Quick test run (1% of actual steps)
    ./venv/Scripts/python.exe scripts/run_two_snake_experiments.py --test

    # Run only specific experiments
    ./venv/Scripts/python.exe scripts/run_two_snake_experiments.py --only dqn ppo_direct_2m
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import subprocess
import argparse
import time
import json
from datetime import datetime


# Experiment definitions
EXPERIMENTS = {
    'dqn': {
        'name': 'DQN Direct Co-evolution (2M)',
        'script': 'scripts/training/train_dqn_two_snake_mlp.py',
        'args': {
            'full': ['--total-steps', '2000000', '--save-dir', 'results/weights/dqn_direct_coevolution'],
            'test': ['--total-steps', '20000', '--save-dir', 'results/weights/test_dqn_direct']
        }
    },
    'ppo_direct_2m': {
        'name': 'PPO Direct Co-evolution (2M)',
        'script': 'scripts/training/train_ppo_direct_coevolution.py',
        'args': {
            'full': ['--total-steps', '2000000', '--save-dir', 'results/weights/ppo_direct_coevolution_2M'],
            'test': ['--total-steps', '20000', '--save-dir', 'results/weights/test_ppo_direct_2m']
        }
    },
    'curriculum_128': {
        'name': 'PPO Curriculum 128x128',
        'script': 'scripts/training/train_ppo_curriculum_independent.py',
        'args': {
            'full': ['--hidden-dims', '128', '128', '--save-dir', 'results/weights/ppo_curriculum_128x128'],
            'test': ['--hidden-dims', '128', '128', '--save-dir', 'results/weights/test_ppo_curriculum_128x128']
        },
        'test_override': {
            'stages': [
                {'min_steps': 5000, 'max_steps': 5000},  # Stage 0
                {'min_steps': 5000, 'max_steps': 5000},  # Stage 1
                {'min_steps': 30000, 'max_steps': 30000},  # Stage 2
                {'min_steps': 20000, 'max_steps': 20000},  # Stage 3
            ]
        }
    },
    'curriculum_256': {
        'name': 'PPO Curriculum 256x256',
        'script': 'scripts/training/train_ppo_curriculum_independent.py',
        'args': {
            'full': ['--hidden-dims', '256', '256', '--save-dir', 'results/weights/ppo_curriculum_256x256'],
            'test': ['--hidden-dims', '256', '256', '--save-dir', 'results/weights/test_ppo_curriculum_256x256']
        },
        'test_override': {
            'stages': [
                {'min_steps': 5000, 'max_steps': 5000},  # Stage 0
                {'min_steps': 5000, 'max_steps': 5000},  # Stage 1
                {'min_steps': 30000, 'max_steps': 30000},  # Stage 2
                {'min_steps': 20000, 'max_steps': 20000},  # Stage 3
            ]
        }
    },
    'coevolution': {
        'name': 'PPO Co-evolution (8M)',
        'script': 'scripts/training/train_ppo_coevolution_cross.py',
        'args': {
            'full': [
                '--checkpoint-128', 'results/weights/ppo_curriculum_128x128/stage3_*128x128*.pt',
                '--checkpoint-256', 'results/weights/ppo_curriculum_256x256/stage3_*256x256*.pt',
                '--min-steps', '8000000',
                '--max-steps', '8000000',
                '--save-dir', 'results/weights/ppo_coevolution'
            ],
            'test': [
                '--checkpoint-128', 'results/weights/test_ppo_curriculum_128x128/stage3_*128x128*.pt',
                '--checkpoint-256', 'results/weights/test_ppo_curriculum_256x256/stage3_*256x256*.pt',
                '--min-steps', '80000',
                '--max-steps', '80000',
                '--save-dir', 'results/weights/test_ppo_coevolution'
            ]
        },
        'depends_on': ['curriculum_128', 'curriculum_256']
    },
    'ppo_direct_14m': {
        'name': 'PPO Direct Co-evolution (14M)',
        'script': 'scripts/training/train_ppo_direct_coevolution.py',
        'args': {
            'full': ['--total-steps', '14000000', '--save-dir', 'results/weights/ppo_direct_coevolution_14M'],
            'test': ['--total-steps', '140000', '--save-dir', 'results/weights/test_ppo_direct_14m']
        }
    }
}

# Default run order
DEFAULT_ORDER = ['dqn', 'ppo_direct_2m', 'curriculum_128', 'curriculum_256', 'coevolution', 'ppo_direct_14m']


def run_experiment(name: str, mode: str = 'full', python_exe: str = './venv/Scripts/python.exe'):
    """Run a single experiment"""
    exp = EXPERIMENTS[name]
    script = exp['script']
    args = exp['args'][mode]

    print("\n" + "=" * 70)
    print(f"STARTING: {exp['name']}")
    print(f"Mode: {mode}")
    print(f"Script: {script}")
    print(f"Args: {' '.join(args)}")
    print("=" * 70 + "\n")

    start_time = time.time()

    # Build command
    cmd = [python_exe, script] + args

    # Run the experiment
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    elapsed = time.time() - start_time

    print("\n" + "-" * 70)
    print(f"COMPLETED: {exp['name']}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Exit code: {result.returncode}")
    print("-" * 70 + "\n")

    return {
        'name': name,
        'elapsed_seconds': elapsed,
        'exit_code': result.returncode
    }


def run_all_experiments(experiments: list, mode: str = 'full', python_exe: str = './venv/Scripts/python.exe'):
    """Run all specified experiments in order"""
    results = []
    total_start = time.time()

    print("\n" + "=" * 70)
    print("TWO-SNAKE COMPETITIVE TRAINING EXPERIMENTS")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Experiments: {', '.join(experiments)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    for exp_name in experiments:
        if exp_name not in EXPERIMENTS:
            print(f"Warning: Unknown experiment '{exp_name}', skipping...")
            continue

        exp = EXPERIMENTS[exp_name]

        # Check dependencies
        if 'depends_on' in exp:
            for dep in exp['depends_on']:
                if dep not in [r['name'] for r in results]:
                    print(f"Warning: {exp_name} depends on {dep} which hasn't run yet")

        result = run_experiment(exp_name, mode, python_exe)
        results.append(result)

        # Stop on failure
        if result['exit_code'] != 0:
            print(f"\nError: {exp_name} failed with exit code {result['exit_code']}")
            print("Stopping experiments.")
            break

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENTS SUMMARY")
    print("=" * 70)
    for r in results:
        status = "SUCCESS" if r['exit_code'] == 0 else "FAILED"
        print(f"  {r['name']}: {status} ({r['elapsed_seconds']/60:.1f} min)")
    print("-" * 70)
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print("=" * 70 + "\n")

    # Save summary
    summary = {
        'mode': mode,
        'total_time_seconds': total_elapsed,
        'experiments': results,
        'timestamp': datetime.now().isoformat()
    }

    data_dir = Path('results/data')
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_path = data_dir / f'experiments_summary_{mode}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run two-snake competitive training experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full training run (all experiments)
    ./venv/Scripts/python.exe scripts/run_two_snake_experiments.py

    # Quick test run (~1% of steps)
    ./venv/Scripts/python.exe scripts/run_two_snake_experiments.py --test

    # Run only specific experiments
    ./venv/Scripts/python.exe scripts/run_two_snake_experiments.py --only dqn ppo_direct_2m

    # List available experiments
    ./venv/Scripts/python.exe scripts/run_two_snake_experiments.py --list
        """
    )
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with reduced steps (~1% of full)')
    parser.add_argument('--only', nargs='+', choices=list(EXPERIMENTS.keys()),
                        help='Run only specified experiments')
    parser.add_argument('--list', action='store_true',
                        help='List available experiments and exit')
    parser.add_argument('--python', type=str, default='./venv/Scripts/python.exe',
                        help='Path to Python executable')

    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        print("-" * 50)
        for name, exp in EXPERIMENTS.items():
            deps = f" (depends on: {', '.join(exp.get('depends_on', []))})" if 'depends_on' in exp else ""
            print(f"  {name}: {exp['name']}{deps}")
        print("-" * 50)
        print(f"\nDefault order: {', '.join(DEFAULT_ORDER)}")
        return

    mode = 'test' if args.test else 'full'
    experiments = args.only if args.only else DEFAULT_ORDER

    run_all_experiments(experiments, mode, args.python)


if __name__ == '__main__':
    main()
