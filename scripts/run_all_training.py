"""
Run all training scripts sequentially with configurable episodes and collect results.
This script runs each training script one at a time to avoid GPU memory issues.
"""

import subprocess
import time
import re
import csv
import argparse
from pathlib import Path
from datetime import datetime

# List of all training scripts to run
TRAINING_SCRIPTS = [
    # DQN variants (MLP)
    "scripts/training/train_dqn_mlp.py",
    "scripts/training/train_dqn_mlp_floodfill.py",
    "scripts/training/train_dqn_mlp_selective.py",
    "scripts/training/train_dqn_mlp_enhanced.py",
    "scripts/training/train_dqn_mlp_floodfill_large.py",
    "scripts/training/train_dqn_mlp_selective_large.py",
    "scripts/training/train_dqn_mlp_enhanced_large.py",

    # Advanced DQN variants
    "scripts/training/train_double_dqn_mlp.py",
    "scripts/training/train_double_dqn_mlp_floodfill.py",
    "scripts/training/train_dueling_dqn_mlp.py",
    "scripts/training/train_dueling_dqn_mlp_floodfill.py",
    "scripts/training/train_per_dqn_mlp.py",
    "scripts/training/train_per_dqn_mlp_floodfill.py",

    # CNN variants
    "scripts/training/train_dqn_cnn.py",
    "scripts/training/train_double_dqn_cnn.py",

    # PPO variants
    "scripts/training/train_ppo_mlp.py",
    "scripts/training/train_ppo_mlp_floodfill.py",
    "scripts/training/train_ppo_cnn.py",

    # REINFORCE variants
    "scripts/training/train_reinforce_mlp.py",
    "scripts/training/train_reinforce_mlp_floodfill.py",
    "scripts/training/train_reinforce_cnn.py",

    # A2C variants
    "scripts/training/train_a2c.py",
    "scripts/training/train_a2c_floodfill.py",
]

def extract_metrics_from_output(output):
    """Extract final metrics from training output."""
    metrics = {
        'avg_reward': None,
        'avg_score': None,
        'avg_length': None,
        'training_time': None,
        'episodes_per_sec': None
    }

    # Look for final summary statistics
    reward_match = re.search(r'Final\s+Avg\s+Reward:\s*([-\d.]+)', output, re.IGNORECASE)
    score_match = re.search(r'Final\s+Avg\s+Score:\s*([\d.]+)', output, re.IGNORECASE)
    length_match = re.search(r'Final\s+Avg\s+Length:\s*([\d.]+)', output, re.IGNORECASE)
    time_match = re.search(r'Total\s+Time:\s*(\d+)m\s*([\d.]+)s', output, re.IGNORECASE)
    eps_match = re.search(r'Episodes/second:\s*([\d.]+)', output, re.IGNORECASE)

    if reward_match:
        metrics['avg_reward'] = float(reward_match.group(1))
    if score_match:
        metrics['avg_score'] = float(score_match.group(1))
    if length_match:
        metrics['avg_length'] = float(length_match.group(1))
    if time_match:
        minutes = int(time_match.group(1))
        seconds = float(time_match.group(2))
        metrics['training_time'] = minutes * 60 + seconds
    if eps_match:
        metrics['episodes_per_sec'] = float(eps_match.group(1))

    return metrics

def run_training_script(script_path, episodes=1024, log_interval=100, timeout=7200):
    """Run a single training script and collect results."""
    script_name = Path(script_path).stem
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Run the training script
        result = subprocess.run(
            ["./venv/Scripts/python.exe", script_path,
             "--episodes", str(episodes),
             "--log-interval", str(log_interval)],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        elapsed_time = time.time() - start_time

        # Extract metrics from output
        output = result.stdout + result.stderr
        metrics = extract_metrics_from_output(output)

        # Check if training was successful
        success = result.returncode == 0 and metrics['avg_reward'] is not None

        result_data = {
            'script': script_name,
            'success': success,
            'returncode': result.returncode,
            'elapsed_time': elapsed_time,
            **metrics
        }

        if success:
            reward_str = f"{metrics['avg_reward']:.2f}" if metrics['avg_reward'] is not None else "N/A"
            score_str = f"{metrics['avg_score']:.2f}" if metrics['avg_score'] is not None else "N/A"
            time_str = f"{metrics['training_time']:.1f}s" if metrics['training_time'] is not None else "N/A"
            print(f"[SUCCESS] - Avg Reward: {reward_str}, "
                  f"Avg Score: {score_str}, "
                  f"Time: {time_str}")
        else:
            print(f"[FAILED] - Return code: {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr[:500]}")

        return result_data

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] after 2 hours")
        return {
            'script': script_name,
            'success': False,
            'returncode': -1,
            'elapsed_time': timeout,
            'avg_reward': None,
            'avg_score': None,
            'avg_length': None,
            'training_time': None,
            'episodes_per_sec': None
        }
    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return {
            'script': script_name,
            'success': False,
            'returncode': -1,
            'elapsed_time': time.time() - start_time,
            'avg_reward': None,
            'avg_score': None,
            'avg_length': None,
            'training_time': None,
            'episodes_per_sec': None,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Run all training scripts sequentially')
    parser.add_argument('--episodes', type=int, default=1024, help='Number of episodes to train')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout per script in seconds (default: 7200 = 2 hours)')
    args = parser.parse_args()

    print("="*80)
    print(f"RUNNING ALL TRAINING SCRIPTS - {args.episodes} EPISODES EACH")
    print("="*80)
    print(f"Total scripts to run: {len(TRAINING_SCRIPTS)}")
    print(f"Timeout per script: {args.timeout}s ({args.timeout/3600:.1f} hours)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []
    total_start_time = time.time()

    # Run each script sequentially
    for i, script in enumerate(TRAINING_SCRIPTS, 1):
        print(f"\n[{i}/{len(TRAINING_SCRIPTS)}] Starting {script}...")
        result = run_training_script(script, episodes=args.episodes, log_interval=args.log_interval, timeout=args.timeout)
        all_results.append(result)

        # Small delay between scripts to let GPU cool down
        time.sleep(5)

    total_elapsed = time.time() - total_start_time

    # Save results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('results/data')
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f'training_results_{args.episodes}ep_{timestamp}.csv'

    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['script', 'success', 'avg_reward', 'avg_score', 'avg_length',
                      'training_time', 'episodes_per_sec', 'elapsed_time', 'returncode']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary table
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Script':<40} {'Status':<10} {'Avg Reward':<12} {'Avg Score':<12} {'Time (s)':<10}")
    print("-"*80)

    successful = 0
    failed = 0

    for result in all_results:
        status = "SUCCESS" if result['success'] else "FAILED"
        reward = f"{result['avg_reward']:.2f}" if result['avg_reward'] is not None else "N/A"
        score = f"{result['avg_score']:.2f}" if result['avg_score'] is not None else "N/A"
        train_time = f"{result['training_time']:.1f}" if result['training_time'] is not None else "N/A"

        print(f"{result['script']:<40} {status:<10} {reward:<12} {score:<12} {train_time:<10}")

        if result['success']:
            successful += 1
        else:
            failed += 1

    print("="*80)
    print(f"Total scripts: {len(TRAINING_SCRIPTS)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"Results saved to: {csv_path}")
    print("="*80)

if __name__ == "__main__":
    main()
