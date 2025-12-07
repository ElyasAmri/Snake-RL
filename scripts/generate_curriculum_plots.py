"""
Generate Curriculum Training Plots for Two-Snake Report Section

Generates the following figures:
1. curriculum_128x128_stages.png - 2x2 grid, win rate per stage for 128x128
2. curriculum_256x256_stages.png - 2x2 grid, win rate per stage for 256x256
3. coevolution_comparison.png - Dual curves (128 vs 256 win rates over time)
4. competition.png - Bar chart comparing all methods

Usage:
    ./venv/Scripts/python.exe scripts/generate_curriculum_plots.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def smooth(data: List[float], window: int = 10) -> np.ndarray:
    """Smooth data using moving average"""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_curriculum_stages(
    history_path: str,
    output_path: str,
    network_size: str = "128x128"
):
    """
    Plot win rate per stage in a 2x2 grid with CUMULATIVE steps on x-axis.

    Args:
        history_path: Path to curriculum_NxN_history.json
        output_path: Output PNG path
        network_size: "128x128" or "256x256" for title
    """
    with open(history_path, 'r') as f:
        data = json.load(f)

    stages = data.get('stages', {})
    if not stages:
        print(f"No stage data found in {history_path}")
        return

    # Get stage boundaries for visual reference
    stage_boundaries = data.get('stage_boundaries', [])

    # Stage thresholds
    thresholds = {
        '0': 0.95,
        '1': 0.95,
        '2': 0.35,
        '3': 0.90
    }

    stage_names = {
        '0': 'Static',
        '1': 'Random',
        '2': 'Greedy',
        '3': 'Frozen'
    }

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    for idx, (stage_id, stage_data) in enumerate(sorted(stages.items(), key=lambda x: int(x[0]))):
        if idx >= 4:
            break

        ax = axes[idx]
        history = stage_data.get('history', [])

        if not history:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        # Use total_step (cumulative) instead of stage_step
        steps = [h.get('total_step', h.get('stage_step', 0)) for h in history]
        win_rates = [h['win_rate'] for h in history]

        # Plot raw data
        ax.plot(steps, win_rates, linewidth=1, color=colors[idx], alpha=0.3, label='Raw')

        # Plot smoothed
        if len(win_rates) >= 5:
            smoothed = smooth(win_rates, min(5, len(win_rates) // 3 + 1))
            offset = len(win_rates) - len(smoothed)
            ax.plot(steps[offset:], smoothed, linewidth=2.5, color=colors[idx], label='Smoothed')

        # Threshold line
        threshold = thresholds.get(stage_id, 0.5)
        ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2,
                  label=f'Threshold ({threshold:.0%})')

        # 50% baseline
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50% baseline')

        # Mark stage start point with vertical line
        if stage_boundaries and idx < len(stage_boundaries):
            stage_start = stage_boundaries[idx].get('start_step', 0)
            if stage_start > 0:
                ax.axvline(x=stage_start, color='gray', linestyle='-', alpha=0.3, linewidth=1)

        ax.set_xlabel('Cumulative Training Steps', fontsize=11)
        ax.set_ylabel('Win Rate', fontsize=11)
        ax.set_title(f"Stage {stage_id}: {stage_names.get(stage_id, 'Unknown')}", fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        if steps:
            ax.set_xlim(min(steps) * 0.98, max(steps) * 1.02)

        # Format x-axis with K/M suffixes
        def format_steps(x, p):
            if x >= 1_000_000:
                return f'{x/1e6:.1f}M'
            elif x >= 1_000:
                return f'{x/1e3:.0f}K'
            else:
                return f'{x:.0f}'
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_steps))

    plt.suptitle(f'Curriculum Training - {network_size} Network', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_coevolution_comparison(
    history_path: str,
    output_path: str,
    max_steps: int = None
):
    """
    Plot both network win rates on same graph during co-evolution.

    Args:
        history_path: Path to coevolution_history.json
        output_path: Output PNG path
        max_steps: Maximum steps to show (truncate data beyond this)
    """
    with open(history_path, 'r') as f:
        data = json.load(f)

    history = data.get('history', [])
    if not history:
        print(f"No history data found in {history_path}")
        return

    # Truncate to max_steps if specified
    if max_steps:
        history = [h for h in history if h['step'] <= max_steps]

    steps = [h['step'] for h in history]
    win_rate_128 = [h['win_rate_128'] for h in history]
    win_rate_256 = [h['win_rate_256'] for h in history]
    draw_rate = [h.get('draw_rate', 0) for h in history]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw data with low alpha
    ax.plot(steps, win_rate_128, linewidth=1, color='#3498db', alpha=0.3)
    ax.plot(steps, win_rate_256, linewidth=1, color='#e74c3c', alpha=0.3)

    # Plot smoothed
    if len(win_rate_128) >= 10:
        window = min(10, len(win_rate_128) // 5 + 1)
        smoothed_128 = smooth(win_rate_128, window)
        smoothed_256 = smooth(win_rate_256, window)
        offset = len(win_rate_128) - len(smoothed_128)

        ax.plot(steps[offset:], smoothed_128, linewidth=2.5, color='#3498db',
               label='128x128 Win Rate')
        ax.plot(steps[offset:], smoothed_256, linewidth=2.5, color='#e74c3c',
               label='256x256 Win Rate')

        if len(draw_rate) == len(win_rate_128):
            smoothed_draw = smooth(draw_rate, window)
            ax.plot(steps[offset:], smoothed_draw, linewidth=2, color='gray',
                   linestyle='--', label='Draw Rate', alpha=0.7)
    else:
        ax.plot(steps, win_rate_128, linewidth=2, color='#3498db', label='128x128 Win Rate')
        ax.plot(steps, win_rate_256, linewidth=2, color='#e74c3c', label='256x256 Win Rate')

    # 50% baseline
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='50% (Random)')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_title('Co-evolution: 128x128 vs 256x256 Network', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max_steps if max_steps else (max(steps) if steps else 1))

    # Format x-axis with millions
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_competition(
    results_path: str,
    output_path: str
):
    """
    Plot final competition results comparing all methods.

    Args:
        results_path: Path to final_competition_results.json
        output_path: Output PNG path
    """
    with open(results_path, 'r') as f:
        data = json.load(f)

    # Handle both single result and multi-method format
    if 'methods' in data:
        # New format with 'methods' key
        results = data['methods']
    elif 'num_games' in data:
        # Single result format - wrap in dict
        results = {'Single Evaluation': data}
    else:
        # Already in correct format
        results = data

    methods = list(results.keys())
    methods = [m for m in methods if 'error' not in str(results.get(m, {}))]

    if not methods:
        print("No valid results found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Win rate comparison
    x = np.arange(len(methods))
    width = 0.25

    wins_256 = [results[m].get('win_rate_256', 0) * 100 for m in methods]
    wins_128 = [results[m].get('win_rate_128', 0) * 100 for m in methods]
    draws = [results[m].get('draw_rate', 0) * 100 for m in methods]

    bars1 = ax1.bar(x - width, wins_256, width, label='256x256 Wins', color='#e74c3c')
    bars2 = ax1.bar(x, wins_128, width, label='128x128 Wins', color='#3498db')
    bars3 = ax1.bar(x + width, draws, width, label='Draws', color='#95a5a6')

    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.set_xlabel('Training Method', fontsize=12)
    ax1.set_title('Final Competition: Win Rates by Method', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.axhline(y=50, color='black', linestyle='--', alpha=0.3)
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:
                # Place label inside bar if it's too tall (>90%)
                if height > 90:
                    ax1.annotate(f'{height:.0f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height - 5),
                               ha='center', va='top', fontsize=9, color='white', fontweight='bold')
                else:
                    ax1.annotate(f'{height:.0f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)

    # Average scores
    avg_256 = [results[m].get('avg_score_256', 0) for m in methods]
    avg_128 = [results[m].get('avg_score_128', 0) for m in methods]

    bars4 = ax2.bar(x - width/2, avg_256, width, label='256x256', color='#e74c3c')
    bars5 = ax2.bar(x + width/2, avg_128, width, label='128x128', color='#3498db')

    ax2.set_ylabel('Average Score', fontsize=12)
    ax2.set_xlabel('Training Method', fontsize=12)
    ax2.set_title('Average Scores in Competition', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_direct_coevolution(
    history_path: str,
    output_path: str,
    title: str = "PPO Direct Co-evolution"
):
    """
    Plot win rates for direct co-evolution (no curriculum).

    Args:
        history_path: Path to ppo_direct_coevolution_history.json
        output_path: Output PNG path
        title: Plot title
    """
    with open(history_path, 'r') as f:
        data = json.load(f)

    history = data.get('history', [])
    if not history:
        print(f"No history data found in {history_path}")
        return

    steps = [h['step'] for h in history]

    # Handle different key names (256/128 vs 128/256)
    if 'win_rate_256' in history[0]:
        win_rate_big = [h['win_rate_256'] for h in history]
        win_rate_small = [h['win_rate_128'] for h in history]
    else:
        win_rate_big = [h.get('win_rate_big', h.get('win_rate', 0.5)) for h in history]
        win_rate_small = [1 - h.get('win_rate_big', h.get('win_rate', 0.5)) for h in history]

    # Get draw rate if available
    draw_rate = [h.get('draw_rate', 0) for h in history]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot raw
    ax.plot(steps, win_rate_big, linewidth=1, color='#e74c3c', alpha=0.3)
    ax.plot(steps, win_rate_small, linewidth=1, color='#3498db', alpha=0.3)
    if any(d > 0 for d in draw_rate):
        ax.plot(steps, draw_rate, linewidth=1, color='#95a5a6', alpha=0.3)

    # Plot smoothed
    if len(win_rate_big) >= 10:
        window = min(10, len(win_rate_big) // 5 + 1)
        smoothed_big = smooth(win_rate_big, window)
        smoothed_small = smooth(win_rate_small, window)
        offset = len(win_rate_big) - len(smoothed_big)

        ax.plot(steps[offset:], smoothed_big, linewidth=2.5, color='#e74c3c',
               label='256x256 Win Rate')
        ax.plot(steps[offset:], smoothed_small, linewidth=2.5, color='#3498db',
               label='128x128 Win Rate')

        # Plot smoothed draw rate
        if any(d > 0 for d in draw_rate):
            smoothed_draw = smooth(draw_rate, window)
            ax.plot(steps[offset:], smoothed_draw, linewidth=2, color='#95a5a6',
                   linestyle='--', label='Draw Rate')
    else:
        ax.plot(steps, win_rate_big, linewidth=2, color='#e74c3c', label='256x256 Win Rate')
        ax.plot(steps, win_rate_small, linewidth=2, color='#3498db', label='128x128 Win Rate')
        if any(d > 0 for d in draw_rate):
            ax.plot(steps, draw_rate, linewidth=2, color='#95a5a6', linestyle='--', label='Draw Rate')

    # 50% baseline
    ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, label='50% (Random)')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 14_000_000)  # Fixed to 14M for consistency

    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all curriculum training plots"""
    data_dir = Path('results/data')
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("GENERATING CURRICULUM TRAINING PLOTS")
    print("=" * 70 + "\n")

    # 1. Curriculum stage plots
    for size in ['128x128', '256x256']:
        history_path = data_dir / f"curriculum_{size}_history.json"
        if history_path.exists():
            plot_curriculum_stages(
                str(history_path),
                str(output_dir / f"curriculum_{size}_stages.png"),
                size
            )
        else:
            print(f"Not found: {history_path}")

    # 2. Co-evolution comparison (curriculum)
    coevo_path = data_dir / "coevolution_history.json"
    if coevo_path.exists():
        plot_coevolution_comparison(
            str(coevo_path),
            str(output_dir / "coevolution_comparison.png")
        )
    else:
        print(f"Not found: {coevo_path}")

    # 3. PPO Direct co-evolution (check for both 2M and 14M)
    for steps_label in ['2M', '14M']:
        direct_path = data_dir / f"ppo_direct_coevolution_{steps_label}_history.json"
        if direct_path.exists():
            plot_direct_coevolution(
                str(direct_path),
                str(output_dir / f"ppo_direct_coevolution_{steps_label}.png"),
                f"PPO Direct Co-evolution ({steps_label}, No Curriculum)"
            )
    # Also check for old format without step label
    direct_path = data_dir / "ppo_direct_coevolution_history.json"
    if direct_path.exists():
        plot_direct_coevolution(
            str(direct_path),
            str(output_dir / "ppo_direct_coevolution.png"),
            "PPO Direct Co-evolution (No Curriculum)"
        )

    # 4. Competition results
    competition_path = data_dir / "two_snake_competition_results.json"
    if competition_path.exists():
        plot_final_competition(
            str(competition_path),
            str(output_dir / "competition.png")
        )
    else:
        # Try old filename
        competition_path = data_dir / "competition_results.json"
        if competition_path.exists():
            plot_final_competition(
                str(competition_path),
                str(output_dir / "competition.png")
            )
        else:
            print(f"Not found: competition results file")

    print("\n" + "=" * 70)
    print("PLOT GENERATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
