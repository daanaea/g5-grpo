#!/usr/bin/env python3
"""
Visualize GRPO training logs from JSONL file.
Creates comprehensive plots of training metrics.

Usage:
    python3 visualize_logs.py --log_file ./qwen_gsm8k_grpo/grpo_training_logs.jsonl --output ./plots
"""

import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for AWS/headless environments
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def load_logs(log_file):
    """Load JSONL training logs into a list of dictionaries."""
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs


def extract_metrics(logs):
    """Extract metrics from logs into separate lists."""
    metrics = {
        'steps': [],
        'epochs': [],
        'timestamps': [],
        'grpo_loss': [],
        'policy_loss': [],
        'kl_loss': [],
        'reward_mean': [],
        'reward_std': [],
        'adv_mean': [],
        'adv_std': [],
        'grad_norm': [],
        'learning_rate': [],
        'time_per_step': [],
        'generation_time': [],
        'backward_time': [],
        'other_time': [],
        'tokens_generated': [],
    }

    for log in logs:
        metrics['steps'].append(log.get('step', 0))
        metrics['epochs'].append(log.get('epoch', 0))
        metrics['timestamps'].append(log.get('timestamp_iso', ''))
        metrics['grpo_loss'].append(log.get('grpo_loss', 0))
        metrics['policy_loss'].append(log.get('policy_loss', 0))
        metrics['kl_loss'].append(log.get('kl_loss', 0))
        metrics['reward_mean'].append(log.get('reward_mean'))
        metrics['reward_std'].append(log.get('reward_std'))
        metrics['adv_mean'].append(log.get('adv_mean'))
        metrics['adv_std'].append(log.get('adv_std'))
        metrics['grad_norm'].append(log.get('grad_norm', 0))
        metrics['learning_rate'].append(log.get('learning_rate', 0))
        metrics['time_per_step'].append(log.get('time_per_step_s', 0))
        metrics['generation_time'].append(log.get('generation_time_s', 0))
        metrics['backward_time'].append(log.get('backward_time_s', 0))
        metrics['other_time'].append(log.get('other_time_s', 0))
        metrics['tokens_generated'].append(log.get('tokens_generated', 0))

    return metrics


def plot_losses(metrics, output_dir):
    """Plot loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    steps = metrics['steps']

    # GRPO Loss
    axes[0, 0].plot(steps, metrics['grpo_loss'], 'b-', linewidth=2, label='GRPO Loss')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('GRPO Loss Over Training')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Policy Loss and KL Loss
    axes[0, 1].plot(steps, metrics['policy_loss'], 'g-', linewidth=2, label='Policy Loss')
    axes[0, 1].plot(steps, metrics['kl_loss'], 'r-', linewidth=2, label='KL Loss')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Policy Loss and KL Divergence')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Gradient Norm
    axes[1, 0].plot(steps, metrics['grad_norm'], 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('Gradient Norm Over Training')
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(steps, metrics['learning_rate'], 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    output_path = output_dir / 'losses.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved loss plots to {output_path}")
    plt.close()


def plot_rewards(metrics, output_dir):
    """Plot reward metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    steps = metrics['steps']

    # Reward Mean with Std
    reward_mean = np.array([r if r is not None else 0 for r in metrics['reward_mean']])
    reward_std = np.array([r if r is not None else 0 for r in metrics['reward_std']])

    axes[0].plot(steps, reward_mean, 'b-', linewidth=2, label='Reward Mean')
    axes[0].fill_between(steps,
                          reward_mean - reward_std,
                          reward_mean + reward_std,
                          alpha=0.3, label='± Std')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Reward Mean ± Std')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Advantages Mean with Std
    adv_mean = np.array([a if a is not None else 0 for a in metrics['adv_mean']])
    adv_std = np.array([a if a is not None else 0 for a in metrics['adv_std']])

    axes[1].plot(steps, adv_mean, 'g-', linewidth=2, label='Advantage Mean')
    axes[1].fill_between(steps,
                          adv_mean - adv_std,
                          adv_mean + adv_std,
                          alpha=0.3, label='± Std')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Advantage')
    axes[1].set_title('Advantage Mean ± Std')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'rewards.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved reward plots to {output_path}")
    plt.close()


def plot_timing(metrics, output_dir):
    """Plot timing breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    steps = metrics['steps']

    # Stacked time breakdown
    generation_time = np.array(metrics['generation_time'])
    backward_time = np.array(metrics['backward_time'])
    other_time = np.array(metrics['other_time'])

    axes[0].stackplot(steps, generation_time, backward_time, other_time,
                      labels=['Generation', 'Backward', 'Other'],
                      colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                      alpha=0.7)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].set_title('Time Breakdown Per Step')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')

    # Total time per step
    total_time = np.array(metrics['time_per_step'])
    axes[1].plot(steps, total_time, 'b-', linewidth=2)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Total Time Per Step')
    axes[1].grid(True, alpha=0.3)

    # Add average line
    if len(total_time) > 0:
        avg_time = np.mean(total_time[total_time > 0])
        axes[1].axhline(y=avg_time, color='r', linestyle='--',
                        label=f'Average: {avg_time:.2f}s')
        axes[1].legend()

    plt.tight_layout()
    output_path = output_dir / 'timing.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved timing plots to {output_path}")
    plt.close()


def plot_tokens(metrics, output_dir):
    """Plot tokens generated."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    steps = metrics['steps']
    tokens = metrics['tokens_generated']

    ax.plot(steps, tokens, 'purple', linewidth=2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Tokens Generated')
    ax.set_title('Tokens Generated Per Step')
    ax.grid(True, alpha=0.3)

    # Add average line
    tokens_array = np.array(tokens)
    if len(tokens_array) > 0 and np.any(tokens_array > 0):
        avg_tokens = np.mean(tokens_array[tokens_array > 0])
        ax.axhline(y=avg_tokens, color='r', linestyle='--',
                   label=f'Average: {avg_tokens:.0f} tokens')
        ax.legend()

    plt.tight_layout()
    output_path = output_dir / 'tokens.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved tokens plot to {output_path}")
    plt.close()


def print_summary(metrics):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    if len(metrics['steps']) == 0:
        print("No training data found!")
        return

    print(f"Total Steps: {max(metrics['steps'])}")
    print(f"Total Epochs: {max(metrics['epochs']):.2f}")

    # Loss statistics
    grpo_loss = np.array(metrics['grpo_loss'])
    if len(grpo_loss) > 0:
        print(f"\nGRPO Loss:")
        print(f"  Initial: {grpo_loss[0]:.4f}")
        print(f"  Final:   {grpo_loss[-1]:.4f}")
        print(f"  Min:     {np.min(grpo_loss):.4f}")
        print(f"  Mean:    {np.mean(grpo_loss):.4f}")

    # Reward statistics
    reward_mean = np.array([r for r in metrics['reward_mean'] if r is not None])
    if len(reward_mean) > 0:
        print(f"\nRewards:")
        print(f"  Mean:    {np.mean(reward_mean):.4f}")
        print(f"  Std:     {np.std(reward_mean):.4f}")
        print(f"  Min:     {np.min(reward_mean):.4f}")
        print(f"  Max:     {np.max(reward_mean):.4f}")

    # Timing statistics
    time_per_step = np.array([t for t in metrics['time_per_step'] if t > 0])
    if len(time_per_step) > 0:
        print(f"\nTiming:")
        print(f"  Avg time/step:  {np.mean(time_per_step):.2f}s")
        print(f"  Total time:     {np.sum(time_per_step) / 60:.2f} minutes")

        gen_time = np.array([t for t in metrics['generation_time'] if t > 0])
        back_time = np.array([t for t in metrics['backward_time'] if t > 0])
        if len(gen_time) > 0 and len(back_time) > 0:
            print(f"  Avg generation: {np.mean(gen_time):.2f}s")
            print(f"  Avg backward:   {np.mean(back_time):.2f}s")

    # Tokens statistics
    tokens = np.array([t for t in metrics['tokens_generated'] if t > 0])
    if len(tokens) > 0:
        print(f"\nTokens:")
        print(f"  Avg/step: {np.mean(tokens):.0f}")
        print(f"  Total:    {np.sum(tokens):.0f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize GRPO training logs")
    parser.add_argument(
        "--log_file",
        type=str,
        default="./qwen_gsm8k_grpo/grpo_training_logs.jsonl",
        help="Path to JSONL log file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./plots",
        help="Output directory for plots"
    )
    args = parser.parse_args()

    # Check if log file exists
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        print(f"Please run training first or specify correct path with --log_file")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading logs from {log_path}...")
    logs = load_logs(log_path)
    print(f"Loaded {len(logs)} log entries")

    if len(logs) == 0:
        print("No logs found in file!")
        return

    # Extract metrics
    metrics = extract_metrics(logs)

    # Create plots
    print("\nGenerating plots...")
    plot_losses(metrics, output_dir)
    plot_rewards(metrics, output_dir)
    plot_timing(metrics, output_dir)
    plot_tokens(metrics, output_dir)

    # Print summary
    print_summary(metrics)

    print(f"\nAll plots saved to {output_dir}/")
    print("Generated files:")
    print(f"  - {output_dir}/losses.png")
    print(f"  - {output_dir}/rewards.png")
    print(f"  - {output_dir}/timing.png")
    print(f"  - {output_dir}/tokens.png")


if __name__ == "__main__":
    main()
