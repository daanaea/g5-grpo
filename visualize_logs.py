#!/usr/bin/env python3
"""
Visualize GRPO training logs from JSONL file.
Creates comprehensive plots of training metrics.

Usage:
    python3 visualize_logs.py --log_file ./qwen_gsm8k_grpo/grpo_training_logs.jsonl --output ./plots
    python3 visualize_logs.py --debug  # Show available log keys
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for AWS/headless environments
import matplotlib.pyplot as plt
import numpy as np


def load_logs(log_file):
    """Load JSONL training logs into a list of dictionaries."""
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON line: {e}")
    return logs


def analyze_log_structure(logs):
    """Analyze log structure and print available keys."""
    if not logs:
        print("No logs to analyze!")
        return

    print("\n" + "=" * 60)
    print("LOG STRUCTURE ANALYSIS")
    print("=" * 60)

    # Collect all keys
    all_keys = set()
    key_types = defaultdict(set)

    for log in logs:
        for key, value in log.items():
            all_keys.add(key)
            key_types[key].add(type(value).__name__)

    print(f"\nTotal log entries: {len(logs)}")
    print(f"\nAvailable keys ({len(all_keys)}):")
    for key in sorted(all_keys):
        types = ', '.join(sorted(key_types[key]))
        # Sample value from first log
        sample = logs[0].get(key, 'N/A')
        if isinstance(sample, float):
            print(f"  - {key:25s} (types: {types:15s}) sample: {sample:.4f}")
        else:
            print(f"  - {key:25s} (types: {types:15s}) sample: {sample}")

    print("\nFirst log entry:")
    print(json.dumps(logs[0], indent=2))

    if len(logs) > 1:
        print("\nLast log entry:")
        print(json.dumps(logs[-1], indent=2))

    print("=" * 60 + "\n")


def has_valid_data(values):
    """Check if a list has any non-zero, non-None values."""
    if not values:
        return False
    valid = [v for v in values if v is not None and v != 0]
    return len(valid) > 0


def extract_metrics(logs, debug=False):
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
        metrics['grpo_loss'].append(log.get('grpo_loss', log.get('loss', 0)))
        metrics['policy_loss'].append(log.get('policy_loss', 0))
        metrics['kl_loss'].append(log.get('kl_loss', log.get('kl', 0)))

        # Try multiple possible key names for reward
        reward_mean = (log.get('reward_mean') or
                      log.get('rewards/mean') or
                      log.get('reward/default/mean') or
                      log.get('reward'))
        reward_std = (log.get('reward_std') or
                     log.get('rewards/std') or
                     log.get('reward/default/std'))

        metrics['reward_mean'].append(reward_mean)
        metrics['reward_std'].append(reward_std)

        # Try multiple possible key names for advantages
        adv_mean = (log.get('adv_mean') or
                   log.get('advantages/mean') or
                   log.get('advantage/default/mean'))
        adv_std = (log.get('adv_std') or
                  log.get('advantages/std') or
                  log.get('advantage/default/std'))

        metrics['adv_mean'].append(adv_mean)
        metrics['adv_std'].append(adv_std)

        metrics['grad_norm'].append(log.get('grad_norm', 0))
        metrics['learning_rate'].append(log.get('learning_rate', 0))
        metrics['time_per_step'].append(log.get('time_per_step_s', 0))
        metrics['generation_time'].append(log.get('generation_time_s', 0))
        metrics['backward_time'].append(log.get('backward_time_s', 0))
        metrics['other_time'].append(log.get('other_time_s', 0))
        metrics['tokens_generated'].append(log.get('tokens_generated', 0))

    if debug:
        print("\nMetric data availability:")
        for key, values in metrics.items():
            valid = has_valid_data(values)
            status = "✓" if valid else "✗"
            non_none = sum(1 for v in values if v is not None)
            non_zero = sum(1 for v in values if v not in (None, 0))
            print(f"  {status} {key:25s} - {non_none}/{len(values)} non-None, {non_zero}/{len(values)} non-zero")

    return metrics


def plot_losses(metrics, output_dir):
    """Plot loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    steps = metrics['steps']

    # GRPO Loss
    grpo_loss = metrics['grpo_loss']
    if has_valid_data(grpo_loss):
        print('found grpo loss!')
        print(grpo_loss)
        axes[0, 0].scatter(steps, grpo_loss, label='GRPO loss')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('GRPO Loss Over Training')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, 'No GRPO loss data', ha='center', va='center')
        axes[0, 0].set_title('GRPO Loss (No Data)')

    # Policy Loss and KL Loss
    policy_loss = metrics['policy_loss']
    kl_loss = metrics['kl_loss']
    has_policy = has_valid_data(policy_loss)
    has_kl = has_valid_data(kl_loss)

    if has_policy or has_kl:
        if has_policy:
            axes[0, 1].scatter(steps, policy_loss, c='green', label='Policy Loss')
        if has_kl:
            axes[0, 1].scatter(steps, kl_loss, c='red', label='KL Loss')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Policy Loss and KL Divergence')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No policy/KL loss data', ha='center', va='center')
        axes[0, 1].set_title('Policy/KL Loss (No Data)')

    # Gradient Norm
    grad_norm = metrics['grad_norm']
    if has_valid_data(grad_norm):
        axes[1, 0].scatter(steps, grad_norm, c='magenta')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Norm Over Training')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No gradient norm data', ha='center', va='center')
        axes[1, 0].set_title('Gradient Norm (No Data)')

    # Learning Rate
    lr = metrics['learning_rate']
    if has_valid_data(lr):
        axes[1, 1].scatter(steps, lr, c='orange')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
    else:
        axes[1, 1].text(0.5, 0.5, 'No learning rate data', ha='center', va='center')
        axes[1, 1].set_title('Learning Rate (No Data)')

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
    reward_mean_data = metrics['reward_mean']
    reward_std_data = metrics['reward_std']

    has_reward = has_valid_data(reward_mean_data)

    if has_reward:
        reward_mean = np.array([r if r is not None else 0 for r in reward_mean_data])
        reward_std = np.array([r if r is not None else 0 for r in reward_std_data])

        axes[0].plot(steps, reward_mean, 'b-', linewidth=2, label='Reward Mean')
        if has_valid_data(reward_std_data):
            axes[0].fill_between(steps,
                                  reward_mean - reward_std,
                                  reward_mean + reward_std,
                                  alpha=0.3, label='± Std')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Reward Mean ± Std')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'No reward data\nCheck log keys', ha='center', va='center')
        axes[0].set_title('Reward (No Data)')

    # Advantages Mean with Std
    adv_mean_data = metrics['adv_mean']
    adv_std_data = metrics['adv_std']

    has_adv = has_valid_data(adv_mean_data)

    if has_adv:
        adv_mean = np.array([a if a is not None else 0 for a in adv_mean_data])
        adv_std = np.array([a if a is not None else 0 for a in adv_std_data])

        axes[1].plot(steps, adv_mean, 'g-', linewidth=2, label='Advantage Mean')
        if has_valid_data(adv_std_data):
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
    else:
        axes[1].text(0.5, 0.5, 'No advantage data\nCheck log keys', ha='center', va='center')
        axes[1].set_title('Advantage (No Data)')

    plt.tight_layout()
    output_path = output_dir / 'rewards.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved reward plots to {output_path}")
    plt.close()


def plot_timing(metrics, output_dir):
    """Plot timing breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    steps = metrics['steps']

    # Check if we have timing data
    has_gen = has_valid_data(metrics['generation_time'])
    has_back = has_valid_data(metrics['backward_time'])
    has_other = has_valid_data(metrics['other_time'])
    has_total = has_valid_data(metrics['time_per_step'])

    # Stacked time breakdown
    if has_gen or has_back or has_other:
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
    else:
        axes[0].text(0.5, 0.5, 'No timing breakdown data', ha='center', va='center')
        axes[0].set_title('Time Breakdown (No Data)')

    # Total time per step
    if has_total:
        total_time = np.array(metrics['time_per_step'])
        axes[1].plot(steps, total_time, 'b-', linewidth=2)
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].set_title('Total Time Per Step')
        axes[1].grid(True, alpha=0.3)

        # Add average line
        if len(total_time) > 0:
            avg_time = np.mean(total_time[total_time > 0])
            if avg_time > 0:
                axes[1].axhline(y=avg_time, color='r', linestyle='--',
                                label=f'Average: {avg_time:.2f}s')
                axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'No total time data', ha='center', va='center')
        axes[1].set_title('Total Time (No Data)')

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

    if has_valid_data(tokens):
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
    else:
        ax.text(0.5, 0.5, 'No tokens generated data\nGRPO may not log this metric',
                ha='center', va='center', fontsize=12)
        ax.set_title('Tokens Generated (No Data)')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Tokens Generated')

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
    grpo_loss = np.array([l for l in metrics['grpo_loss'] if l != 0])
    if len(grpo_loss) > 0:
        print(f"\nGRPO Loss:")
        print(f"  Initial: {grpo_loss[0]:.4f}")
        print(f"  Final:   {grpo_loss[-1]:.4f}")
        print(f"  Min:     {np.min(grpo_loss):.4f}")
        print(f"  Mean:    {np.mean(grpo_loss):.4f}")

    # Reward statistics
    reward_mean = np.array([r for r in metrics['reward_mean'] if r is not None and r != 0])
    if len(reward_mean) > 0:
        print(f"\nRewards:")
        print(f"  Mean:    {np.mean(reward_mean):.4f}")
        print(f"  Std:     {np.std(reward_mean):.4f}")
        print(f"  Min:     {np.min(reward_mean):.4f}")
        print(f"  Max:     {np.max(reward_mean):.4f}")
    else:
        print(f"\nRewards: No data (check if GRPO is logging rewards)")

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
    else:
        print(f"\nTokens: No data (GRPO may not log this)")

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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed log structure and available keys"
    )
    args = parser.parse_args()

    # Check if log file exists
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        print(f"Please run training first or specify correct path with --log_file")
        return

    print(f"Loading logs from {log_path}...")
    logs = load_logs(log_path)
    print(f"Loaded {len(logs)} log entries")

    if len(logs) == 0:
        print("No logs found in file!")
        return

    # Debug mode: show log structure
    if args.debug:
        analyze_log_structure(logs)

    # Extract metrics
    metrics = extract_metrics(logs, debug=args.debug)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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
