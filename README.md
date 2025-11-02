# Qwen3 0.6B Fine-tuning on GSM8K with Custom Reward Function

Fine-tune Qwen3 0.6B on the GSM8K math reasoning dataset using TRL with a custom reward function that performs exact numeric answer matching.

## Setup for AWS g5.2xlarge

This project is optimized for AWS g5.2xlarge instances (8 vCPUs, 1 A10G GPU, 32GB RAM) with NVIDIA PyTorch 2.8 Deep Learning AMI.

### Installation

```bash
# Clone or upload this directory to your AWS instance
cd /path/to/g5-grpo

# Install additional dependencies (PyTorch 2.8 should already be installed in the AMI)
python3 -m pip install --user trl peft bitsandbytes

# Make the training script executable
chmod +x run_training.sh
```

## Project Structure

```
.
├── reward_function.py       # Custom reward function for exact numeric matching
├── train.py                 # Main training script
├── visualize_logs.py        # Visualization script for training logs
├── run_training.sh          # Bash script to run the full pipeline
└── README.md                # This file
```

## Custom Reward Function

The reward function extracts and compares numeric answers:

- **Extraction**: Looks for `#### NUMBER` pattern in generated text
- **Comparison**: Exact match with ground truth (with floating point tolerance)
- **Rewards**: 1.0 for correct, 0.0 for incorrect

Test the reward function:
```bash
python reward_function.py
```

## Training Pipeline

The training uses GRPO (Group Relative Policy Optimization):

- Optimizes policy using custom reward function
- Performs exact numeric answer matching
- Comprehensive logging of training metrics
- Optional SFT phase for pre-training on reasoning chains

## Usage

### Quick Start

Run the complete pipeline:
```bash
./run_training.sh
```

### Manual Training

**Test run (100 samples):**
```bash
python3 train.py \
    --mode grpo \
    --model_name Qwen/Qwen3-0.6B \
    --max_samples 100 \
    --grpo_output ./qwen_gsm8k_grpo
```

**Full training:**
```bash
# GRPO training only (recommended)
python3 train.py \
    --mode grpo \
    --model_name Qwen/Qwen3-0.6B \
    --grpo_output ./qwen_gsm8k_grpo

# SFT + GRPO (optional)
python3 train.py \
    --mode both \
    --model_name Qwen/Qwen3-0.6B \
    --num_epochs 3 \
    --batch_size 4 \
    --sft_output ./qwen_gsm8k_sft \
    --grpo_output ./qwen_gsm8k_grpo
```

**Evaluation:**
```bash
python3 train.py --mode eval --grpo_output ./qwen_gsm8k_grpo
```

## Command-line Arguments

- `--model_name`: HuggingFace model name (default: Qwen/Qwen3-0.6B)
- `--mode`: Training mode: sft, grpo, both, or eval
- `--sft_output`: Directory for SFT model (default: ./qwen_gsm8k_sft)
- `--grpo_output`: Directory for GRPO model (default: ./qwen_gsm8k_grpo)
- `--num_epochs`: Number of SFT training epochs (default: 3)
- `--batch_size`: Per-device SFT batch size (default: 4)
- `--use_4bit`: Enable 4-bit quantization for SFT (default: False)
- `--max_samples`: Limit training samples for testing

## Optimization for g5.2xlarge

- **BFloat16**: Better numerical stability than FP16
- **Data loading**: Uses multiple vCPUs for parallel data loading
- **GRPO Config**: Optimized for single A10G GPU (batch_size=1, num_generations=4)
- **Optional LoRA**: Available for SFT phase with 4-bit quantization

## Training Logs

GRPO training logs are saved to `./qwen_gsm8k_grpo/grpo_training_logs.jsonl` with comprehensive metrics:
- timestamp_iso, step, epoch, device, seed
- grpo_loss, policy_loss, kl_loss
- reward_mean, reward_std
- adv_mean, adv_std
- grad_norm, learning_rate
- time_per_step_s, generation_time_s, backward_time_s, other_time_s
- tokens_generated

### Visualizing Training Progress

Visualize training metrics with plots:

```bash
# Basic usage (uses default paths)
python3 visualize_logs.py

# Specify custom log file and output directory
python3 visualize_logs.py \
    --log_file ./qwen_gsm8k_grpo/grpo_training_logs.jsonl \
    --output ./plots
```

This generates 4 plots:
- **losses.png**: GRPO loss, policy loss, KL loss, gradient norm, learning rate
- **rewards.png**: Reward mean ± std, advantage mean ± std
- **timing.png**: Time breakdown (generation/backward/other), total time per step
- **tokens.png**: Tokens generated per step

The script also prints a summary of training statistics to the console.

## Notes

- **Model**: Using Qwen/Qwen3-0.6B (0.6B parameters)
- **Answer Format**: The model learns to output answers as `#### NUMBER`
- **Reward Signal**: Only the final numeric answer is evaluated, not the reasoning steps
- **GRPO Parameters**: temperature=1.0, beta=0.1, epsilon=0.2, num_generations=4

## Troubleshooting

**Out of Memory:**
- GRPO is already optimized with batch_size=1
- For SFT, reduce `--batch_size` to 2 or enable `--use_4bit`

**Slow Training:**
- Check GPU utilization: `nvidia-smi dmon`
- Verify data loading workers in config

**Poor Results:**
- Check that reward function is working: `python3 reward_function.py`
- Verify training logs show non-zero rewards
- Try adjusting GRPO hyperparameters (beta, epsilon, temperature)

## Citation

```bibtex
@article{cobbe2021training,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```
