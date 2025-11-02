# Qwen3 0.6B Fine-tuning on GSM8K with Custom Reward Function

Fine-tune Qwen3 0.6B on the GSM8K math reasoning dataset using TRL with a custom reward function that performs exact numeric answer matching.

## Setup for AWS g5.2xlarge

This project is optimized for AWS g5.2xlarge instances (8 vCPUs, 1 A10G GPU, 32GB RAM) with NVIDIA PyTorch 2.8 Deep Learning AMI.

### Installation

```bash
# Clone or upload this directory to your AWS instance
cd /path/to/g5

# Install additional dependencies (PyTorch 2.8 should already be installed in the AMI)
pip install trl peft bitsandbytes wandb

# Make the training script executable
chmod +x run_training.sh
```

### Optional: Weights & Biases Setup

For experiment tracking:

```bash
export WANDB_API_KEY="your_api_key_here"
export WANDB_PROJECT="qwen-gsm8k"
```

## Project Structure

```
.
├── reward_function.py       # Custom reward function for exact numeric matching
├── train_qwen_gsm8k.py     # Main training script
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

The training consists of two phases:

### 1. Supervised Fine-Tuning (SFT)
- Trains on full reasoning chains from GSM8K
- Uses LoRA for parameter-efficient training
- 4-bit quantization for memory efficiency

### 2. Reinforcement Learning (RL)
- Uses PPO with custom reward function
- Optimizes for exact numeric answer matching
- Fine-tunes based on correctness rewards

## Usage

### Quick Start

Run the complete pipeline:
```bash
./run_training.sh
```

### Manual Training

**Test run (100 samples):**
```bash
python train_qwen_gsm8k.py \
    --mode both \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_epochs 1 \
    --batch_size 4 \
    --use_4bit \
    --max_samples 100
```

**Full training:**
```bash
# SFT phase only
python train_qwen_gsm8k.py \
    --mode sft \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_epochs 3 \
    --batch_size 4 \
    --use_4bit

# RL phase only (requires SFT model)
python train_qwen_gsm8k.py \
    --mode rl \
    --sft_output ./qwen_gsm8k_sft \
    --rl_output ./qwen_gsm8k_rl

# Both phases
python train_qwen_gsm8k.py \
    --mode both \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_epochs 3 \
    --batch_size 4 \
    --use_4bit
```

**Evaluation:**
```bash
python train_qwen_gsm8k.py --mode eval --rl_output ./qwen_gsm8k_rl
```

## Command-line Arguments

- `--model_name`: HuggingFace model name (default: Qwen/Qwen2.5-0.5B)
- `--mode`: Training mode: sft, rl, both, or eval
- `--sft_output`: Directory for SFT model (default: ./qwen_gsm8k_sft)
- `--rl_output`: Directory for RL model (default: ./qwen_gsm8k_rl)
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Per-device batch size (default: 4)
- `--use_4bit`: Enable 4-bit quantization (default: True)
- `--max_samples`: Limit training samples for testing

## Optimization for g5.2xlarge

- **4-bit quantization**: Reduces memory usage
- **LoRA**: Parameter-efficient fine-tuning (trains only 0.5-2% of parameters)
- **Gradient checkpointing**: Saves memory at cost of computation
- **BFloat16**: Better numerical stability than FP16
- **Data loading**: Uses 4/8 vCPUs for parallel data loading
- **Gradient accumulation**: Effective batch size = batch_size × accumulation_steps

## Expected Training Time

- **SFT phase**: ~2-3 hours on full GSM8K train set (7.5K samples)
- **RL phase**: ~1-2 hours
- **Evaluation**: ~10-15 minutes on test set (1.3K samples)

## Memory Usage

With 4-bit quantization and batch size 4:
- **Model**: ~2-3 GB
- **Activations**: ~4-6 GB
- **Total**: ~8-10 GB (well within A10G's 24GB)

## Notes

- **Model Name**: Replace `Qwen/Qwen2.5-0.5B` with `Qwen/Qwen3-0.6B` when available on HuggingFace
- **Answer Format**: The model learns to output answers as `#### NUMBER`
- **Reward Signal**: Only the final numeric answer is evaluated, not the reasoning steps
- **LoRA Targets**: Configured for Qwen architecture (all attention and FFN projections)

## Troubleshooting

**Out of Memory:**
- Reduce `--batch_size` to 2
- Increase `gradient_accumulation_steps` in code
- Ensure `--use_4bit` is enabled

**Slow Training:**
- Check GPU utilization: `nvidia-smi dmon`
- Verify data loading workers: Set `dataloader_num_workers=2` if CPU-bound

**Poor Results:**
- Increase training epochs
- Try different learning rates
- Check that reward function is working: `python reward_function.py`

## Citation

```bibtex
@article{cobbe2021training,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```
