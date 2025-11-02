#!/bin/bash

# Training script optimized for AWS g5.2xlarge instance
# This script runs the complete training pipeline

set -e

echo "========================================="
echo "Qwen GSM8K Fine-tuning on AWS g5.2xlarge"
echo "========================================="

# Check if we're on AWS instance
if [ -f /etc/os-release ]; then
    echo "OS Info:"
    cat /etc/os-release | grep PRETTY_NAME
fi

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q trl peft bitsandbytes wandb

# Optional: Setup wandb for experiment tracking
# Uncomment and set your API key
# export WANDB_API_KEY="your_wandb_api_key_here"
# export WANDB_PROJECT="qwen-gsm8k"

# Test reward function
echo ""
echo "Testing reward function..."
python reward_function.py

# Run training
echo ""
echo "Starting training pipeline..."

# Option 1: Quick test run with limited samples (recommended for testing)
# python train_qwen_gsm8k.py \
#     --mode both \
#     --model_name Qwen/Qwen2.5-0.5B \
#     --num_epochs 1 \
#     --batch_size 4 \
#     --use_4bit \
#     --max_samples 100

# Option 2: Full training run (replace Qwen2.5-0.5B with Qwen3-0.6B when available)
python train_qwen_gsm8k.py \
    --mode both \
    --model_name Qwen/Qwen2.5-0.5B \
    --num_epochs 3 \
    --batch_size 4 \
    --use_4bit \
    --sft_output ./qwen_gsm8k_sft \
    --rl_output ./qwen_gsm8k_rl

# Evaluate the model
echo ""
echo "Evaluating model..."
python train_qwen_gsm8k.py --mode eval --rl_output ./qwen_gsm8k_rl

echo ""
echo "Training complete! Models saved to:"
echo "  - SFT model: ./qwen_gsm8k_sft"
echo "  - RL model: ./qwen_gsm8k_rl"
