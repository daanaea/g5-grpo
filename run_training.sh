#!/bin/bash

# Training script optimized for AWS g5.2xlarge instance
# This script runs the complete training pipeline

set -e

echo "========================================="
echo "Qwen GSM8K Fine-tuning on AWS g5.2xlarge"
echo "========================================="

# Setup CUDA environment
# Find and export CUDA paths
if [ -d "/usr/local/cuda-12.9" ]; then
    export CUDA_HOME=/usr/local/cuda-12.9
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
elif [ -d "/usr/local/cuda-12" ]; then
    export CUDA_HOME=/usr/local/cuda-12
elif [ -d "/usr/local/cuda-11" ]; then
    export CUDA_HOME=/usr/local/cuda-11
fi

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
nvcc --version 2>/dev/null || echo "nvcc not found"

# Check if we're on AWS instance
if [ -f /etc/os-release ]; then
    echo "OS Info:"
    cat /etc/os-release | grep PRETTY_NAME
fi

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Check Python and pip
echo ""
echo "Python version:"
python3 --version || python --version
which pip3 || which pip

# Install dependencies
# Use --ignore-installed to avoid conflicts with system packages
echo ""
echo "Installing dependencies..."
python3 -m pip install --user --no-warn-script-location trl peft bitsandbytes wandb

# Optional: Setup wandb for experiment tracking
# Uncomment and set your API key
# export WANDB_API_KEY="your_wandb_api_key_here"
# export WANDB_PROJECT="qwen-gsm8k"

# Test reward function
echo ""
echo "Testing reward function..."
python3 reward_function.py || python reward_function.py

# Run training
echo ""
echo "Starting training pipeline..."

# Option 1: Quick test run with limited samples (recommended for testing)
# python3 train_qwen_gsm8k.py \
#     --mode both \
#     --model_name Qwen/Qwen3-0.6B \
#     --num_epochs 1 \
#     --batch_size 4 \
#     --use_4bit \
#     --max_samples 100

# Option 2: Full training run with Qwen2.5-Math-0.5B
python3 train_qwen_gsm8k.py \
    --mode both \
    --model_name Qwen/Qwen3-0.6B \
    --num_epochs 3 \
    --batch_size 4 \
    --use_4bit \
    --sft_output ./qwen_gsm8k_sft \
    --rl_output ./qwen_gsm8k_rl

# Evaluate the model
echo ""
echo "Evaluating model..."
python3 train_qwen_gsm8k.py --mode eval --rl_output ./qwen_gsm8k_rl

echo ""
echo "Training complete! Models saved to:"
echo "  - SFT model: ./qwen_gsm8k_sft"
echo "  - RL model: ./qwen_gsm8k_rl"
