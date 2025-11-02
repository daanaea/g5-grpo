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
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:$LD_LIBRARY_PATH

# Also check for alternative library locations
if [ -d "$CUDA_HOME/lib" ]; then
    export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
fi

echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""
echo "Checking CUDA installation:"
nvcc --version 2>/dev/null || echo "nvcc not found"
ls -la $CUDA_HOME/lib64/libcusparseLt.so* 2>/dev/null || echo "libcusparseLt.so not found in $CUDA_HOME/lib64"

# Test if PyTorch can find CUDA
echo ""
echo "Testing PyTorch CUDA detection:"
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>&1 || echo "PyTorch import failed - will attempt to continue"

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

# Check if PyTorch can import, if not, reinstall with bundled CUDA
if ! python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch import failed. Reinstalling PyTorch with bundled CUDA 12.1..."
    python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    python3 -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

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
