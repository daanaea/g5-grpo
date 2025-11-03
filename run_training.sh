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

python3 -m pip install --user --no-warn-script-location trl peft bitsandbytes

# Test reward function
echo ""
echo "Testing reward function..."
python3 reward_function.py || python reward_function.py

# Run training
echo ""
echo "Starting training pipeline..."

# Option 1: Quick test run with limited samples (recommended for testing)
# python3 train.py \
#     --mode both \
#     --model_name Qwen/Qwen3-4B \
#     --max_samples 100 \
#     --num_epochs 1 \
#     --sft_output ./qwen_gsm8k_sft \
#     --grpo_output ./qwen_gsm8k_grpo

# Option 2: Full SFT + GRPO training (recommended)
python3 train.py \
    --mode both \
    --model_name Qwen/Qwen3-4B \
    --num_epochs 3 \
    --batch_size 4 \
    --sft_output ./qwen_gsm8k_sft \
    --grpo_output ./qwen_gsm8k_grpo

# Option 3: SFT only (if you only want the supervised fine-tuned model)
# python3 train.py \
#     --mode sft \
#     --model_name Qwen/Qwen3-4B \
#     --num_epochs 3 \
#     --batch_size 4 \
#     --sft_output ./qwen_gsm8k_sft

# Option 4: GRPO only (requires existing SFT model)
# python3 train.py \
#     --mode grpo \
#     --sft_output ./qwen_gsm8k_sft \
#     --grpo_output ./qwen_gsm8k_grpo

# Evaluate the model
echo ""
echo "Evaluating model..."
python3 train.py --mode eval --grpo_output ./qwen_gsm8k_grpo

echo ""
echo "Training complete!"
echo "  - SFT model: ./qwen_gsm8k_sft"
echo "  - GRPO model: ./qwen_gsm8k_grpo"
echo "  - Training logs: ./qwen_gsm8k_grpo/grpo_training_logs.jsonl"
