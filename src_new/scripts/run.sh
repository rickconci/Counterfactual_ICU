#!/bin/bash

# SDE Cardiovascular Model Training Script
# Usage: ./train_sde.sh

set -e  # Exit on any error

echo "Starting SDE Cardiovascular Model Training..."
echo "Timestamp: $(date)"
echo "Environment: $CONDA_DEFAULT_ENV"


# Check if required packages are available
echo "Checking dependencies..."
python -c "import torch, lightning, torchsde, torch_geometric; print('All dependencies OK')"

# Create logs directory
mkdir -p ../../results/logs
mkdir -p ../../results/model_checkpoints

# Training command
echo "Starting training with nohup..."
python ../models/hybrid_sde_main.py \
    --dataset_type mimic \
    --use_encoder none \
    --num_samples 1 \
    --batch_size 32 \
    --seed 14 \
    --max_epochs 20 \
    --data_root '../../data/mimic_3_data/processed_data' \
    --log_wandb True \
    --model_checkpoint True \
    --early_stopping True \
    --run_eval \
    > ../../results/logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Get the process ID
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo "Log file: ../../results/logs/training_$(date +%Y%m%d_%H%M%S).log"

# Save PID for later reference
echo $TRAIN_PID > ../../results/logs/train_pid.txt
echo "PID saved to ../../results/logs/train_pid.txt"

echo ""
echo "=== MONITORING COMMANDS ==="
echo "View live log:    tail -f ../../results/logs/training_$(date +%Y%m%d_%H%M%S).log"
echo "Check if running: ps aux | grep $TRAIN_PID"
echo "Kill training:    kill $TRAIN_PID"
echo "Kill all python:  pkill -f main_beta.py"

echo ""
echo "Training is now running in background!"
echo "You can safely disconnect from SSH."