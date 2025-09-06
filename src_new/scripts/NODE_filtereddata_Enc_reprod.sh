#!/bin/bash

# Multi-GPU SDE Cardiovascular Model Training Script
# Usage: ./train_sde_multi.sh
# Runs 3 parallel jobs with different seeds on GPUs 5-7

set -e  # Exit on any error

echo "Starting Multi-GPU SDE Cardiovascular Model Training..."
echo "Timestamp: $(date)"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "GPUs: 5, 6, 7"
echo "Seeds: 14, 42, 96"

# Check if required packages are available
echo "Checking dependencies..."
python -c "import torch, lightning, torchsde, torch_geometric; print('All dependencies OK')"

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | grep -E "^[5-7]," || echo "Warning: GPUs 5-7 status unknown"

# Create logs directory
mkdir -p ../../results/logs
mkdir -p ../../results/model_checkpoints

# Generate timestamp for unique file names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Array of seeds and corresponding GPUs
declare -a seeds=("14" "42" "96")
declare -a gpu_ids=("5" "6" "7")
declare -a run_names=("sde_seed14" "sde_seed42" "sde_seed96")

# Array to store PIDs
declare -a pids=()

echo ""
echo "=== Starting Training Jobs ==="

# Launch 3 parallel training jobs
for i in {0..2}; do
    seed=${seeds[$i]}
    gpu=${gpu_ids[$i]}
    run_name=${run_names[$i]}

    echo "Starting job $((i+1))/3: Seed=${seed}, GPU=${gpu}, Run=${run_name}"

    # Set GPU and run training
    CUDA_VISIBLE_DEVICES=${gpu} nohup python ../models/NODE_main.py \
        --dataset_type mimic \
        --use_encoder raindrop \
        --num_samples 1 \
        --batch_size 32 \
        --seed ${seed} \
        --max_epochs 100 \
        --data_root '../../data/mimic_3_data/processed_data' \
        --log_wandb \
        --model_checkpoint \
        --early_stopping \
        --integration_step_size 2 \
        --HPC_work \
        --output_scale 3 \
        --run_eval \
        --learning_rate 0.0005 \
        --early_stopping_patience 10 \
        > ../../results/logs/training_${run_name}_${TIMESTAMP}.log 2>&1 &

    # Store the PID
    pid=$!
    pids+=($pid)
    echo "  └─ Started with PID: $pid"
    echo "  └─ Log: ../../results/logs/training_${run_name}_${TIMESTAMP}.log"

    # Small delay to prevent resource conflicts during startup
    sleep 2
done

# Save all PIDs
echo ""
echo "=== Saving Process Information ==="
pid_file="../../results/logs/train_pids_${TIMESTAMP}.txt"
for i in {0..2}; do
    echo "${seeds[$i]} ${gpu_ids[$i]} ${pids[$i]} ${run_names[$i]}" >> $pid_file
done
echo "PIDs saved to: $pid_file"

echo ""
echo "=== MONITORING COMMANDS ==="
echo "View all processes:   ps aux | grep NODE_main.py"
echo "Check GPU usage:      nvidia-smi"
echo "View logs:"
for i in {0..2}; do
    echo "  Seed ${seeds[$i]} (GPU ${gpu_ids[$i]}): tail -f ../../results/logs/training_${run_names[$i]}_${TIMESTAMP}.log"
done

echo ""
echo "Kill specific job:"
for i in {0..2}; do
    echo "  Seed ${seeds[$i]} (PID ${pids[$i]}): kill ${pids[$i]}"
done

echo ""
echo "Kill all jobs:        kill ${pids[@]}"
echo "Kill all NODE_main:   pkill -f NODE_main.py"

echo ""
echo "=== SUMMARY ==="
echo "3 training jobs are now running in background!"
echo "Seeds: ${seeds[@]}"
echo "GPUs: ${gpu_ids[@]}"
echo "PIDs: ${pids[@]}"
echo "You can safely disconnect from SSH."

# Optional: Wait a few seconds and check if all jobs are still running
sleep 5
echo ""
echo "=== STATUS CHECK ==="
for i in {0..2}; do
    if ps -p ${pids[$i]} > /dev/null 2>&1; then
        echo "✓ Job $((i+1)) (Seed=${seeds[$i]}, GPU=${gpu_ids[$i]}) is running"
    else
        echo "✗ Job $((i+1)) (Seed=${seeds[$i]}, GPU=${gpu_ids[$i]}) failed to start"
    fi
done

echo ""
echo "Training jobs launched successfully!"