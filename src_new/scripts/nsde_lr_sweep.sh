#!/bin/bash

# NSDE Learning Rate Sweep Script
# Runs each combination on a different GPU (1-7)

echo "Starting NSDE learning rate sweep..."
echo "Time: $(date)"

# Learning rates to test
LRS=("1e-3" "1e-4" "5e-4")
ENCODERS=("none" "raindrop")

# Counter for GPU assignment
gpu_counter=1

# Run each combination
for lr in "${LRS[@]}"; do
    for encoder in "${ENCODERS[@]}"; do
        echo "Launching: LR=${lr}, Encoder=${encoder} on GPU ${gpu_counter}"

        nohup python ../models/NSDE_main.py \
            trainer.devices=[${gpu_counter}] \
            model.learning_rate=${lr} \
            model.use_encoder=${encoder} \
            data.data_root='../../data/mimic_3_data/processed_data' \
            > logs/nsde_lr${lr}_${encoder}_gpu${gpu_counter}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

        echo "Process started with PID: $!"

        # Move to next GPU (cycle through 1-7)
        gpu_counter=$((gpu_counter % 7 + 1))

        # Small delay to avoid startup conflicts
        sleep 2
    done
done

echo "All runs launched!"
echo "Monitor with: nvidia-smi"
echo "Check logs in: logs/ directory"
echo "Kill all with: pkill -f NSDE_main"