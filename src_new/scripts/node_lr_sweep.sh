#!/bin/bash

# NODE Learning Rate Sweep Script
# Runs each LR with prior_tx_sigma=0 on separate GPUs

echo "Starting NODE learning rate sweep..."
echo "Time: $(date)"

# Learning rates to test
LRS=("1e-3" "1e-4" "5e-4")

# Counter for GPU assignment (GPUs 1-7)
gpu_counter=1

# Create logs directory
mkdir -p logs

# Run each learning rate
for lr in "${LRS[@]}"; do
    echo "Launching: LR=${lr}, prior_tx_sigma=0 on GPU ${gpu_counter}"

    # Create informative run name for wandb
    run_name="node_lr${lr}_sigma0_gpu${gpu_counter}"

    nohup python ../models/NSDE_main.py \
        trainer.devices=[${gpu_counter}] \
        trainer.strategy=auto \
        trainer.accumulate_grad_batches=1 \
        trainer.max_epochs=25 \
        trainer.precision=bf16-mixed \
        model.learning_rate=${lr} \
        model.prior_tx_sigma=0 \
        model.loss_type=nll \
        model.log_lik_scale_mode=annealing \
        model.controller_type=mlp \
        run_name="${run_name}" \
        data.data_root='../../data/mimic_3_data/processed_data' \
        > logs/node_lr${lr}_sigma0_gpu${gpu_counter}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    echo "Process PID: $! | Log: logs/node_lr${lr}_sigma0_gpu${gpu_counter}_*.log"

    # Move to next GPU (cycle through 1-7)
    gpu_counter=$((gpu_counter % 7 + 1))

    # Small delay to avoid startup conflicts
    sleep 3
done

echo ""
echo "=== All 3 runs launched! ==="
echo "LRs: 1e-3, 1e-4, 5e-4"
echo "Config: prior_tx_sigma=0"
echo "GPUs: 1-3 (cycling through 1-7)"
echo "Additional configs: bf16-mixed precision, NLL loss, annealing scale, MLP controller"
echo ""
echo "Monitor with:"
echo "  nvidia-smi"
echo "  ps aux | grep NSDE_main"
echo "  tail -f logs/node_lr1e-4_sigma0_*.log"
echo ""
echo "Kill all with:"
echo "  pkill -f NSDE_main"