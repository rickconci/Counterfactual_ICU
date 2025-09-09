#!/bin/bash

# NSDE Learning Rate Sweep Script
# Runs each encoder combination with lr=1e-4 on separate GPUs

echo "Starting NSDE learning rate sweep..."
echo "Time: $(date)"

# Encoders and seeds to test
ENCODERS=("none" "raindrop")
SEEDS=(64 42 96)

# Fixed learning rate
LR="1e-4"

# Counter for GPU assignment (GPUs 1-7)
gpu_counter=1

# Create logs directory
mkdir -p logs

# Run each combination
for encoder in "${ENCODERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Launching: LR=${LR}, Encoder=${encoder}, Seed=${seed}, prior_tx_sigma=0.3 on GPU ${gpu_counter}"

        # Create informative run name for wandb
        run_name="nsde_lr${LR}_encoder_${encoder}_sigma0.3_seed${seed}_gpu${gpu_counter}"

        CUDA_VISIBLE_DEVICES=${gpu_counter} nohup python ../models/NSDE_main.py \
            trainer.devices=1 \
            trainer.strategy=auto \
            trainer.accumulate_grad_batches=1 \
            trainer.max_epochs=50 \
            trainer.precision=bf16-mixed \
            model.learning_rate=${LR} \
            model.use_encoder=${encoder} \
            model.prior_tx_sigma=0.3 \
            model.loss_type=nll \
            model.log_lik_scale_mode=annealing \
            model.controller_type=mlp \
            seed=${seed} \
            run_name="${run_name}" \
            data.data_root='../../data/mimic_3_data/processed_data' \
            data.test_both_filtered_and_unfiltered=true \
            data.num_workers=4 \
            > logs/nsde_lr${LR}_${encoder}_sigma0.3_seed${seed}_gpu${gpu_counter}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

        echo "Process PID: $! | Log: logs/nsde_lr${LR}_${encoder}_sigma0.3_seed${seed}_gpu${gpu_counter}_*.log"

        # Move to next GPU (cycle through 1-7)
        gpu_counter=$((gpu_counter % 7 + 1))

        # Small delay to avoid startup conflicts
        sleep 3
    done
done

echo ""
echo "=== All 6 runs launched! ==="
echo "LR: 1e-4"
echo "Encoders: none, raindrop"
echo "Seeds: 64, 42, 96"
echo "Config: NSDE (prior_tx_sigma=0.3), test_both_filtered_and_unfiltered=true, num_workers=4"
echo "GPUs: 1-6 (cycling through 1-7)"
echo "Additional configs: 50 epochs, bf16-mixed precision, NLL loss, annealing scale, MLP controller"
echo ""
echo "Monitor with:"
echo "  nvidia-smi"
echo "  ps aux | grep NSDE_main"
echo "  tail -f logs/nsde_lr1e-4_raindrop_sigma0.3_seed42_gpu*_*.log"
echo ""
echo "Kill all with:"
echo "  pkill -f NSDE_main"