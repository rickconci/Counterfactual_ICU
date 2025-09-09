#!/bin/bash

# NODE Seed Sweep Script
# Runs encoder-specific LRs with different seeds and prior_tx_sigma=0

echo "Starting NODE seed sweep..."
echo "Time: $(date)"

# Seeds to test
SEEDS=(64 42 96)

# Counter for GPU assignment (GPUs 1-7)
gpu_counter=1

# Create logs directory
mkdir -p logs

# Run encoder "none" with lr=1e-4
encoder="none"
lr="1e-4"
for seed in "${SEEDS[@]}"; do
    echo "Launching: Encoder=${encoder}, LR=${lr}, Seed=${seed}, prior_tx_sigma=0 on GPU ${gpu_counter}"

    # Create informative run name for wandb
    run_name="node_${encoder}_lr${lr}_seed${seed}_sigma0_gpu${gpu_counter}"

    CUDA_VISIBLE_DEVICES=${gpu_counter} nohup python ../models/NSDE_main.py \
        trainer.devices=1 \
        trainer.strategy=auto \
        trainer.accumulate_grad_batches=1 \
        trainer.max_epochs=50 \
        trainer.precision=bf16-mixed \
        model.learning_rate=${lr} \
        model.use_encoder=${encoder} \
        model.prior_tx_sigma=0 \
        model.loss_type=nll \
        model.log_lik_scale_mode=annealing \
        model.controller_type=mlp \
        +seed_everything=${seed}\
        run_name="${run_name}" \
        data.data_root='../../data/mimic_3_data/processed_data' \
        data.test_both_filtered_and_unfiltered=true \
        data.num_workers=4 \
        > logs/node_${encoder}_lr${lr}_seed${seed}_gpu${gpu_counter}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    echo "Process PID: $! | Log: logs/node_${encoder}_lr${lr}_seed${seed}_gpu${gpu_counter}_*.log"

    # Move to next GPU (cycle through 1-7)
    gpu_counter=$((gpu_counter % 7 + 1))

    # Small delay to avoid startup conflicts
    sleep 3
done

# Run encoder "raindrop" with lr=1e-5
encoder="raindrop"
lr="5e-4"
for seed in "${SEEDS[@]}"; do
    echo "Launching: Encoder=${encoder}, LR=${lr}, Seed=${seed}, prior_tx_sigma=0 on GPU ${gpu_counter}"

    # Create informative run name for wandb
    run_name="node_${encoder}_lr${lr}_seed${seed}_sigma0_gpu${gpu_counter}"

    CUDA_VISIBLE_DEVICES=${gpu_counter} nohup python ../models/NSDE_main.py \
        trainer.devices=1 \
        trainer.strategy=auto \
        trainer.accumulate_grad_batches=1 \
        trainer.max_epochs=50 \
        trainer.precision=bf16-mixed \
        model.learning_rate=${lr} \
        model.use_encoder=${encoder} \
        model.prior_tx_sigma=0 \
        model.loss_type=nll \
        model.log_lik_scale_mode=annealing \
        model.controller_type=mlp \
        seed_everything=${seed} \
        run_name="${run_name}" \
        data.data_root='../../data/mimic_3_data/processed_data' \
        data.test_both_filtered_and_unfiltered=true \
        data.num_workers=4 \
        > logs/node_${encoder}_lr${lr}_seed${seed}_gpu${gpu_counter}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    echo "Process PID: $! | Log: logs/node_${encoder}_lr${lr}_seed${seed}_gpu${gpu_counter}_*.log"

    # Move to next GPU (cycle through 1-7)
    gpu_counter=$((gpu_counter % 7 + 1))

    # Small delay to avoid startup conflicts
    sleep 3
done

echo ""
echo "=== All 6 runs launched! ==="
echo "Encoder 'none': LR=1e-4, Seeds=64,42,96"
echo "Encoder 'raindrop': LR=1e-5, Seeds=64,42,96"
echo "Config: prior_tx_sigma=0, num_workers=4"
echo "GPUs: 1-6 (cycling through 1-7)"
echo "Additional configs: 100 epochs, bf16-mixed precision, NLL loss, annealing scale, MLP controller"
echo ""
echo "Monitor with:"
echo "  nvidia-smi"
echo "  ps aux | grep NSDE_main"
echo "  tail -f logs/node_raindrop_lr1e-5_seed42_gpu*_*.log"
echo ""
echo "Kill all with:"
echo "  pkill -f NSDE_main"