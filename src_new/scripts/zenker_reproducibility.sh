#!/bin/bash

# Hybrid SDE Seed Sweep Script
# Runs each seed with encoder="none" and lr=1e-4 on separate GPUs

echo "Starting Hybrid SDE seed sweep..."
echo "Time: $(date)"

# Seeds to test
SEEDS=(64 42 96)

# Fixed parameters
ENCODER="none"
LR="1e-4"

# Counter for GPU assignment (GPUs 1-7)
gpu_counter=1

# Create logs directory
mkdir -p logs

# Run each seed
for seed in "${SEEDS[@]}"; do
    echo "Launching: LR=${LR}, Encoder=${ENCODER}, Seed=${seed} on GPU ${gpu_counter}"

    # Create informative run name for wandb
    run_name="hybrid_sde_lr${LR}_encoder_${ENCODER}_seed${seed}_gpu${gpu_counter}"

    CUDA_VISIBLE_DEVICES=${gpu_counter} nohup python ./models/hybrid_sde_main.py -m \
        trainer.accelerator=gpu \
        trainer.devices=1 \
        trainer.strategy=auto \
        trainer.precision=bf16-mixed \
        trainer.accumulate_grad_batches=1 \
        trainer.max_epochs=1 \
        model.plot_outputs_train=true \
        model.plot_every=15 \
        data_config.batch_size=64 \
        model.controller_type=mlp \
        model.learning_rate=${LR} \
        seed=${seed} \
        model.use_encoder=${ENCODER} \
        model.num_samples=3 \
        model.loss_type=nll \
        model.log_lik_scale_mode=learnable \
        model.direct_pressure_controls=false \
        model.test_zenker=true \
        run_name="${run_name}" \
        > logs/hybrid_sde_lr${LR}_${ENCODER}_seed${seed}_gpu${gpu_counter}_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    echo "Process PID: $! | Log: logs/hybrid_sde_lr${LR}_${ENCODER}_seed${seed}_gpu${gpu_counter}_*.log"

    # Move to next GPU (cycle through 1-7)
    gpu_counter=$((gpu_counter % 7 + 1))

    # Small delay to avoid startup conflicts
    sleep 3
done

echo ""
echo "=== All 3 runs launched! ==="
echo "LR: 1e-4"
echo "Encoder: none"
echo "Seeds: 64, 42, 96"
echo "Config: Hybrid SDE (max_epochs=1, batch_size=64, num_samples=3, test_zenker=true)"
echo "GPUs: 1-3 (cycling through 1-7)"
echo "Additional configs: bf16-mixed precision, NLL loss, learnable scale, MLP controller"
echo ""
echo "Monitor with:"
echo "  nvidia-smi"
echo "  ps aux | grep hybrid_sde_main"
echo "  tail -f logs/hybrid_sde_lr1e-4_none_seed42_gpu*_*.log"
echo ""
echo "Kill all with:"
echo "  pkill -f hybrid_sde_main"