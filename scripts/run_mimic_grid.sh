#!/usr/bin/env bash
set -euo pipefail

# Grid search for Hybrid SDE (MIMIC). Adjust arrays below as needed.
# Fixed params per request
BS=32
PLOT_EVERY=10
OUTPUT_SCALE=2
DT=2
INTEGRATION_ADAPTIVE=False
LOG_WANDB=True

# GPU scheduling
# Option A: provide explicit GPU IDs via env var GPU_IDS (e.g., "0 1 2 3")
# Option B: auto-detect via nvidia-smi; fallback to [0]
if [[ -n "${GPU_IDS:-}" ]]; then
  read -r -a GPU_IDS_ARR <<< "${GPU_IDS}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_IDS_ARR < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  else
    GPU_IDS_ARR=(0)
  fi
fi
NUM_GPUS=${#GPU_IDS_ARR[@]}
echo "Detected/Using GPUs: ${GPU_IDS_ARR[*]} (count=${NUM_GPUS})"

# Variable params (ordered by importance)
ENCODERS=(none raindrop)
SDE_WEIGHTS=(0.5 1 2)
LRS=(0.0001 0.0005 0.001)
SIGMAS=(0.00001 0.0001 0.001)
NUM_SAMPLES=(1 3)
SEEDS=(42 25 96)

REPO_ROOT="/n/holylfs06/LABS/mzitnik_lab/Lab/rconci/Counterfactual_ICU"
PYTHON_BIN="python"

OUT_ROOT="${REPO_ROOT}/experiments"
mkdir -p "${OUT_ROOT}"

run_one_on_gpu() {
  local gpu_id="$1" enc="$2" sdew="$3" lr="$4" sigma="$5" ns="$6" seed="$7"
  local name="enc-${enc}_sdew-${sdew}_lr-${lr}_sigma-${sigma}_ns-${ns}_seed-${seed}"
  local train_dir="${OUT_ROOT}/${name}"
  mkdir -p "${train_dir}"

  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/src_new/models/hybrid_sde_main.py"
    --dataset_type mimic
    --precision 16-mixed
    --log_every_n_steps 4
    --use_encoder "${enc}"
    --SDE_control_weighting "${sdew}"
    --learning_rate "${lr}"
    --prior_tx_sigma "${sigma}"
    --num_samples "${ns}"
    --seed "${seed}"
    --batch_size "${BS}"
    --plot_outputs_train
    --plot_every "${PLOT_EVERY}"
    --log_wandb "${LOG_WANDB}"
    --output_scale "${OUTPUT_SCALE}"
    --integration_step_size "${DT}"
    --integration_adaptive "${INTEGRATION_ADAPTIVE}"
    --train_dir "${train_dir}"
    --control_energy_weight 0.0
    --redirect_output
    --log_file "${train_dir}/train.log"
  )

  echo "[GPU ${gpu_id}] Running: ${cmd[*]}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" &
  echo $!
}

# Simple scheduler: launch tasks up to NUM_GPUS in parallel, then reuse freed GPUs
declare -A GPU_BUSY=()
declare -A GPU_PID=()
for gid in "${GPU_IDS_ARR[@]}"; do GPU_BUSY["$gid"]=0; GPU_PID["$gid"]=""; done

launch_or_wait() {
  local enc="$1" sdew="$2" lr="$3" sigma="$4" ns="$5" seed="$6"
  while :; do
    for gid in "${GPU_IDS_ARR[@]}"; do
      if [[ "${GPU_BUSY[$gid]}" -eq 0 ]]; then
        # Launch on free GPU
        pid=$(run_one_on_gpu "$gid" "$enc" "$sdew" "$lr" "$sigma" "$ns" "$seed")
        GPU_BUSY["$gid"]=1
        GPU_PID["$gid"]="$pid"
        echo "[GPU ${gid}] Launched PID ${pid} for enc=${enc} sdew=${sdew} lr=${lr} sigma=${sigma} ns=${ns} seed=${seed}"
        return 0
      fi
    done
    # Wait for any PID to finish and free its GPU
    if wait -n 2>/dev/null; then :; fi
    for gid in "${GPU_IDS_ARR[@]}"; do
      if [[ -n "${GPU_PID[$gid]}" ]]; then
        if ! kill -0 "${GPU_PID[$gid]}" 2>/dev/null; then
          echo "[GPU ${gid}] PID ${GPU_PID[$gid]} finished. Marking GPU free."
          GPU_BUSY["$gid"]=0
          GPU_PID["$gid"]=""
        fi
      fi
    done
    sleep 1
  done
}

# Iterate in the requested importance order
for enc in "${ENCODERS[@]}"; do
  for sdew in "${SDE_WEIGHTS[@]}"; do
    for lr in "${LRS[@]}"; do
      for sigma in "${SIGMAS[@]}"; do
        for ns in "${NUM_SAMPLES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            launch_or_wait "$enc" "$sdew" "$lr" "$sigma" "$ns" "$seed"
          done
        done
      done
    done
  done
done

# Final wait for all jobs
echo "Waiting for all jobs to complete..."
wait
echo "All jobs completed."


