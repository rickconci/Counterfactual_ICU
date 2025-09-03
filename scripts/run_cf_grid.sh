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
SDE_WEIGHTS=(1)
LRS=(1e-5 5e-5)
SIGMAS=(0.00001)
NUM_SAMPLES=(3)
SEEDS=(42 64)
CONTROLLER_TYPE=(mlp gat)
# Wider SDEnet variants to reach ~1–5M params (depth=6): 512~1.6M, 640~2.5M, 768~3.6M
SDENET_HDS=(512 768)

REPO_ROOT="/n/holylfs06/LABS/mzitnik_lab/Lab/rconci/Counterfactual_ICU"
PYTHON_BIN="python"

OUT_ROOT="${REPO_ROOT}/experiments"
mkdir -p "${OUT_ROOT}"

run_one_on_gpu() {
  local gpu_id="$1" enc="$2" controller="$3" sdew="$4" lr="$5" sigma="$6" ns="$7" seed="$8" hdim="$9"
  local name="enc-${enc}_ctrl-${controller}_sdew-${sdew}_lr-${lr}_sigma-${sigma}_ns-${ns}_hd-${hdim}_seed-${seed}"
  local train_dir="${OUT_ROOT}/${name}"
  mkdir -p "${train_dir}"

  local cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/src_new/models/hybrid_sde_main.py"
    --dataset_type mimic
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
    --control_energy_weight 0.000001
    --SDEnet_hidden_dim "${hdim}"
    --controller_type "${controller}"
    --redirect_output
    --log_file "${train_dir}/train.log"
  )

  # Log command to stderr so stdout remains clean for PID capture
  echo "[GPU ${gpu_id}] Running: ${cmd[*]}" 1>&2
  # Redirect child stdout/stderr to the job's train.log to prevent PID capture pollution
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" >>"${train_dir}/train.log" 2>&1 &
  pid=$!
  # Print only the PID on stdout
  printf '%s\n' "${pid}"
}

# Simple scheduler: launch tasks up to NUM_GPUS in parallel, then reuse freed GPUs
declare -A GPU_BUSY=()
declare -A GPU_PID=()
for gid in "${GPU_IDS_ARR[@]}"; do GPU_BUSY["$gid"]=0; GPU_PID["$gid"]=""; done

launch_or_wait() {
  local enc="$1" controller="$2" sdew="$3" lr="$4" sigma="$5" ns="$6" seed="$7" hdim="$8"
  while :; do
    for gid in "${GPU_IDS_ARR[@]}"; do
      if [[ "${GPU_BUSY[$gid]}" -eq 0 ]]; then
        # Launch on free GPU
        pid=$(run_one_on_gpu "$gid" "$enc" "$controller" "$sdew" "$lr" "$sigma" "$ns" "$seed" "$hdim")
        # Validate PID is numeric
        if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
          echo "[GPU ${gid}] Warning: Non-numeric PID captured ('${pid}'). Skipping slot reuse and retrying launch..." 1>&2
          sleep 1
          continue
        fi
        GPU_BUSY["$gid"]=1
        GPU_PID["$gid"]="$pid"
        echo "[GPU ${gid}] Launched PID ${pid} for enc=${enc} ctrl=${controller} sdew=${sdew} lr=${lr} sigma=${sigma} ns=${ns} hd=${hdim} seed=${seed}"
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
  for controller in "${CONTROLLER_TYPE[@]}"; do
    for sdew in "${SDE_WEIGHTS[@]}"; do
      for lr in "${LRS[@]}"; do
        for sigma in "${SIGMAS[@]}"; do
          for ns in "${NUM_SAMPLES[@]}"; do
            for hdim in "${SDENET_HDS[@]}"; do
              for seed in "${SEEDS[@]}"; do
                launch_or_wait "$enc" "$controller" "$sdew" "$lr" "$sigma" "$ns" "$seed" "$hdim"
              done
            done
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


