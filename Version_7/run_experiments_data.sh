#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="main_6_new.py"


# Define parameter arrays

gammas=(0 1 10)
confounder_types=('visible' 'partial' 'invisible')
normalises=(True False)
expert_latent_dimss=(4 8 12)

# Nested loops to cover all combinations of parameters
for seed in "${seeds[@]}"; do
    for gamma in "${gammas[@]}"; do
        for confounder_type in "${confounder_types[@]}"; do
            for normalise in "${normalises[@]}"; do
                for expert_latent_dims in "${expert_latent_dimss[@]}"; do
                    echo "Running experiment with seed=$seed, gamma=$gamma, confounder_type=$confounder_type, normalise=$normalise, expert_latent_dims=$expert_latent_dims"
                    python $SCRIPT_PATH \
                        --seed $seed \
                        --gamma $gamma \
                        --confounder_type $confounder_type \
                        --normalise $normalise \
                        --expert_latent_dim $expert_latent_dims
                done
            done
        done
    done
done