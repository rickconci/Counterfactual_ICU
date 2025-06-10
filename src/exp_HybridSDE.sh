#!/bin/bash

SCRIPT_PATH="main_beta.py"

gamma=(0 2 6)
prior_tx_sigma=(0 0.01 0.05)

for seed in 28 11 96; do
    for g in "${gamma[@]}"; do
        echo "Running experiment with seed=$seed, gamma=$g"
        python $SCRIPT_PATH \
            --gamma $g \
            --seed $seed 
    done
done
