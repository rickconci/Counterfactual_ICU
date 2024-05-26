#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="main_7.py"

# Define parameter arrays
SDEnet_hidden_dim=(300 400)
SDEnet_depth=(5 6)
gamma=(0 1 5)
non_zero_sigma=(0.3 1)  # Sigma values for self_reverting_prior_control=True

# Loop through each combination of parameters
for hidden_dim in "${SDEnet_hidden_dim[@]}"; do
    for depth in "${SDEnet_depth[@]}"; do
        for sigma in "${non_zero_sigma[@]}"; do
            for g in "${gamma[@]}"; do
                echo "Running experiment with hidden_dim=$hidden_dim, depth=$depth, sigma=$sigma, gamma=$g"
                python $SCRIPT_PATH \
                    --SDEnet_hidden_dim $hidden_dim \
                    --SDEnet_depth $depth \
                    --prior_tx_sigma $sigma \
                    --gamma $g \
                    --self_reverting_prior_control True
            done
        done
    done
done
