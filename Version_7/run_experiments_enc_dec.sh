#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="main_6_new.py"

# Define parameter arrays
gamma=(0 1 5)
use_encoder=('full')
normalise_for_SDENN=(True False)
non_zero_sigma=(0.3 1)  # Sigma values for self_reverting_prior_control=True
all_sigma=(0 0.3 1)     # All Sigma values for self_reverting_prior_control=False

# Loop for self_reverting_prior_control=True, sigma is non-zero
for encoder in "${use_encoder[@]}"; do
    for norm in "${normalise_for_SDENN[@]}"; do
        for sigma in "${non_zero_sigma[@]}"; do
            for g in "${gamma[@]}"; do
                echo "Running experiment with encoder=$encoder, normalise=$norm, sigma=$sigma, gamma=$g, self_revert=True"
                python $SCRIPT_PATH \
                    --use_encoder $encoder \
                    --normalise_for_SDENN $norm \
                    --prior_tx_sigma $sigma \
                    --gamma $g \
                    --self_reverting_prior_control True
            done
        done
    done
done

# Loop for self_reverting_prior_control=False, sigma can be any value
for encoder in "${use_encoder[@]}"; do
    for norm in "${normalise_for_SDENN[@]}"; do
        for sigma in "${all_sigma[@]}"; do
            for g in "${gamma[@]}"; do
                echo "Running experiment with encoder=$encoder, normalise=$norm, sigma=$sigma, gamma=$g, self_revert=False"
                python $SCRIPT_PATH \
                    --use_encoder $encoder \
                    --normalise_for_SDENN $norm \
                    --prior_tx_sigma $sigma \
                    --gamma $g \
                    --self_reverting_prior_control False
            done
        done
    done
done