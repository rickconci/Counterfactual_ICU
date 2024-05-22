#!/bin/bash

# Define the path to your Python script
SCRIPT_PATH="main_6_new.py"

# Define parameter arrays
gamma=(0 1 5)
use_encoder=('full')
normalise_for_SDENN=(True False)
prior_tx_sigma=(0 0.1 0.1)
self_reverting_prior_control=(True False)

# Nested loops to cover all combinations of parameters
for encoder in "${use_encoder[@]}"; do
    for norm in "${normalise_for_SDENN[@]}"; do
        for sigma in "${prior_tx_sigma[@]}"; do
            for g in "${gamma[@]}"; do
                if [ "$sigma" != "0" ]; then
                    for self_revert in "${self_reverting_prior_control[@]}"; do
                        echo "Running experiment with encoder=$encoder, normalise=$norm, sigma=$sigma, gamma=$g, self_revert=$self_revert"
                        python $SCRIPT_PATH \
                            --use_encoder $encoder \
                            --normalise_for_SDENN $norm \
                            --prior_tx_sigma $sigma \
                            --gamma $g \
                            --self_reverting_prior_control $self_revert
                    done
                else
                    echo "Running experiment with encoder=$encoder, normalise=$norm, sigma=$sigma, gamma=$g, self_revert=False"
                    python $SCRIPT_PATH \
                        --use_encoder $encoder \
                        --normalise_for_SDENN $norm \
                        --prior_tx_sigma $sigma \
                        --gamma $g \
                        --self_reverting_prior_control False
                fi
            done
        done
    done
done