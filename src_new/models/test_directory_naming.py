#!/usr/bin/env python3
"""
Test script to demonstrate Hydra's informative directory naming.
"""

import os
from datetime import datetime

def demonstrate_directory_naming():
    """Demonstrate what the directory names will look like with Hydra configuration."""
    
    # Simulate the Hydra configuration
    saving_dir = "/n/holylfs06/LABS/mzitnik_lab/Lab/rconci/Counterfactual_ICU/results"
    
    # Example configuration values
    config_examples = [
        {
            "use_encoder": "none",
            "split_mode": "temporal",
            "learning_rate": 0.001,
            "batch_size": 32,
            "max_epochs": 100,
            "seed": 64
        },
        {
            "use_encoder": "mlp",
            "split_mode": "random",
            "learning_rate": 0.0005,
            "batch_size": 64,
            "max_epochs": 200,
            "seed": 42
        },
        {
            "use_encoder": "transformer",
            "split_mode": "temporal",
            "learning_rate": 0.001,
            "batch_size": 16,
            "max_epochs": 150,
            "seed": 123
        }
    ]
    
    print("🎯 Hydra Directory Naming Examples")
    print("=" * 60)
    
    for i, config in enumerate(config_examples, 1):
        # Simulate the Hydra directory naming pattern
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"enc_{config['use_encoder']}_split_{config['split_mode']}_lr{config['learning_rate']}_bs{config['batch_size']}_epochs{config['max_epochs']}_seed{config['seed']}_{timestamp}"
        full_path = os.path.join(saving_dir, dir_name)
        
        print(f"\nExample {i}:")
        print(f"  Config: {config}")
        print(f"  Directory: {dir_name}")
        print(f"  Full path: {full_path}")
    
    print(f"\n📁 Base directory: {saving_dir}")
    print("\n✅ Benefits of this naming scheme:")
    print("  - Easy to identify experiment parameters at a glance")
    print("  - Chronologically ordered with timestamp")
    print("  - No duplicate names (timestamp ensures uniqueness)")
    print("  - Automatically includes command-line overrides")
    print("  - Hydra handles all the complexity for you!")

if __name__ == "__main__":
    demonstrate_directory_naming()
