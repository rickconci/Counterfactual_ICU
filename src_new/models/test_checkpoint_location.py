#!/usr/bin/env python3
"""
Test script to verify checkpoint directory structure will be created correctly.
"""

import os
from datetime import datetime

def test_checkpoint_directory():
    """Test the checkpoint directory creation logic."""
    
    # Simulate the directory creation logic from the main script
    train_dir_final = "/n/holylfs06/LABS/mzitnik_lab/Lab/rconci/Counterfactual_ICU/results"
    
    # Create the directory if it doesn't exist
    os.makedirs(train_dir_final, exist_ok=True)
    
    # Create a timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(train_dir_final, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Create checkpoints subdirectory
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("✅ Checkpoint directory structure created successfully!")
    print(f"📁 Base directory: {train_dir_final}")
    print(f"📁 Run directory: {run_dir}")
    print(f"📁 Checkpoints directory: {checkpoint_dir}")
    
    # Verify directories exist
    assert os.path.exists(train_dir_final), f"Base directory not created: {train_dir_final}"
    assert os.path.exists(run_dir), f"Run directory not created: {run_dir}"
    assert os.path.exists(checkpoint_dir), f"Checkpoints directory not created: {checkpoint_dir}"
    
    print("🎉 All directories verified and ready for checkpoint saving!")

if __name__ == "__main__":
    test_checkpoint_directory()
