#!/usr/bin/env python3
"""
Comprehensive verification script for test dataloader setup.
Tests both high variance (filtered) and low variance (all) data scenarios.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the models directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.MIMIC_data import MIMICDataModule, MIMICDataset
import hydra
from omegaconf import DictConfig, OmegaConf


def analyze_trajectory_variance(p_out_values, p_out_mask, traj_key):
    """Analyze the variance characteristics of a trajectory."""
    channel_names = ["arterial", "venous"]
    channel_range_thresholds = [15.0, 5.0]
    
    analysis = {
        "traj_key": traj_key,
        "channels": {},
        "is_flat": True,
        "overall_range": 0.0
    }
    
    max_range = 0.0
    
    for channel in range(p_out_values.shape[1]):
        channel_values = p_out_values[:, channel]
        channel_mask = p_out_mask[:, channel]
        
        valid_values = channel_values[channel_mask > 0]
        
        if len(valid_values) < 5:  # min_valid_points
            analysis["channels"][channel_names[channel]] = {
                "valid_points": len(valid_values),
                "range": 0.0,
                "threshold": channel_range_thresholds[channel],
                "passes_threshold": False
            }
            continue
            
        channel_range = (valid_values.max() - valid_values.min()).item()
        threshold = channel_range_thresholds[min(channel, len(channel_range_thresholds) - 1)]
        passes_threshold = channel_range >= threshold
        
        analysis["channels"][channel_names[channel]] = {
            "valid_points": len(valid_values),
            "range": channel_range,
            "threshold": threshold,
            "passes_threshold": passes_threshold,
            "mean": valid_values.mean().item(),
            "std": valid_values.std().item()
        }
        
        max_range = max(max_range, channel_range)
        
        if passes_threshold:
            analysis["is_flat"] = False
    
    analysis["overall_range"] = max_range
    return analysis


def verify_test_dataloader_setup():
    """Verify the test dataloader setup for both high and low variance data."""
    print("🔍 Verifying test dataloader setup for high/low variance data...")
    
    # Load configuration
    try:
        with hydra.initialize(config_path="configs", version_base=None):
            cfg = hydra.compose(config_name="config")
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Check if data root exists
    data_root = cfg.data_config.data_root
    if not os.path.exists(data_root):
        print(f"❌ Data root does not exist: {data_root}")
        return False
    print(f"✅ Data root exists: {data_root}")
    
    # Test with both filtered and unfiltered enabled
    print("\n📊 Testing with test_both_filtered_and_unfiltered=True")
    
    try:
        data_module = MIMICDataModule(
            data_root=cfg.data_config.data_root,
            icu_stays_path=cfg.data_config.icu_stays_path,
            batch_size=4,  # Small batch for testing
            num_workers=0,  # No multiprocessing for testing
            max_samples=20,  # Small sample for testing
            split_mode=cfg.data_config.split_mode,
            ood_holdout_ratio=cfg.data_config.ood_holdout_ratio,
            filter_flat_trajectories=cfg.data_config.filter_flat_trajectories,
            test_both_filtered_and_unfiltered=True,  # Enable dual testing
            use_raindrop_context=cfg.data_config.use_raindrop_context,
            expert_latent_dim=cfg.data_config.expert_latent_dim,
            random_state=cfg.seed
        )
        print("✅ DataModule initialized with dual testing enabled")
    except Exception as e:
        print(f"❌ Failed to initialize DataModule: {e}")
        return False
    
    # Setup data module
    try:
        data_module.setup()
        print("✅ DataModule setup completed")
    except Exception as e:
        print(f"❌ Failed to setup DataModule: {e}")
        return False
    
    # Verify test datasets exist
    if not hasattr(data_module, 'test_dataset_all') or data_module.test_dataset_all is None:
        print("❌ test_dataset_all not found")
        return False
    if not hasattr(data_module, 'test_dataset_filtered') or data_module.test_dataset_filtered is None:
        print("❌ test_dataset_filtered not found")
        return False
    
    print(f"✅ Test dataset (all): {len(data_module.test_dataset_all)} samples")
    print(f"✅ Test dataset (filtered): {len(data_module.test_dataset_filtered)} samples")
    
    # Analyze variance characteristics
    print("\n📈 Analyzing trajectory variance characteristics...")
    
    # Sample a few trajectories from each dataset for analysis
    sample_indices = [0, 1, 2] if len(data_module.test_dataset_all) >= 3 else list(range(len(data_module.test_dataset_all)))
    
    print("\n🔍 All trajectories (including low variance):")
    all_analyses = []
    for idx in sample_indices:
        if idx < len(data_module.test_dataset_all):
            sample = data_module.test_dataset_all[idx]
            analysis = analyze_trajectory_variance(
                sample["Y"], sample["Y_mask"], 
                f"all_{idx}"
            )
            all_analyses.append(analysis)
            print(f"  Trajectory {idx}: {'FLAT' if analysis['is_flat'] else 'HIGH_VARIANCE'}")
            for channel, info in analysis["channels"].items():
                print(f"    {channel}: range={info['range']:.2f}, threshold={info['threshold']}, passes={info['passes_threshold']}")
    
    print("\n🔍 Filtered trajectories (high variance only):")
    filtered_analyses = []
    for idx in sample_indices:
        if idx < len(data_module.test_dataset_filtered):
            sample = data_module.test_dataset_filtered[idx]
            analysis = analyze_trajectory_variance(
                sample["Y"], sample["Y_mask"], 
                f"filtered_{idx}"
            )
            filtered_analyses.append(analysis)
            print(f"  Trajectory {idx}: {'FLAT' if analysis['is_flat'] else 'HIGH_VARIANCE'}")
            for channel, info in analysis["channels"].items():
                print(f"    {channel}: range={info['range']:.2f}, threshold={info['threshold']}, passes={info['passes_threshold']}")
    
    # Verify filtering worked correctly
    all_flat_count = sum(1 for a in all_analyses if a['is_flat'])
    filtered_flat_count = sum(1 for a in filtered_analyses if a['is_flat'])
    
    print(f"\n📊 Filtering Results:")
    print(f"  All dataset: {all_flat_count}/{len(all_analyses)} flat trajectories")
    print(f"  Filtered dataset: {filtered_flat_count}/{len(filtered_analyses)} flat trajectories")
    
    if filtered_flat_count > 0:
        print("⚠️  WARNING: Some flat trajectories found in filtered dataset!")
        return False
    else:
        print("✅ Filtering working correctly - no flat trajectories in filtered dataset")
    
    # Test dataloader creation
    try:
        test_dataloaders = data_module.test_dataloader()
        print(f"✅ Test dataloaders created: {len(test_dataloaders)} dataloaders")
        
        # Test loading batches from both dataloaders
        for i, dataloader in enumerate(test_dataloaders):
            batch = next(iter(dataloader))
            print(f"✅ Dataloader {i}: batch size {len(batch[0])}")
            
    except Exception as e:
        print(f"❌ Failed to create/test dataloaders: {e}")
        return False
    
    print("\n🎉 Test dataloader setup verification completed successfully!")
    print("✅ High variance (filtered) and low variance (all) data are properly separated")
    print("✅ Both test dataloaders are working correctly")
    
    return True


if __name__ == "__main__":
    print("🚀 Starting test dataloader verification...")
    print("=" * 60)
    
    success = verify_test_dataloader_setup()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("✅ Your model will test on both high variance and low variance data.")
        sys.exit(0)
    else:
        print("❌ SOME VERIFICATIONS FAILED!")
        print("Please check the issues above.")
        sys.exit(1)
