#!/usr/bin/env python3
"""
Quick test to verify the variance-based test setup is working.
"""

import os
import sys
import torch

# Add the models directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.MIMIC_data import MIMICDataModule
import hydra
from omegaconf import DictConfig, OmegaConf


def test_variance_setup():
    """Test the variance-based test setup."""
    print("🔍 Testing variance-based test setup...")
    
    # Load configuration
    try:
        with hydra.initialize(config_path="configs", version_base=None):
            cfg = hydra.compose(config_name="config")
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False
    
    # Check configuration
    print(f"📊 test_both_filtered_and_unfiltered: {cfg.data_config.test_both_filtered_and_unfiltered}")
    print(f"📊 filter_flat_trajectories: {cfg.data_config.filter_flat_trajectories}")
    
    if not cfg.data_config.test_both_filtered_and_unfiltered:
        print("❌ test_both_filtered_and_unfiltered is not enabled!")
        return False
    
    # Initialize DataModule with small sample
    try:
        data_module = MIMICDataModule(
            data_root=cfg.data_config.data_root,
            icu_stays_path=cfg.data_config.icu_stays_path,
            batch_size=2,
            num_workers=0,
            max_samples=10,
            split_mode=cfg.data_config.split_mode,
            ood_holdout_ratio=cfg.data_config.ood_holdout_ratio,
            filter_flat_trajectories=cfg.data_config.filter_flat_trajectories,
            test_both_filtered_and_unfiltered=cfg.data_config.test_both_filtered_and_unfiltered,
            use_raindrop_context=cfg.data_config.use_raindrop_context,
            expert_latent_dim=cfg.data_config.expert_latent_dim,
            random_state=cfg.seed
        )
        print("✅ DataModule initialized")
    except Exception as e:
        print(f"❌ DataModule error: {e}")
        return False
    
    # Setup
    try:
        data_module.setup()
        print("✅ DataModule setup completed")
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False
    
    # Check test datasets
    if not hasattr(data_module, 'test_dataset_all'):
        print("❌ test_dataset_all not found")
        return False
    if not hasattr(data_module, 'test_dataset_filtered'):
        print("❌ test_dataset_filtered not found")
        return False
    
    print(f"✅ Test datasets created:")
    print(f"  - All trajectories: {len(data_module.test_dataset_all)} samples")
    print(f"  - Filtered trajectories: {len(data_module.test_dataset_filtered)} samples")
    
    # Test dataloaders
    try:
        test_dataloaders = data_module.test_dataloader()
        print(f"✅ Test dataloaders: {len(test_dataloaders)} dataloaders")
        
        # Test batch loading
        for i, dataloader in enumerate(test_dataloaders):
            batch = next(iter(dataloader))
            print(f"  - Dataloader {i}: batch size {len(batch[0])}")
            
    except Exception as e:
        print(f"❌ Dataloader error: {e}")
        return False
    
    print("🎉 Variance-based test setup is working correctly!")
    return True


if __name__ == "__main__":
    success = test_variance_setup()
    if success:
        print("\n✅ SUCCESS: Your model will test on both high and low variance data!")
    else:
        print("\n❌ FAILED: Check the issues above.")
    sys.exit(0 if success else 1)
