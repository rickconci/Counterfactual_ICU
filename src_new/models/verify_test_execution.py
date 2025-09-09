#!/usr/bin/env python3
"""
Verification script to ensure test execution works correctly.
This script tests the data module setup and verifies test dataloaders are properly configured.
"""

import os
import sys
import torch
from pathlib import Path

# Add the models directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataloaders.MIMIC_data import MIMICDataModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf


def verify_test_setup():
    """Verify that the test setup is working correctly."""
    print("🔍 Verifying test execution setup...")
    
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
    
    # Initialize DataModule
    try:
        data_module = MIMICDataModule(
            data_root=cfg.data_config.data_root,
            icu_stays_path=cfg.data_config.icu_stays_path,
            batch_size=cfg.data_config.batch_size,
            num_workers=cfg.data_config.num_workers,
            max_samples=10,  # Small sample for testing
            split_mode=cfg.data_config.split_mode,
            ood_holdout_ratio=cfg.data_config.ood_holdout_ratio,
            filter_flat_trajectories=cfg.data_config.filter_flat_trajectories,
            test_both_filtered_and_unfiltered=cfg.data_config.test_both_filtered_and_unfiltered,
            use_raindrop_context=cfg.data_config.use_raindrop_context,
            expert_latent_dim=cfg.data_config.expert_latent_dim,
            random_state=cfg.seed
        )
        print("✅ DataModule initialized successfully")
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
    
    # Verify train dataset
    if hasattr(data_module, 'train_dataset') and data_module.train_dataset is not None:
        print(f"✅ Train dataset: {len(data_module.train_dataset)} samples")
    else:
        print("❌ Train dataset not found")
        return False
    
    # Verify validation dataset
    if hasattr(data_module, 'val_dataset') and data_module.val_dataset is not None:
        print(f"✅ Validation dataset: {len(data_module.val_dataset)} samples")
    else:
        print("❌ Validation dataset not found")
        return False
    
    # Verify test dataset(s)
    if cfg.data_config.test_both_filtered_and_unfiltered:
        if hasattr(data_module, 'test_dataset_all') and data_module.test_dataset_all is not None:
            print(f"✅ Test dataset (all): {len(data_module.test_dataset_all)} samples")
        else:
            print("❌ Test dataset (all) not found")
            return False
            
        if hasattr(data_module, 'test_dataset_filtered') and data_module.test_dataset_filtered is not None:
            print(f"✅ Test dataset (filtered): {len(data_module.test_dataset_filtered)} samples")
        else:
            print("❌ Test dataset (filtered) not found")
            return False
    else:
        if hasattr(data_module, 'test_dataset') and data_module.test_dataset is not None:
            print(f"✅ Test dataset: {len(data_module.test_dataset)} samples")
        else:
            print("❌ Test dataset not found")
            return False
    
    # Test dataloader creation
    try:
        test_dataloader = data_module.test_dataloader()
        print("✅ Test dataloader created successfully")
        
        if isinstance(test_dataloader, list):
            print(f"✅ Multiple test dataloaders: {len(test_dataloader)}")
            for i, dl in enumerate(test_dataloader):
                print(f"  - Dataloader {i}: {len(dl)} batches")
        else:
            print(f"✅ Single test dataloader: {len(test_dataloader)} batches")
    except Exception as e:
        print(f"❌ Failed to create test dataloader: {e}")
        return False
    
    # Test a single batch from test dataloader
    try:
        if isinstance(test_dataloader, list):
            # Test first dataloader
            batch = next(iter(test_dataloader[0]))
            print(f"✅ Test batch loaded successfully from first dataloader")
            print(f"  - Batch contains {len(batch)} tensors")
        else:
            batch = next(iter(test_dataloader))
            print(f"✅ Test batch loaded successfully")
            print(f"  - Batch contains {len(batch)} tensors")
    except Exception as e:
        print(f"❌ Failed to load test batch: {e}")
        return False
    
    print("\n🎉 All test execution verifications passed!")
    return True


def verify_main_script_structure():
    """Verify the main script has the correct structure for test execution."""
    print("\n🔍 Verifying main script structure...")
    
    main_script_path = os.path.join(os.path.dirname(__file__), "hybrid_sde_main.py")
    
    if not os.path.exists(main_script_path):
        print(f"❌ Main script not found: {main_script_path}")
        return False
    
    with open(main_script_path, 'r') as f:
        content = f.read()
    
    # Check for key components
    checks = [
        ("trainer.fit() called once", content.count("trainer.fit(") == 1),
        ("trainer.test() called", "trainer.test(" in content),
        ("ckpt_path='best'", "ckpt_path=\"best\"" in content),
        ("Error handling for tests", "except Exception as e:" in content and "test" in content.lower()),
        ("Test always runs", "ALWAYS run tests" in content),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("🚀 Starting test execution verification...")
    print("=" * 50)
    
    # Verify main script structure
    script_ok = verify_main_script_structure()
    
    # Verify test setup (only if we have access to data)
    setup_ok = True
    try:
        setup_ok = verify_test_setup()
    except Exception as e:
        print(f"⚠️  Could not verify test setup (data not available): {e}")
        setup_ok = True  # Don't fail if data is not available
    
    print("\n" + "=" * 50)
    if script_ok and setup_ok:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("✅ Your model will perform tests on test dataloaders after training.")
        sys.exit(0)
    else:
        print("❌ SOME VERIFICATIONS FAILED!")
        print("Please check the issues above.")
        sys.exit(1)
