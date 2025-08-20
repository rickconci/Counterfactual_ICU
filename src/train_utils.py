# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import random
from datetime import datetime


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEBUG = True  # Set to True to enable debug printing

def debug_print(*args, **kwargs):
    """Print only if DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)

def toggle_debug(enable=None):
    """Toggle or set the DEBUG flag
    
    Args:
        enable: If None, toggle the current state. If True/False, set to that value.
    
    Returns:
        Current DEBUG value after toggling/setting
    """
    global DEBUG
    if enable is None:
        DEBUG = not DEBUG
    else:
        DEBUG = bool(enable)
    return DEBUG


def get_lightning_devices(devices_arg=None):
    """
    Automatically detect available devices for PyTorch Lightning
    
    Args:
        devices_arg: Optional manually specified devices value
    
    Returns:
        Number of devices to use
    """
    if devices_arg is not None:
        return devices_arg
        
    if torch.backends.mps.is_available():
        return 1  # MPS only supports 1 device
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1  # Default to 1 CPU


def train_one_epoch(model, data_loader, optimizer, device, model_type):
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        data_loader: DataLoader with training data
        optimizer: Optimizer for model
        device: Device to train on
        model_type: Type of model being trained
        
    Returns:
        dict: Dictionary containing loss metrics
    """
    model.train()
    total_loss = 0
    mortality_loss = 0
    readmission_loss = 0
    phecode_loss = 0
    
    # Loss function
    bce_loss = nn.BCEWithLogitsLoss()
    
    for batch in tqdm(data_loader, desc="Training"):
        # Skip empty batches
        if not batch:
            continue
            
        # Get labels
        mortality_labels = batch['mortality_label'].float().to(device)
        readmission_labels = batch['readmission_label'].float().to(device)
        
        # Forward pass based on model type
        if model_type == 'ds_only':
            outputs = model(batch['ds_embedding'].to(device))
        elif model_type == 'raindrop_v2':
            # Prepare input for RaindropV2
            values = batch['values'].to(device)
            mask = batch['mask'].to(device)
            static = batch['static'].to(device) if batch['static'].numel() > 0 else None
            times = batch['times'].to(device)
            length = batch['length'].to(device)
            
            # RaindropV2 expects: src [max_len, batch_size, 2*d_inp]
            src = torch.cat([values, mask], dim=-1).permute(1, 0, 2)  # [T, B, F]
            times = times.permute(1, 0)  # [T, B]
            
            outputs = model(src, static, times, length)
        else:
            # Prepare input for MultiTaskKEDGN
            values = batch['values'].to(device)
            mask = batch['mask'].to(device)
            static = batch['static'].to(device) if batch['static'].numel() > 0 else None
            times = batch['times'].to(device)
            length = batch['length'].to(device)
            
            # Combine values and mask for the model input (follows KEDGN format)
            P = torch.cat([values, mask], dim=-1)
            P_static = static
            P_avg_interval = None  # Not used in simplified version
            P_length = length
            P_time = times
            P_var_plm_rep_tensor = torch.empty(0).to(device)  # Placeholder
            
            outputs = model(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
        
        # Calculate individual losses
        m_loss = bce_loss(outputs['mortality'].squeeze(-1), mortality_labels)
        r_loss = bce_loss(outputs['readmission'].squeeze(-1), readmission_labels)
        
        # Calculate PHE code loss using the modular function
        if 'next_idx_padded' in batch and 'phecodes' in outputs:
            idxs = batch["next_idx_padded"].to(device)
            lens = batch["next_len"].to(device)
            phecode_logits = outputs["phecodes"]
            p_loss = calculate_phecode_loss(batch, device, model.phe_code_size)
        else:
            p_loss = torch.tensor(0.0, device=device)
        
        
        loss = m_loss + r_loss + p_loss
            
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        mortality_loss += m_loss.item()
        readmission_loss += r_loss.item()
        phecode_loss += p_loss.item()
    
    # Calculate average losses
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_m_loss = mortality_loss / num_batches
    avg_r_loss = readmission_loss / num_batches
    avg_p_loss = phecode_loss / num_batches
    
    return {
        'loss': avg_loss,
        'mortality_loss': avg_m_loss,
        'readmission_loss': avg_r_loss,
        'phecode_loss': avg_p_loss
    }


def evaluate(model, data_loader, device, model_type):
    """
    Evaluate a model on test/validation data.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader with evaluation data
        device: Device to run evaluation on
        model_type: Type of model being evaluated
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    
    # Initialize metrics
    all_mortality_preds = []
    all_mortality_labels = []
    all_readmission_preds = []
    all_readmission_labels = []
    all_phecode_preds = []
    all_phecode_labels = []
    
    dataset = data_loader.dataset
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Skip empty batches
            if not batch:
                continue
                
            # Get labels
            mortality_labels = batch['mortality_label'].float().cpu().numpy()
            readmission_labels = batch['readmission_label'].float().cpu().numpy()
            
            # Forward pass - depends on model type
            if model_type == 'ds_only':
                outputs = model(batch['ds_embedding'].to(device))
            elif model_type == 'raindrop_v2':
                # Prepare input for RaindropV2
                values = batch['values'].to(device)
                mask = batch['mask'].to(device)
                static = batch['static'].to(device) if batch['static'].numel() > 0 else None
                times = batch['times'].to(device)
                length = batch['length'].to(device)
                
                # RaindropV2 expects: src [max_len, batch_size, 2*d_inp]
                src = torch.cat([values, mask], dim=-1).permute(1, 0, 2)  # [T, B, F]
                times = times.permute(1, 0)  # [T, B]
                
                outputs = model(src, static, times, length)
            else:
                # Prepare input for MultiTaskKEDGN
                values = batch['values'].to(device)
                mask = batch['mask'].to(device)
                static = batch['static'].to(device) if batch['static'].numel() > 0 else None
                times = batch['times'].to(device)
                length = batch['length'].to(device)
                
                # Combine values and mask for the model input
                P = torch.cat([values, mask], dim=-1)
                P_static = static
                P_avg_interval = None  # Not used in simplified version
                P_length = length
                P_time = times
                P_var_plm_rep_tensor = torch.empty(0).to(device)  # Placeholder
                
                outputs = model(P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor)
            
            # Get predictions
            mortality_preds = torch.sigmoid(outputs['mortality'].squeeze(-1)).cpu().numpy()
            readmission_preds = torch.sigmoid(outputs['readmission'].squeeze(-1)).cpu().numpy()
            
            # Store for metrics calculation
            all_mortality_preds.extend(mortality_preds)
            all_mortality_labels.extend(mortality_labels)
            all_readmission_preds.extend(readmission_preds)
            all_readmission_labels.extend(readmission_labels)
            
            # Process PHE code predictions using next_idx_padded and next_len
            if 'next_idx_padded' in batch and 'phecodes' in outputs:
                phecode_preds = torch.sigmoid(outputs['phecodes']).cpu().numpy()
                
                # Prepare PHE code targets
                if hasattr(model, 'phe_code_size'):
                    phecode_targets, valid_samples = prepare_phecode_targets(batch, device, model.phe_code_size)
                    if phecode_targets is not None:
                        phecode_labels_np = phecode_targets.cpu().numpy()
                        phecode_preds_np = phecode_preds
                        if valid_samples is not None:
                            phecode_preds_np = phecode_preds_np[valid_samples.cpu().numpy()]
                        all_phecode_preds.append(phecode_preds_np)
                        all_phecode_labels.append(phecode_labels_np)
    
    # Calculate metrics
    metrics = {}
    
    # Binary classification metrics
    if all_mortality_preds:
        m_metrics = calculate_binary_classification_metrics(all_mortality_preds, all_mortality_labels)
        metrics['mortality_auroc'] = m_metrics['auroc']
        metrics['mortality_auprc'] = m_metrics['auprc']
    
    if all_readmission_preds:
        r_metrics = calculate_binary_classification_metrics(all_readmission_preds, all_readmission_labels)
        metrics['readmission_auroc'] = r_metrics['auroc']
        metrics['readmission_auprc'] = r_metrics['auprc']
    
    # Add PHE code metrics if we have data
    if all_phecode_preds and all_phecode_labels:
        try:
            all_phecode_preds = np.vstack(all_phecode_preds)
            all_phecode_labels = np.vstack(all_phecode_labels)
            
            phe_metrics = calculate_phecode_metrics(all_phecode_preds, all_phecode_labels, dataset)
            
            metrics['phecode_macro_auc'] = phe_metrics.get('macro_auc', 0.0)
            metrics['phecode_micro_auc'] = phe_metrics.get('micro_auc', 0.0)
            metrics['phecode_micro_ap'] = phe_metrics.get('micro_ap', 0.0)
            metrics['phecode_prec@5'] = phe_metrics.get('prec@5', 0.0)
            
            if 'top_phecodes' in phe_metrics:
                metrics['top_phecodes'] = phe_metrics['top_phecodes']
        
        except Exception as e:
            logging.warning(f"Error calculating PHE code metrics: {e}")
    
    return metrics


def calculate_binary_classification_metrics(predictions, labels):
    """
    Calculate AUROC and AUPRC for binary classification.
    
    Args:
        predictions: List/array of model predictions (probabilities)
        labels: List/array of ground truth labels
        
    Returns:
        dict: Dictionary containing AUROC and AUPRC metrics
    """
    auroc = roc_auc_score(labels, predictions)
    auprc = average_precision_score(labels, predictions)
    
    return {
        'auroc': auroc,
        'auprc': auprc
    }

def calculate_phecode_metrics(phecode_preds, phecode_labels, dataset=None):
    """
    Calculate metrics for PHE code prediction.
    
    Args:
        phecode_preds: Model predictions for PHE codes [N, P]
        phecode_labels: Ground truth PHE code labels [N, P]
        dataset: Optional dataset object for additional information
        
    Returns:
        dict: Dictionary containing PHE code metrics
    """
    metrics = {}
    
    # Calculate metrics for PHE codes that have at least one positive example
    valid_cols = np.where(phecode_labels.sum(axis=0) > 0)[0]
    
    if len(valid_cols) > 0:
        # Macro AUC (average AUC across codes)
        phecode_aucs = []
        for col in valid_cols:
            if np.unique(phecode_labels[:, col]).shape[0] > 1:  # Need both classes present
                phecode_aucs.append(roc_auc_score(phecode_labels[:, col], phecode_preds[:, col]))
        
        if phecode_aucs:
            metrics['macro_auc'] = np.mean(phecode_aucs)
        
        # Micro AUC (flatten all predictions and calculate a single AUC)
        flat_preds = phecode_preds[:, valid_cols].flatten()
        flat_labels = phecode_labels[:, valid_cols].flatten()
        metrics['micro_auc'] = roc_auc_score(flat_labels, flat_preds)
        
        # Micro-averaged average precision
        metrics['micro_ap'] = average_precision_score(flat_labels, flat_preds)

        # Precision@5
        topk = 5
        top_preds = np.argsort(-phecode_preds, axis=1)[:, :topk]
        prec5_list = []
        for i in range(phecode_preds.shape[0]):
            true_set = set(np.where(phecode_labels[i])[0])
            pred_set = set(top_preds[i])
            prec5_list.append(len(pred_set & true_set) / topk)
        metrics['prec@5'] = float(np.mean(prec5_list))
        
        # Top PHE codes by frequency and their performance
        if dataset and hasattr(dataset, 'idx_to_phecode'):
            top_codes = []
            freqs = phecode_labels.sum(axis=0)
            top_indices = np.argsort(-freqs)[:10]  # Top 10 most frequent codes
            
            for idx in top_indices:
                if freqs[idx] > 0 and np.unique(phecode_labels[:, idx]).shape[0] > 1:
                    code = dataset.idx_to_phecode[idx]
                    freq = freqs[idx]
                    auc = roc_auc_score(phecode_labels[:, idx], phecode_preds[:, idx])
                    top_codes.append((code, freq, auc))
            
            metrics['top_phecodes'] = top_codes
    
    return metrics

def prepare_phecode_targets(batch_data, device, phecode_size, idx_key='next_idx_padded', len_key='next_phecode_len'):
    """
    Enhanced function to prepare phecode targets that better handles sparse data.
    
    Args:
        batch_data: Data batch containing phecode indices and lengths
        device: Device to place tensors on
        phecode_size: Total number of phecodes
        idx_key: Key for the phecode indices in batch_data
        len_key: Key for the phecode lengths in batch_data
        
    Returns:
        tuple of (targets, valid_samples)
    """
    if idx_key not in batch_data or len_key not in batch_data:
        return None, None
        
    # Get phecode indices and lengths
    idx_padded = batch_data[idx_key].to(device)  # [B, max_codes]
    phecode_len = batch_data[len_key].to(device)  # [B]
    
    if isinstance(phecode_len, dict) and 'mask' in phecode_len:
        # Handle specific data format if needed
        phecode_len = phecode_len['mask'].squeeze(-1)
    elif phecode_len.dim() > 1:
        phecode_len = phecode_len.squeeze()
        
    batch_size = idx_padded.size(0)
    
    # Create mask for samples with at least one phecode
    valid_samples = phecode_len > 0
    num_valid = valid_samples.sum().item()
    
    # If no samples have phecodes, return None
    if num_valid == 0:
        return None, None
    
    # Create binary target tensor just for valid samples
    targets = torch.zeros(num_valid, phecode_size, device=device)
    
    # Process only valid samples
    valid_idx_padded = idx_padded[valid_samples]
    valid_phecode_len = phecode_len[valid_samples]
    
    # For each valid sample, set corresponding phecodes to 1
    for i in range(num_valid):
        # Get indices for this sample (up to its length)
        sample_len = valid_phecode_len[i].item()
        if sample_len > 0:
            indices = valid_idx_padded[i, :sample_len]
            # Check if indices are valid (less than phecode_size)
            valid_indices = indices[indices < phecode_size]
            if len(valid_indices) > 0:
                targets[i, valid_indices] = 1.0
    
    return targets, valid_samples


def calculate_phecode_loss(batch_data, device, phecode_size, idx_key='next_idx_padded', len_key='next_phecode_len'):
    """
    Enhanced function to calculate phecode loss directly from batch.
    This function better handles batches with few valid phecode samples.
    
    Args:
        batch_data: Data batch containing phecode indices and lengths
        device: Device for tensors
        phecode_size: Number of possible phecodes
        idx_key: Key for the phecode indices in batch_data
        len_key: Key for the phecode lengths in batch_data
        
    Returns:
        loss: Computed loss value
    """
    # Skip if we don't have PHEcode data
    if idx_key not in batch_data or len_key not in batch_data:
        return torch.tensor(0.0, device=device)
    
    # Check if we have the phecode_logits key (for compatibility with train_one_epoch)
    phecode_logits = None
    if 'phecodes' in batch_data:
        phecode_logits = batch_data['phecodes']
    
    # Prepare targets using our function
    targets, valid_samples = prepare_phecode_targets(
        batch_data, device, phecode_size, idx_key, len_key
    )
    
    # If no valid targets, return zero loss
    if targets is None:
        return torch.tensor(0.0, device=device)
    
    # Apply valid samples mask to logits
    if valid_samples is not None:
        valid_logits = phecode_logits[valid_samples]
    else:
        valid_logits = phecode_logits
    
    # Calculate binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(valid_logits, targets)
    return loss


def report_phecode_statistics(phecode_targets, name="PHEcodes", logger=print):
    """
    Report detailed statistics about phecode distribution in dataset
    
    Args:
        phecode_targets: Binary phecode target tensor [N, phecode_size]
        name: Name to use in logging output
        logger: Function to use for logging (print or logging.info)
    """
    try:
        # Count codes per sample
        codes_per_sample = phecode_targets.sum(dim=1)
        
        # Count samples per code 
        samples_per_code = phecode_targets.sum(dim=0)
        
        # Get basic statistics
        total_samples = phecode_targets.shape[0]
        total_codes = phecode_targets.shape[1]
        
        # Calculate distribution statistics
        avg_codes = codes_per_sample.float().mean().item()
        median_codes = codes_per_sample.median().item()
        max_codes = codes_per_sample.max().item()
        min_codes = codes_per_sample.min().item()
        
        # Count codes that appear at least once
        active_codes = (samples_per_code > 0).sum().item()
        active_percent = (active_codes / total_codes) * 100
        
        # Calculate distribution of codes
        logger(f"{name} Statistics:")
        logger(f"  Total samples: {total_samples}, Active codes: {active_codes}/{total_codes} ({active_percent:.2f}%)")
        logger(f"  Codes per sample: Avg={avg_codes:.2f}, Median={median_codes}, Min={min_codes}, Max={max_codes}")
        
        # Calculate and report code frequency distribution
        if total_codes > 0:
            rare_codes = (samples_per_code > 0) & (samples_per_code <= 5)
            common_codes = samples_per_code > 100
            very_common_codes = samples_per_code > 500
            
            rare_count = rare_codes.sum().item()
            common_count = common_codes.sum().item()
            very_common_count = very_common_codes.sum().item()
            
            logger(f"  Code frequency: Rare (<=5 samples): {rare_count} ({rare_count/active_codes*100:.2f}% of active)")
            logger(f"  Code frequency: Common (>100 samples): {common_count} ({common_count/active_codes*100:.2f}% of active)")
            logger(f"  Code frequency: Very common (>500 samples): {very_common_count} ({very_common_count/active_codes*100:.2f}% of active)")
    except Exception as e:
        if callable(logger):
            logger(f"Error reporting phecode statistics: {e}")


def visualize_phecode_predictions(preds, targets, name="PHEcodes", num_samples=3, top_k=5, logger=print):
    """
    Visualize phecode predictions with accuracy indicators
    
    Args:
        preds: Model predictions after sigmoid [N, phecode_size]
        targets: Binary targets [N, phecode_size]
        name: Name for logging
        num_samples: Number of samples to visualize
        top_k: Number of top predictions to show
        logger: Function to use for logging (print or logging.info)
    """
    try:
        num_samples = min(num_samples, preds.shape[0])
        
        logger(f"\n==== {name} Prediction Visualization ====")
        
        for i in range(num_samples):
            # Get top predicted phecodes
            sample_preds = preds[i]
            sample_targets = targets[i]
            
            # Find top predicted phecodes
            top_preds = torch.topk(sample_preds, min(top_k, sample_preds.shape[0]))
            top_indices = top_preds.indices.cpu().numpy()
            top_values = top_preds.values.cpu().numpy()
            
            # Find actual positive phecodes
            actual_indices = torch.where(sample_targets > 0)[0].cpu().numpy()
            
            logger(f"  Sample {i} top predictions:")
            for idx, val in zip(top_indices, top_values):
                match = "✓" if idx in actual_indices else "✗"
                logger(f"    Phecode idx {idx}: {val:.4f} {match}")
            
            logger(f"  Sample {i} actual positives ({len(actual_indices)}):")
            logger(f"    {actual_indices[:10]}...")
        
        logger("==== End Prediction Visualization ====\n")
    except Exception as e:
        if callable(logger):
            logger(f"Error visualizing phecode predictions: {e}")


def get_phecode_statistics(batch_data, idx_key='next_idx_padded', len_key='next_phecode_len', logger=print):
    """
    Calculate and report phecode statistics from batch data
    
    Args:
        batch_data: Data batch containing phecode indices and lengths
        idx_key: Key for the phecode indices in batch_data
        len_key: Key for the phecode lengths in batch_data
        logger: Function to use for logging
        
    Returns:
        dict: Dictionary of statistics
    """
    if idx_key not in batch_data or len_key not in batch_data:
        return {}
        
    # Get phecode indices and lengths
    phecode_len = batch_data[len_key]
    
    if isinstance(phecode_len, dict) and 'mask' in phecode_len:
        phecode_len = phecode_len['mask'].squeeze(-1)
    elif phecode_len.dim() > 1:
        phecode_len = phecode_len.squeeze()
        
    # Create mask for samples with at least one phecode
    valid_samples = phecode_len > 0
    num_valid = valid_samples.sum().item()
    total_samples = len(phecode_len)
    valid_percent = (num_valid / total_samples) * 100 if total_samples > 0 else 0
    
    # Calculate average number of phecodes per sample
    if num_valid > 0:
        avg_codes = phecode_len[valid_samples].float().mean().item()
        max_codes = phecode_len.max().item()
    else:
        avg_codes = 0
        max_codes = 0
        
    # Log statistics
    if callable(logger):
        logger(f"Phecode Stats for {idx_key}: {num_valid}/{total_samples} ({valid_percent:.2f}%) samples have valid phecodes")
        logger(f"Average phecodes per valid sample: {avg_codes:.2f}, Max: {max_codes}")
    
    return {
        'num_valid': num_valid,
        'total_samples': total_samples,
        'valid_percent': valid_percent,
        'avg_codes': avg_codes,
        'max_codes': max_codes
    }

def zenker_derivatives(y, device):

    print(f"Y shape: {y.shape}. Expect 161 x 24")

    batch_size = y.shape[0]

    # y now contains: [i_ext (2), expert_latents (14), neural_embedding (4)]
    i_ext_1 = y[:, 0].unsqueeze(1)
    i_ext_2 = y[:, 1].unsqueeze(1)
    p_a = y[:, 2].unsqueeze(1)
    p_v = y[:, 3].unsqueeze(1)
    p_a = torch.clamp(p_a, min=40.0, max=200.0)  # MAP: 40-200 mmHg
    p_v = torch.clamp(p_v, min=0.0, max=39.0)

    s_reflex = y[:, 4].unsqueeze(1)
    sv = y[:, 5].unsqueeze(1)
    r_tpr_mod = y[:, 6].unsqueeze(1)
    f_hr_max = y[:, 7].unsqueeze(1)
    f_hr_min = y[:, 8].unsqueeze(1)
    r_tpr_max = y[:, 9].unsqueeze(1)
    r_tpr_min = y[:, 10].unsqueeze(1)
    c_a = y[:, 11].unsqueeze(1)
    c_v = y[:, 12].unsqueeze(1)
    k_width = y[:, 13].unsqueeze(1)
    p_aset = y[:, 14].unsqueeze(1)
    tau = y[:, 15].unsqueeze(1)


    f_hr = fhr(s_reflex,f_hr_max,f_hr_min)
    r_tpr = rtpr(s_reflex,r_tpr_max,r_tpr_min, r_tpr_mod)

    dpa_dt = dpa(p_a, p_v, r_tpr, sv, f_hr, c_a)
    dpv_dt = dpv(dpa_dt, c_a, c_v)
    ds_dt = dsdt(tau, k_width, p_a, p_aset, s_reflex)

    # TODO fix this
    dsv_dt = torch.zeros([batch_size, 1], device=device)

    # Fixed parameters don't change
    dt_r_tpr_mod = torch.zeros([batch_size, 1]).to(device)
    dt_f_hr_max = torch.zeros([batch_size, 1]).to(device)
    dt_f_hr_min = torch.zeros([batch_size, 1]).to(device)
    dt_r_tpr_max = torch.zeros([batch_size, 1]).to(device)
    dt_r_tpr_min = torch.zeros([batch_size, 1]).to(device)
    dt_ca = torch.zeros([batch_size, 1]).to(device)
    dt_cv = torch.zeros([batch_size, 1]).to(device)
    dt_k_width = torch.zeros([batch_size, 1]).to(device)
    dt_p_aset = torch.zeros([batch_size, 1]).to(device)
    dt_tau = torch.zeros([batch_size, 1]).to(device)

    # For expert latents (all 14 dims in the correct order)
    dt_expert = torch.cat([
        dt_r_tpr_mod, dt_f_hr_max, dt_f_hr_min,  # Next 3 (indices 6-8)
        dt_r_tpr_max, dt_r_tpr_min,  # Next 2 (indices 9-10)
        dt_ca, dt_cv, dt_k_width, dt_p_aset, dt_tau
    ], dim=-1)

    return dpa_dt, dpv_dt, ds_dt, dsv_dt, dt_expert, dt_r_tpr_mod, dt_f_hr_max, dt_f_hr_min, dt_r_tpr_max, dt_r_tpr_min, dt_ca, dt_cv, dt_k_width, dt_p_aset, dt_tau


def fhr(s_reflex, f_hr_max, f_hr_min):
    return s_reflex * (f_hr_max - f_hr_min) + f_hr_min

def rtpr(s_reflex, r_tpr_max, r_tpr_min, r_tpr_mod):
    return s_reflex * (r_tpr_max - r_tpr_min) + r_tpr_min + r_tpr_mod

def dpa(p_a, p_v, r_tpr, sv, f_hr, c_a):
    """
    Computes the derivative of the arterial pressure
    Args:
        p_a:
        p_v:
        r_tpr:
        sv:
        f_hr:

    Returns: dpa_dt, the derivative of the arterial pressure
    """
    return (1 / (c_a * 100))*(((p_a-p_v)/r_tpr) - sv*f_hr)

def dpv(dpa_dt, c_a, c_v):
    """
    Computes the venous pressure
    Args:
        dpa_dt:
        c_a:
        c_v:

    Returns:

    """
    # TODO note to self: we do not include ANY control here and assume this will be handled during the forward latent step (dependent on which model)
    return (1/(c_v*10))*(-c_a*dpa_dt)

def dsdt(tau, k_width, p_a, p_aset, s_reflex):

    return (1. / tau) * (1. - 1. / (1 + torch.exp(-k_width * (p_a - p_aset))) - s_reflex)






