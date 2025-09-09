# -*- coding:utf-8 -*-
import logging
import random
import torch
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from tqdm import tqdm
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import pandas as pd


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


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
        mortality_labels = batch["mortality_label"].float().to(device)
        readmission_labels = batch["readmission_label"].float().to(device)

        # Forward pass based on model type
        if model_type == "ds_only":
            outputs = model(batch["ds_embedding"].to(device))
        elif model_type == "raindrop_v2":
            # Prepare input for RaindropV2
            values = batch["values"].to(device)
            mask = batch["mask"].to(device)
            static = batch["static"].to(device) if batch["static"].numel() > 0 else None
            times = batch["times"].to(device)
            length = batch["length"].to(device)

            # RaindropV2 expects: src [max_len, batch_size, 2*d_inp]
            src = torch.cat([values, mask], dim=-1).permute(1, 0, 2)  # [T, B, F]
            times = times.permute(1, 0)  # [T, B]

            outputs = model(src, static, times, length)
        else:
            # Prepare input for MultiTaskKEDGN
            values = batch["values"].to(device)
            mask = batch["mask"].to(device)
            static = batch["static"].to(device) if batch["static"].numel() > 0 else None
            times = batch["times"].to(device)
            length = batch["length"].to(device)

            # Combine values and mask for the model input (follows KEDGN format)
            P = torch.cat([values, mask], dim=-1)
            P_static = static
            P_avg_interval = None  # Not used in simplified version
            P_length = length
            P_time = times
            P_var_plm_rep_tensor = torch.empty(0).to(device)  # Placeholder

            outputs = model(
                P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor
            )

        # Calculate individual losses
        m_loss = bce_loss(outputs["mortality"].squeeze(-1), mortality_labels)
        r_loss = bce_loss(outputs["readmission"].squeeze(-1), readmission_labels)

        # Calculate PHE code loss using the modular function
        if "next_idx_padded" in batch and "phecodes" in outputs:
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
        "loss": avg_loss,
        "mortality_loss": avg_m_loss,
        "readmission_loss": avg_r_loss,
        "phecode_loss": avg_p_loss,
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
            mortality_labels = batch["mortality_label"].float().cpu().numpy()
            readmission_labels = batch["readmission_label"].float().cpu().numpy()

            # Forward pass - depends on model type
            if model_type == "ds_only":
                outputs = model(batch["ds_embedding"].to(device))
            elif model_type == "raindrop_v2":
                # Prepare input for RaindropV2
                values = batch["values"].to(device)
                mask = batch["mask"].to(device)
                static = (
                    batch["static"].to(device) if batch["static"].numel() > 0 else None
                )
                times = batch["times"].to(device)
                length = batch["length"].to(device)

                # RaindropV2 expects: src [max_len, batch_size, 2*d_inp]
                src = torch.cat([values, mask], dim=-1).permute(1, 0, 2)  # [T, B, F]
                times = times.permute(1, 0)  # [T, B]

                outputs = model(src, static, times, length)
            else:
                # Prepare input for MultiTaskKEDGN
                values = batch["values"].to(device)
                mask = batch["mask"].to(device)
                static = (
                    batch["static"].to(device) if batch["static"].numel() > 0 else None
                )
                times = batch["times"].to(device)
                length = batch["length"].to(device)

                # Combine values and mask for the model input
                P = torch.cat([values, mask], dim=-1)
                P_static = static
                P_avg_interval = None  # Not used in simplified version
                P_length = length
                P_time = times
                P_var_plm_rep_tensor = torch.empty(0).to(device)  # Placeholder

                outputs = model(
                    P, P_static, P_avg_interval, P_length, P_time, P_var_plm_rep_tensor
                )

            # Get predictions
            mortality_preds = (
                torch.sigmoid(outputs["mortality"].squeeze(-1)).cpu().numpy()
            )
            readmission_preds = (
                torch.sigmoid(outputs["readmission"].squeeze(-1)).cpu().numpy()
            )

            # Store for metrics calculation
            all_mortality_preds.extend(mortality_preds)
            all_mortality_labels.extend(mortality_labels)
            all_readmission_preds.extend(readmission_preds)
            all_readmission_labels.extend(readmission_labels)

            # Process PHE code predictions using next_idx_padded and next_len
            if "next_idx_padded" in batch and "phecodes" in outputs:
                phecode_preds = torch.sigmoid(outputs["phecodes"]).cpu().numpy()

                # Prepare PHE code targets
                if hasattr(model, "phe_code_size"):
                    phecode_targets, valid_samples = prepare_phecode_targets(
                        batch, device, model.phe_code_size
                    )
                    if phecode_targets is not None:
                        phecode_labels_np = phecode_targets.cpu().numpy()
                        phecode_preds_np = phecode_preds
                        if valid_samples is not None:
                            phecode_preds_np = phecode_preds_np[
                                valid_samples.cpu().numpy()
                            ]
                        all_phecode_preds.append(phecode_preds_np)
                        all_phecode_labels.append(phecode_labels_np)

    # Calculate metrics
    metrics = {}

    # Binary classification metrics
    if all_mortality_preds:
        m_metrics = calculate_binary_classification_metrics(
            all_mortality_preds, all_mortality_labels
        )
        metrics["mortality_auroc"] = m_metrics["auroc"]
        metrics["mortality_auprc"] = m_metrics["auprc"]

    if all_readmission_preds:
        r_metrics = calculate_binary_classification_metrics(
            all_readmission_preds, all_readmission_labels
        )
        metrics["readmission_auroc"] = r_metrics["auroc"]
        metrics["readmission_auprc"] = r_metrics["auprc"]

    # Add PHE code metrics if we have data
    if all_phecode_preds and all_phecode_labels:
        try:
            all_phecode_preds = np.vstack(all_phecode_preds)
            all_phecode_labels = np.vstack(all_phecode_labels)

            phe_metrics = calculate_phecode_metrics(
                all_phecode_preds, all_phecode_labels, dataset
            )

            metrics["phecode_macro_auc"] = phe_metrics.get("macro_auc", 0.0)
            metrics["phecode_micro_auc"] = phe_metrics.get("micro_auc", 0.0)
            metrics["phecode_micro_ap"] = phe_metrics.get("micro_ap", 0.0)
            metrics["phecode_prec@5"] = phe_metrics.get("prec@5", 0.0)

            if "top_phecodes" in phe_metrics:
                metrics["top_phecodes"] = phe_metrics["top_phecodes"]

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

    return {"auroc": auroc, "auprc": auprc}


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
            if (
                np.unique(phecode_labels[:, col]).shape[0] > 1
            ):  # Need both classes present
                phecode_aucs.append(
                    roc_auc_score(phecode_labels[:, col], phecode_preds[:, col])
                )

        if phecode_aucs:
            metrics["macro_auc"] = np.mean(phecode_aucs)

        # Micro AUC (flatten all predictions and calculate a single AUC)
        flat_preds = phecode_preds[:, valid_cols].flatten()
        flat_labels = phecode_labels[:, valid_cols].flatten()
        metrics["micro_auc"] = roc_auc_score(flat_labels, flat_preds)

        # Micro-averaged average precision
        metrics["micro_ap"] = average_precision_score(flat_labels, flat_preds)

        # Precision@5
        topk = 5
        top_preds = np.argsort(-phecode_preds, axis=1)[:, :topk]
        prec5_list = []
        for i in range(phecode_preds.shape[0]):
            true_set = set(np.where(phecode_labels[i])[0])
            pred_set = set(top_preds[i])
            prec5_list.append(len(pred_set & true_set) / topk)
        metrics["prec@5"] = float(np.mean(prec5_list))

        # Top PHE codes by frequency and their performance
        if dataset and hasattr(dataset, "idx_to_phecode"):
            top_codes = []
            freqs = phecode_labels.sum(axis=0)
            top_indices = np.argsort(-freqs)[:10]  # Top 10 most frequent codes

            for idx in top_indices:
                if freqs[idx] > 0 and np.unique(phecode_labels[:, idx]).shape[0] > 1:
                    code = dataset.idx_to_phecode[idx]
                    freq = freqs[idx]
                    auc = roc_auc_score(phecode_labels[:, idx], phecode_preds[:, idx])
                    top_codes.append((code, freq, auc))

            metrics["top_phecodes"] = top_codes

    return metrics


def prepare_phecode_targets(
    batch_data,
    device,
    phecode_size,
    idx_key="next_idx_padded",
    len_key="next_phecode_len",
):
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

    if isinstance(phecode_len, dict) and "mask" in phecode_len:
        # Handle specific data format if needed
        phecode_len = phecode_len["mask"].squeeze(-1)
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


def calculate_phecode_loss(
    batch_data,
    device,
    phecode_size,
    idx_key="next_idx_padded",
    len_key="next_phecode_len",
):
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
    if "phecodes" in batch_data:
        phecode_logits = batch_data["phecodes"]

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
        logger(
            f"  Total samples: {total_samples}, Active codes: {active_codes}/{total_codes} ({active_percent:.2f}%)"
        )
        logger(
            f"  Codes per sample: Avg={avg_codes:.2f}, Median={median_codes}, Min={min_codes}, Max={max_codes}"
        )

        # Calculate and report code frequency distribution
        if total_codes > 0:
            rare_codes = (samples_per_code > 0) & (samples_per_code <= 5)
            common_codes = samples_per_code > 100
            very_common_codes = samples_per_code > 500

            rare_count = rare_codes.sum().item()
            common_count = common_codes.sum().item()
            very_common_count = very_common_codes.sum().item()

            logger(
                f"  Code frequency: Rare (<=5 samples): {rare_count} ({rare_count/active_codes*100:.2f}% of active)"
            )
            logger(
                f"  Code frequency: Common (>100 samples): {common_count} ({common_count/active_codes*100:.2f}% of active)"
            )
            logger(
                f"  Code frequency: Very common (>500 samples): {very_common_count} ({very_common_count/active_codes*100:.2f}% of active)"
            )
    except Exception as e:
        if callable(logger):
            logger(f"Error reporting phecode statistics: {e}")


def visualize_phecode_predictions(
    preds, targets, name="PHEcodes", num_samples=3, top_k=5, logger=print
):
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


def get_phecode_statistics(
    batch_data, idx_key="next_idx_padded", len_key="next_phecode_len", logger=print
):
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

    if isinstance(phecode_len, dict) and "mask" in phecode_len:
        phecode_len = phecode_len["mask"].squeeze(-1)
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
        logger(
            f"Phecode Stats for {idx_key}: {num_valid}/{total_samples} ({valid_percent:.2f}%) samples have valid phecodes"
        )
        logger(f"Average phecodes per valid sample: {avg_codes:.2f}, Max: {max_codes}")

    return {
        "num_valid": num_valid,
        "total_samples": total_samples,
        "valid_percent": valid_percent,
        "avg_codes": avg_codes,
        "max_codes": max_codes,
    }


def zenker_derivatives(y, device, expert_start_index: int = 0):
    batch_size = y.shape[0]

    # Expect expert variables in the order:
    # [p_a, p_v, s_reflex, sv, r_tpr_mod, f_hr_max, f_hr_min, r_tpr_max, r_tpr_min, ca, cv, k_width, p_aset, tau]
    # Use expert_start_index to support callers passing a larger state vector with leading controls.
    off = expert_start_index
    p_a = y[:, off + 0].unsqueeze(1)
    p_v = y[:, off + 1].unsqueeze(1)

    s_reflex = y[:, off + 2].unsqueeze(1)
    sv = y[:, off + 3].unsqueeze(1)
    r_tpr_mod = y[:, off + 4].unsqueeze(1)
    f_hr_max = y[:, off + 5].unsqueeze(1)
    f_hr_min = y[:, off + 6].unsqueeze(1)
    r_tpr_max = y[:, off + 7].unsqueeze(1)
    r_tpr_min = y[:, off + 8].unsqueeze(1)
    c_a = y[:, off + 9].unsqueeze(1)
    c_v = y[:, off + 10].unsqueeze(1)
    k_width = y[:, off + 11].unsqueeze(1)
    p_aset = y[:, off + 12].unsqueeze(1)
    tau = y[:, off + 13].unsqueeze(1)

    f_hr = fhr(s_reflex, f_hr_max, f_hr_min)
    r_tpr = rtpr(s_reflex, r_tpr_max, r_tpr_min, r_tpr_mod)

    dpa_dt = dpa(p_a, p_v, r_tpr, sv, f_hr, c_a)
    dpv_dt = dpv(dpa_dt, c_a, c_v)
    ds_dt = dsdt(tau, k_width, p_a, p_aset, s_reflex)

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
    dt_expert = torch.cat(
        [
            dt_r_tpr_mod,
            dt_f_hr_max,
            dt_f_hr_min,  # Next 3 (indices 6-8)
            dt_r_tpr_max,
            dt_r_tpr_min,  # Next 2 (indices 9-10)
            dt_ca,
            dt_cv,
            dt_k_width,
            dt_p_aset,
            dt_tau,
        ],
        dim=-1,
    )

    return (
        dpa_dt,
        dpv_dt,
        ds_dt,
        dsv_dt,
        dt_expert,
        dt_r_tpr_mod,
        dt_f_hr_max,
        dt_f_hr_min,
        dt_r_tpr_max,
        dt_r_tpr_min,
        dt_ca,
        dt_cv,
        dt_k_width,
        dt_p_aset,
        dt_tau,
    )


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

    outflow = (p_a - p_v) / r_tpr  # ml/s
    inflow = sv * f_hr  # ml/s
    dva_dt = -1.0 * outflow + inflow

    # Pressure derivatives exactly as specified
    dpa_dt = dva_dt / (c_a)

    return dpa_dt


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
    return (1 / (c_v)) * (-c_a * dpa_dt)


def dsdt(tau, k_width, p_a, p_aset, s_reflex):
    return (1.0 / tau) * (
        1.0 - 1.0 / (1 + torch.exp(-k_width * (p_a - p_aset))) - s_reflex
    )


def midpoint(r: Tuple[float, float]) -> float:
    return 0.5 * (r[0] + r[1])


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _to_tensor(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)




def _normalize_label(s: str) -> str:
    """Normalize label text for matching across index_map and df_ranges."""
    return str(s).strip().lower().replace("_", " ").replace("-", " ")

def _stats_tensors_from_ranges(
    df_ranges: pd.DataFrame,
    labels,
    group_col: str = 'item_label',
    device=None,
    dtype=torch.float32
):
    """
    Build per-feature mean/std/lower/upper tensors (shape [K]) from df_ranges.
    Label matching is normalized (lowercased, underscores -> spaces, etc.).
    """
    required = {'lower_bound', 'upper_bound', group_col}
    if not required.issubset(df_ranges.columns):
        missing = required - set(df_ranges.columns)
        raise ValueError(f"df_ranges is missing required columns: {missing}")

    # Normalize df_ranges labels
    df_ranges = df_ranges.copy()
    df_ranges['_norm_label'] = df_ranges[group_col].map(_normalize_label)
    stats = df_ranges.set_index('_norm_label')[['lower_bound', 'upper_bound']]

    # Normalize the labels list
    norm_labels = [_normalize_label(l) for l in labels]

    # Build tensors in the order of labels
    try:
        lb = torch.tensor([float(stats.loc[l, 'lower_bound']) for l in norm_labels],
                          device=device, dtype=dtype)
        ub = torch.tensor([float(stats.loc[l, 'upper_bound']) for l in norm_labels],
                          device=device, dtype=dtype)
    except KeyError as e:
        raise KeyError(f"Label '{e.args[0]}' not found after normalization in df_ranges[{group_col}].")

    mean = (lb + ub) / 2.0
    std  = (ub - lb) / 6.0
    return mean, std, lb, ub



def denormalize_selected_from_batched_sample(
    chartevents_sample: torch.Tensor,
    df_ranges: pd.DataFrame,
    index_map: dict,
    group_col: str = 'item_label',
    d_inp: int = 96,
    use_mask: bool = True,
    clip: bool = True,
):
    """
    Vectorized inverse-normalization for a batched tensor.

    Args:
      chartevents_sample: torch.Tensor of shape [B, T, 2*d_inp]
                          first d_inp = z-scored values,
                          next d_inp  = **missing mask** (1 = missing, 0 = present; float/int/bool)
      df_ranges: DataFrame with columns [group_col, lower_bound, upper_bound] (bounds are mean ± 3*std)
      index_map: dict mapping item_label -> feature index (column in X) for features of interest
      group_col: column name in df_ranges that matches item labels in index_map
      d_inp: number of clinical features (e.g., 96)
      use_mask: if True, use the provided missing mask; otherwise treat non-NaN as present
      clip: clip z to [-3, 3] before inverse-transform (recommended)

    Returns (all torch tensors):
      original_values: [B, K] inverse-mapped to original scale
      z_values:        [B, K] raw z-scores extracted at the last present timestep
      timesteps:       [B, K] int64 indices of last present timestep (=-1 where none present)
      means:           [K]
      stds:            [K]
      lower_bounds:    [K]
      upper_bounds:    [K]
      present_mask:    [B, K] bool, True where a present value was found
    """
    if not torch.is_tensor(chartevents_sample) or chartevents_sample.dim() != 3:
        raise ValueError("chartevents_sample must be a torch.Tensor of shape [B, T, 2*d_inp].")

    B, T, F_total = chartevents_sample.shape
    if F_total != 2 * d_inp:
        raise ValueError(f"Expected last dim = 2*d_inp ({2*d_inp}), got {F_total}.")

    device = chartevents_sample.device
    dtype = chartevents_sample.dtype

    # Split into values and mask
    X = chartevents_sample[:, :, :d_inp]           # [B, T, d_inp]
    M = chartevents_sample[:, :, d_inp:]           # [B, T, d_inp]  (1 = missing, 0 = present)
    
    #print('X:', X[0, -1, :])
    #print('M:', M[0, -1, :])

    # Indices/labels of interest
    labels = list(index_map.keys())
    feat_idx = torch.tensor([index_map[l] for l in labels], device=device, dtype=torch.long)
    if (feat_idx < 0).any() or (feat_idx >= d_inp).any():
        raise IndexError("One or more feature indices in index_map are out of bounds for d_inp.")

    # Select features of interest
    X_sel = X.index_select(dim=2, index=feat_idx)                # [B, T, K]
    #print('X_sel:', X_sel[0, -1, :])

    if use_mask:
        # Convert to boolean *missing* mask (True = missing), then invert to get presence
        M_sel_missing = M.index_select(dim=2, index=feat_idx)
        if M_sel_missing.dtype != torch.bool:
            M_sel_missing = M_sel_missing > 0.5
        present = ~M_sel_missing                                  # True = present
    else:
        present = ~torch.isnan(X_sel)
    #print('present:', present[0, -1, :])

    # Compute last present timestep per (B,K)
    time_idx = torch.arange(T, device=device).view(1, T, 1)      # [1, T, 1]
    # Put 0 where absent, then take max; fix up later where none are present
    idx_weighted = torch.where(present, time_idx, torch.zeros(1, dtype=time_idx.dtype, device=device))
    last_idx = idx_weighted.amax(dim=1)                          # [B, K], 0 if none present
    any_present = present.any(dim=1)                             # [B, K] bool
    last_idx = torch.where(any_present, last_idx, torch.full_like(last_idx, -1))

    # Gather z-values at the last present timestep
    last_idx_safe = torch.clamp(last_idx, min=0)                 # [B, K]
    z_values = X_sel.gather(1, last_idx_safe.unsqueeze(1).expand(-1, 1, -1)).squeeze(1)  # [B, K]
    z_values = torch.where(any_present, z_values, torch.full_like(z_values, float('nan')))

    # Stats in K-order
    means, stds, lower_bounds, upper_bounds = _stats_tensors_from_ranges(
        df_ranges, labels, group_col=group_col, device=device, dtype=torch.float32
    )
    means = means.to(dtype)
    stds = stds.to(dtype)
    lower_bounds = lower_bounds.to(dtype)
    upper_bounds = upper_bounds.to(dtype)

    # Inverse transform
    z_use = torch.clamp(z_values, -3.0, 3.0) if clip else z_values
    original_values = z_use * stds.view(1, -1) + means.view(1, -1)              # [B, K]
    original_values = torch.where(any_present, original_values, torch.full_like(original_values, float('nan')))

    return (
        original_values,         # [B, K]
        z_values,                # [B, K]
        last_idx.to(torch.long), # [B, K]
        means,                   # [K]
        stds,                    # [K]
        lower_bounds,            # [K]
        upper_bounds,            # [K]
        any_present              # [B, K] bool
    )



# Order of ODE variables (L = 14)
_EXPERT_ORDER = [
    "p_a", "p_v", "s_reflex", "sv", "r_tpr_mod",
    "f_hr_max", "f_hr_min", "r_tpr_max", "r_tpr_min",
    "ca", "cv", "k_width", "p_aset", "tau",
]

def _midpoint_pair(pair):
    return 0.5 * (float(pair[0]) + float(pair[1]))

def _normalize_label(s: str) -> str:
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def _ranges_midpoint_tensor(physio_ranges: dict, device, dtype) -> torch.Tensor:
    """
    Build a [L] tensor of midpoints in _EXPERT_ORDER from a dict physio_ranges.
    """
    vals = []
    for key in _EXPERT_ORDER:
        pair = physio_ranges[key]
        vals.append(_midpoint_pair(pair))
    return torch.tensor(vals, device=device, dtype=dtype)

def _build_full_state_from_equilibrium(
    Pa: torch.Tensor, Pv: torch.Tensor, Hr: torch.Tensor,
    eq: dict, physio_ranges: dict, device, dtype
) -> torch.Tensor:
    """
    Assemble [B, L] in _EXPERT_ORDER using equilibrium outputs + midpoints.
    """
    B = Pa.shape[0]
    base = _ranges_midpoint_tensor(physio_ranges, device, dtype).unsqueeze(0).expand(B, -1).clone()

    # Write equilibrium-derived entries
    # Indices by order:
    idx = {name: i for i, name in enumerate(_EXPERT_ORDER)}

    base[:, idx["p_a"]]        = eq["Pa"].to(dtype)
    base[:, idx["p_v"]]        = eq["Pv"].to(dtype)
    base[:, idx["s_reflex"]]   = eq["s"].to(dtype)
    base[:, idx["sv"]]         = eq["SV"].to(dtype)
    base[:, idx["r_tpr_mod"]]  = eq["r_tpr_mod"].to(dtype)
    base[:, idx["p_aset"]]     = eq["p_aset_star"].to(dtype)

    # The rest (f_hr_max, f_hr_min, r_tpr_max, r_tpr_min, ca, cv, k_width, tau)
    # remain as midpoints from physio_ranges (already set in base)

    return base

def _dict_to_PhysioRanges(physio_ranges: dict):
    """
    Convert dict (like self.physio_ranges) to PhysioRanges dataclass expected by compute_stable_equilibrium_batch.
    """
    return PhysioRanges(
        p_a=tuple(physio_ranges["p_a"]),
        p_v=tuple(physio_ranges["p_v"]),
        s_reflex=tuple(physio_ranges["s_reflex"]),
        sv=tuple(physio_ranges["sv"]),
        r_tpr_mod=tuple(physio_ranges["r_tpr_mod"]),
        f_hr_max=tuple(physio_ranges["f_hr_max"]),
        f_hr_min=tuple(physio_ranges["f_hr_min"]),
        r_tpr_max=tuple(physio_ranges["r_tpr_max"]),
        r_tpr_min=tuple(physio_ranges["r_tpr_min"]),
        ca=tuple(physio_ranges["ca"]),
        cv=tuple(physio_ranges["cv"]),
        k_width=tuple(physio_ranges["k_width"]),
        p_aset=tuple(physio_ranges["p_aset"]),
        tau=tuple(physio_ranges["tau"]),
    )

def _pick_ic_or_fallback(ic_vals: torch.Tensor, ic_mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Prefer IC where mask==1 and finite; else fallback. All [B]."""
    use_ic = (ic_mask == 1) & torch.isfinite(ic_vals)
    return torch.where(use_ic, ic_vals, fallback)




@dataclass
class PhysioRanges:
    p_a: Tuple[float, float] = (40.0, 180.0)
    p_v: Tuple[float, float] = (0.0, 30.0)
    s_reflex: Tuple[float, float] = (0.0, 1.0)
    sv: Tuple[float, float] = (40.0, 120.0)
    r_tpr_mod: Tuple[float, float] = (-1.0, 1.0)
    f_hr_max: Tuple[float, float] = (2.0, 3.0)
    f_hr_min: Tuple[float, float] = (0.9, 1.1)
    r_tpr_max: Tuple[float, float] = (1.8, 2.4)
    r_tpr_min: Tuple[float, float] = (0.45, 0.6)
    ca: Tuple[float, float] = (2.0, 6.0)
    cv: Tuple[float, float] = (90.0, 120.0)
    k_width: Tuple[float, float] = (0.1, 0.3)
    p_aset: Tuple[float, float] = (50.0, 90.0)
    tau: Tuple[float, float] = (15.0, 25.0)





def compute_stable_equilibrium_batch(
    Pa: torch.Tensor,                   # [B] mmHg (arterial pressure)
    Pv: torch.Tensor,                   # [B] mmHg (venous/CVP)
    Hr: torch.Tensor,                   # [B] heart rate (Hz or bpm; see hr_units)
    *,
    # Units & priors
    hr_units: str = "hz",               # "hz" (default) or "bpm"
    hr_prior_bpm: float = 75.0,         # used if both Pa & Hr missing; ~1.25 Hz
    pv_prior: float = 8.0,              # physiologic CVP prior
    pv_minmax: Tuple[float, float] = (2.0, 15.0),  # clamp for CVP prior
    paset_prior: Optional[torch.Tensor] = None,    # [B] or scalar; default midpoint(ranges.p_aset)

    # Model params (broadcastable to [B] or None => midpoints)
    f_hr_min: Optional[torch.Tensor] = None,
    f_hr_max: Optional[torch.Tensor] = None,
    r_tpr_min: Optional[torch.Tensor] = None,
    r_tpr_max: Optional[torch.Tensor] = None,
    k_width: Optional[torch.Tensor] = None,
    r_tpr_mod_fixed: Optional[torch.Tensor] = 0.0,

    # Ranges & numerics
    ranges = None,                      # PhysioRanges(); pass your instance or None => defaults inside
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Robust, NaN-proof batched solver.
    - Fills any missing Pa/Pv/Hr analytically:
        * If Hr present & Pa missing:   Pa from baroreflex with paset_prior.
        * If Pa present & Hr missing:   Hr from baroreflex with paset_prior.
        * If both Pa & Hr missing:      use hr_prior_bpm + paset_prior to infer a consistent pair.
        * If Pv missing:                Pv <- pv_prior (clamped).
    - Then runs the same core logic (R_base, SV, r_tpr_mod adjust-on-violation).
    - Uses safe logs and final NaN trapping; returns provenance flags.
    """

    # ---- Helpers ----
    class _DefaultRanges:
        # Fallback if ranges is None
        p_a       = (40.0, 180.0)
        p_v       = (0.0, 30.0)
        s_reflex  = (0.0, 1.0)
        sv        = (40.0, 120.0)
        r_tpr_mod = (-1.0, 1.0)
        f_hr_max  = (2.0, 3.0)
        f_hr_min  = (0.9, 1.1)
        r_tpr_max = (1.8, 2.4)
        r_tpr_min = (0.45, 0.6)
        ca        = (2.0, 6.0)
        cv        = (90.0, 120.0)
        k_width   = (0.1, 0.3)
        p_aset    = (50.0, 90.0)
        tau       = (15.0, 25.0)

    rng = ranges if ranges is not None else _DefaultRanges()

    def _mid(pair):  # pair = (lo, hi)
        return 0.5 * (pair[0] + pair[1])

    def _to_full(x, default_pair, B, device, dtype):
        if x is None:
            return torch.full((B,), _mid(default_pair), device=device, dtype=dtype)
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype).expand(B)
        return torch.full((B,), float(x), device=device, dtype=dtype)

    def _fill_nan_with(x: torch.Tensor, fill: float) -> torch.Tensor:
        return torch.where(torch.isnan(x), torch.full_like(x, fill), x)

    def _safe_log_ratio_one_minus_over(x: torch.Tensor) -> torch.Tensor:
        # log((1 - x)/x) computed stably
        return torch.log1p(-x) - torch.log(x)

    def _nan_to_num_like(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Shape checks ----
    if not (Pa.ndim == Pv.ndim == Hr.ndim == 1):
        raise ValueError("Pa, Pv, Hr must be 1-D tensors [B].")

    B, device, dtype = Pa.shape[0], Pa.device, Pa.dtype

    # ---- Unit handling ----
    Hr_in = Hr.clone()
    if hr_units.lower() == "bpm":
        Hr = Hr / 60.0

    # ---- Param vectors (midpoints where None) ----
    f_hr_min = _to_full(f_hr_min, rng.f_hr_min, B, device, dtype)
    f_hr_max = _to_full(f_hr_max, rng.f_hr_max, B, device, dtype)
    r_tpr_min = _to_full(r_tpr_min, rng.r_tpr_min, B, device, dtype)
    r_tpr_max = _to_full(r_tpr_max, rng.r_tpr_max, B, device, dtype)
    k_width   = _to_full(k_width,   rng.k_width,   B, device, dtype)
    if r_tpr_mod_fixed is None:
        r_tpr_mod_fixed = torch.zeros(B, device=device, dtype=dtype)
    elif not torch.is_tensor(r_tpr_mod_fixed):
        r_tpr_mod_fixed = torch.full((B,), float(r_tpr_mod_fixed), device=device, dtype=dtype)
    else:
        r_tpr_mod_fixed = r_tpr_mod_fixed.to(device=device, dtype=dtype).expand(B)

    if paset_prior is None:
        paset_prior = torch.full((B,), _mid(rng.p_aset), device=device, dtype=dtype)
    else:
        paset_prior = (paset_prior if torch.is_tensor(paset_prior)
                       else torch.tensor(paset_prior)).to(device=device, dtype=dtype).expand(B)

    # ---- Sanitize params (avoid NaNs/zeros) ----
    f_hr_min = _fill_nan_with(f_hr_min, _mid(rng.f_hr_min))
    f_hr_max = _fill_nan_with(f_hr_max, _mid(rng.f_hr_max))
    k_width  = _fill_nan_with(k_width,  _mid(rng.k_width)).clamp_min(eps)
    denom_hr = (f_hr_max - f_hr_min)
    denom_hr = torch.where(torch.isnan(denom_hr), torch.full_like(denom_hr, _mid(rng.f_hr_max) - _mid(rng.f_hr_min)), denom_hr)
    denom_hr = denom_hr.clamp_min(eps)

    # ---- Presence masks ----
    mPa = torch.isfinite(Pa)
    mPv = torch.isfinite(Pv)
    mHr = torch.isfinite(Hr)

    # Provenance flags
    Pa_from_HR_paset = torch.zeros(B, device=device, dtype=dtype)
    Hr_from_Pa_paset = torch.zeros(B, device=device, dtype=dtype)
    Pv_from_prior    = torch.zeros(B, device=device, dtype=dtype)
    PaHr_joint_fill  = torch.zeros(B, device=device, dtype=dtype)
    param_sanitized  = torch.zeros(B, device=device, dtype=dtype)  # set if any param was NaN

    # Track if any param had NaN originally
    any_param_nan = torch.isnan(_to_full(None, rng.f_hr_min, B, device, dtype))  # dummy False
    any_param_nan |= torch.isnan(f_hr_min) | torch.isnan(f_hr_max) | torch.isnan(k_width)
    param_sanitized = any_param_nan.to(dtype)

    # ---- Joint fallback when BOTH Pa & Hr are missing ----
    Pa_filled = Pa.clone()
    Hr_filled = Hr.clone()

    both_missing = (~mPa) & (~mHr)
    if both_missing.any():
        # Use HR prior (in Hz), infer Pa via baroreflex with paset_prior
        hr_prior_hz = torch.full((B,), hr_prior_bpm / 60.0, device=device, dtype=dtype)
        s0 = ((hr_prior_hz - f_hr_min) / denom_hr).clamp(eps, 1.0 - eps)
        log_ratio = _safe_log_ratio_one_minus_over(s0)
        Pa_filled[both_missing] = paset_prior[both_missing] - (1.0 / k_width[both_missing]) * log_ratio[both_missing]
        Hr_filled[both_missing] = hr_prior_hz[both_missing]
        PaHr_joint_fill[both_missing] = 1.0

    # ---- One-sided fills ----
    # If Hr present & Pa missing -> infer Pa
    need_Pa = (~mPa) & mHr
    if need_Pa.any():
        s_from_HR = ((Hr_filled[need_Pa] - f_hr_min[need_Pa]) / denom_hr[need_Pa]).clamp(eps, 1.0 - eps)
        log_ratio = _safe_log_ratio_one_minus_over(s_from_HR)
        Pa_filled[need_Pa] = paset_prior[need_Pa] - (1.0 / k_width[need_Pa]) * log_ratio
        Pa_from_HR_paset[need_Pa] = 1.0

    # If Pa present & Hr missing -> infer Hr
    need_Hr = mPa & (~mHr)
    if need_Hr.any():
        s_baro = torch.sigmoid(-k_width[need_Hr] * (Pa_filled[need_Hr] - paset_prior[need_Hr]))
        Hr_filled[need_Hr] = f_hr_min[need_Hr] + s_baro * (f_hr_max[need_Hr] - f_hr_min[need_Hr])
        Hr_from_Pa_paset[need_Hr] = 1.0

    # ---- Pv prior if missing ----
    Pv_filled = Pv.clone()
    need_Pv = ~mPv
    if need_Pv.any():
        lo, hi = pv_minmax
        Pv_filled[need_Pv] = torch.full_like(Pv_filled[need_Pv], pv_prior).clamp(min=lo, max=hi)
        Pv_from_prior[need_Pv] = 1.0

    # After fills, everything should be finite; enforce backstops just in case
    Pa_filled = torch.where(torch.isfinite(Pa_filled), Pa_filled, paset_prior)  # Pa ~ paset if anything slipped
    Hr_filled = torch.where(torch.isfinite(Hr_filled), Hr_filled, (f_hr_min + f_hr_max) * 0.5)
    Pv_filled = torch.where(torch.isfinite(Pv_filled), Pv_filled, torch.full_like(Pv_filled, pv_prior).clamp(*pv_minmax))

    # ---- Core model (same equations), all finite now ----
    s_HR = ((Hr_filled - f_hr_min) / denom_hr).clamp(eps, 1.0 - eps)

    # p_aset* that makes baro stationary at Pa_filled
    log_ratio = _safe_log_ratio_one_minus_over(s_HR)
    p_aset_star = Pa_filled - (1.0 / k_width) * log_ratio

    # Resistances
    delta_r = (r_tpr_max - r_tpr_min)
    R_base = r_tpr_min + s_HR * delta_r

    R_tpr_try = (R_base + r_tpr_mod_fixed)
    R_tpr = torch.where(torch.isfinite(R_tpr_try), R_tpr_try, torch.full_like(R_tpr_try, 1.0))
    R_tpr = R_tpr.clamp_min(eps)

    # Stroke volume candidate
    Hr_safe = Hr_filled.clamp_min(eps)
    SV_star = (Pa_filled - Pv_filled) / (Hr_safe * R_tpr)

    # Bounds and modulation adjustment
    SV_lo, SV_hi = rng.sv
    rmod_lo, rmod_hi = rng.r_tpr_mod

    within_sv_star = (SV_star >= SV_lo) & (SV_star <= SV_hi)

    SV_clipped = SV_star.clamp(min=SV_lo, max=SV_hi)
    R_needed = (Pa_filled - Pv_filled) / (Hr_safe * SV_clipped.clamp_min(eps))
    rmod_needed = R_needed - R_base
    rmod_final = torch.where(within_sv_star, r_tpr_mod_fixed, rmod_needed)
    rmod_final = rmod_final.clamp(min=rmod_lo, max=rmod_hi)

    R_tpr_final = (R_base + rmod_final).clamp_min(eps)
    SV_final = (Pa_filled - Pv_filled) / (Hr_safe * R_tpr_final)

    # Consistency check
    s_baro = torch.sigmoid(-k_width * (Pa_filled - p_aset_star))
    consistency_error_raw = s_HR - s_baro
    consistency_error_had_nan = ~torch.isfinite(consistency_error_raw)
    consistency_error = _nan_to_num_like(consistency_error_raw)

    # Feasibility flags
    within_sv_bounds = (SV_final >= SV_lo) & (SV_final <= SV_hi)
    within_rmod_bounds = (rmod_final >= rmod_lo) & (rmod_final <= rmod_hi)
    pos_outflow = Pa_filled > Pv_filled
    pos_resistance = R_tpr_final > 0
    feasible = (within_sv_bounds & within_rmod_bounds & pos_outflow & pos_resistance).to(dtype)

    # Optional: return HR in original units as well, if helpful
    Hr_out = Hr_filled if hr_units.lower() == "hz" else Hr_filled * 60.0

    # ---- Final NaN guards (belt & suspenders) ----
    for t in [Pa_filled, Pv_filled, Hr_out, s_HR, p_aset_star, R_base, R_tpr_final, SV_final, rmod_final]:
        if not torch.isfinite(t).all():
            # Last-resort clamp if something slipped (shouldn't happen)
            t = _nan_to_num_like(t)

    return {
        # Inputs after inference (Hr returned in same unit family as input)
        "Pa": Pa_filled,
        "Pv": Pv_filled,
        "Hr": Hr_out,

        # Reflex & setpoint
        "s": s_HR,
        "p_aset_star": p_aset_star,
        "s_baro_at_p_aset_star": s_baro,
        "consistency_error": consistency_error,
        "consistency_error_had_nan": consistency_error_had_nan.to(dtype),

        # Resistances & flows
        "R_base": R_base,
        "R_tpr": R_tpr_final,
        "SV": SV_final,
        "r_tpr_mod": rmod_final,

        # Bounds & status (as tensors for convenience)
        "sv_bounds": torch.tensor(rng.sv, device=device, dtype=dtype),
        "r_tpr_mod_bounds": torch.tensor(rng.r_tpr_mod, device=device, dtype=dtype),
        "within_sv_bounds": within_sv_bounds.to(dtype),
        "within_rmod_bounds": within_rmod_bounds.to(dtype),
        "pos_outflow": pos_outflow.to(dtype),
        "feasible": feasible,

        # Provenance flags
        "Pa_inferred_from_HR_paset": Pa_from_HR_paset,
        "Hr_inferred_from_Pa_paset": Hr_from_Pa_paset,
        "Pv_from_prior": Pv_from_prior,
        "PaHr_joint_fill": PaHr_joint_fill,
        "param_sanitized": param_sanitized,
    }