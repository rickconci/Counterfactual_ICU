import pandas as pd
import numpy as np

def to_utc(s: pd.Series) -> pd.Series:
    """Convert series to UTC datetime."""
    return pd.to_datetime(s, errors="coerce", utc=True)

def norm_label(x: str) -> str:
    """Normalize label string."""
    if x is None or (isinstance(x, float) and np.isnan(x)): 
        return ""
    return str(x).strip()

def med_token(row: pd.Series) -> str:
    """
    Generate medication token with special handling for IV bolus.
    
    Args:
        row: Series containing item_label and input_name
        
    Returns:
        Formatted medication token string
    """
    # Treat IV bolus of NaCl/LR as distinct tokens
    _BOLUS_NAME = "03-IV Fluid Bolus"
    _LR_ALIASES = {"LR"}
    _NACL_ALIASES = {"NaCl 0.9%"}  # extend as needed
    
    base = norm_label(row.get("item_label"))
    iname = str(row.get("input_name", "")).strip()
    if iname == _BOLUS_NAME:
        if base in _NACL_ALIASES:
            return f"{base} [Bolus]"
        if base in _LR_ALIASES:
            return f"{base} [Bolus]"
    return base

def encounter_keys(df: pd.DataFrame) -> list[str]:
    """Get encounter keys from a dataframe."""
    keys = []
    if "subject_id" in df.columns: keys.append("subject_id")
    if "hadm_id" in df.columns: keys.append("hadm_id")
    if not keys: raise KeyError("Neither 'subject_id' nor 'hadm_id' found.")
    return keys
