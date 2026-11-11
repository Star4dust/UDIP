"""
UDIP v3 — modules/data_pulse.py
Proactive anomaly detection and dataset monitoring.
"""

import pandas as pd
import numpy as np
from scipy import stats
from modules.core import log

def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0, sample_size: int = None) -> pd.DataFrame:
    """Detect outliers using Z-score method."""
    try:
        if not pd.api.types.is_numeric_dtype(df[column]):
            return pd.DataFrame()
            
        data = df[column].dropna()
        if sample_size and len(data) > sample_size:
            data = data.sample(sample_size)
            
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores > threshold
        
        # Get the original indices of outliers
        outlier_indices = data.index[outlier_mask]
        return df.loc[outlier_indices]
    except Exception as e:
        log.error(f"Z-score detection failed for {column}: {e}")
        return pd.DataFrame()

def detect_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5, sample_size: int = None) -> pd.DataFrame:
    """Detect outliers using Interquartile Range (IQR) method."""
    try:
        if not pd.api.types.is_numeric_dtype(df[column]):
            return pd.DataFrame()
            
        data = df[column].dropna()
        if sample_size and len(data) > sample_size:
            data = data.sample(sample_size)
            
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return df.loc[outliers.index]
    except Exception as e:
        log.error(f"IQR detection failed for {column}: {e}")
        return pd.DataFrame()

def calculate_dataset_drift(df_old: pd.DataFrame, df_new: pd.DataFrame, column: str) -> dict:
    """
    Calculate statistical drift between two versions of a dataset.
    Returns drift score and status.
    """
    try:
        if column not in df_old.columns or column not in df_new.columns:
            return {"error": "Column missing in one of the datasets"}
            
        if not pd.api.types.is_numeric_dtype(df_old[column]) or not pd.api.types.is_numeric_dtype(df_new[column]):
            # Categorical drift (PSI or simple distribution shift)
            dist_old = df_old[column].value_counts(normalize=True)
            dist_new = df_new[column].value_counts(normalize=True)
            
            # Simple JS Divergence or distribution difference
            all_cats = list(set(dist_old.index) | set(dist_new.index))
            diff = sum(abs(dist_old.get(c, 0) - dist_new.get(c, 0)) for c in all_cats) / 2
            
            return {
                "method": "Categorical Distribution Shift",
                "score": float(diff),
                "is_drifted": diff > 0.2
            }
        else:
            # Numerical drift (KS-Test)
            ks_stat, p_value = stats.ks_2samp(df_old[column].dropna(), df_new[column].dropna())
            
            return {
                "method": "Kolmogorov-Smirnov Test",
                "ks_stat": float(ks_stat),
                "p_value": float(p_value),
                "score": float(ks_stat),
                "is_drifted": p_value < 0.05
            }
    except Exception as e:
        log.error(f"Drift calculation failed for {column}: {e}")
        return {"error": str(e)}

def pulse_check_all(datasets: dict, settings: dict = None) -> list:
    """
    Perform a complete health check on all datasets.
    Returns a list of detected anomalies/alerts.
    """
    alerts = []
    if not datasets: return []
    
    settings = settings or {}
    sample_size = settings.get("sample_size")
    max_cols = settings.get("max_cols", 20)
    
    for ds_id, ds in datasets.items():
        df = ds["df"]
        ds_name = ds.get("name", ds_id)
        
        # Default: check numeric columns for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > max_cols:
            numeric_cols = numeric_cols[:max_cols]
            alerts.append({
                "ds_id": ds_id, "ds_name": ds_name, "type": "Info", "column": "System",
                "severity": "Info",
                "message": f"Dataset has too many columns. Analysis limited to first {max_cols} columns."
            })
        
        for col in numeric_cols:
            outliers = detect_outliers_zscore(df, col, sample_size=sample_size)
            if not outliers.empty:
                alerts.append({
                    "ds_id": ds_id,
                    "ds_name": ds_name,
                    "type": "Anomaly",
                    "column": col,
                    "count": len(outliers),
                    "severity": "High" if len(outliers) < 5 else "Medium",
                    "message": f"Detected {len(outliers)} statistical outliers in '{col}' (sampled)" if sample_size else f"Detected {len(outliers)} statistical outliers in '{col}'"
                })
    
    return alerts
