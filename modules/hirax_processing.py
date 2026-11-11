import numpy as np
from typing import Dict, Any, Optional

def get_validation_metrics(original: np.ndarray, cleaned: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    Compute validation metrics to compare original vs cleaned data.
    """
    orig_abs = np.abs(original)
    clean_abs = np.abs(cleaned)
    
    metrics = {
        "mean_diff": float(np.mean(clean_abs) - np.mean(orig_abs)),
        "std_orig": float(np.std(orig_abs)),
        "std_clean": float(np.std(clean_abs)),
        "var_reduction": float(np.var(orig_abs) - np.var(clean_abs)),
        "signal_preservation": float(100 * (1 - np.sum(mask) / original.size))
    }
    return metrics

def detect_rfi(data: np.ndarray, threshold: float = 3.0, chunk_size: int = 1000, method: str = "mad") -> Dict[str, Any]:
    """
    Detect RFI with chunked processing support for large arrays.
    
    Methods:
    - mad: Median Absolute Deviation (robust sigma estimator)
    - sigma: Standard deviation sigma clipping
    - iqr: Interquartile range
    - percentile: Fixed percentile threshold
    """
    if data.size == 0:
        return {"mask": np.array([]), "count": 0, "percentage": 0.0}

    # If data is small, process normally
    if data.shape[0] <= chunk_size:
        return _detect_rfi_core(data, threshold, method)
    
    # Chunked processing along axis 0 (time)
    masks = []
    for i in range(0, data.shape[0], chunk_size):
        chunk = data[i:i+chunk_size]
        res = _detect_rfi_core(chunk, threshold, method)
        masks.append(res["mask"])
    
    final_mask = np.concatenate(masks, axis=0)
    count = np.sum(final_mask)
    
    return {
        "mask": final_mask,
        "count": int(count),
        "percentage": float((count / data.size) * 100),
        "threshold": threshold,
        "method": method
    }

def _detect_rfi_core(data: np.ndarray, threshold: float, method: str = "mad") -> Dict[str, Any]:
    """
    Core RFI detection with multiple methods.
    """
    abs_data = np.abs(data)
    mask = np.zeros_like(data, dtype=bool)

    def _with_common(result: Dict[str, Any]) -> Dict[str, Any]:
        count = int(np.sum(result["mask"]))
        result.update({
            "count": count,
            "percentage": float((count / data.size) * 100) if data.size else 0.0,
            "threshold": threshold,
            "method": method
        })
        return result
    
    if method == "mad":
        # Median Absolute Deviation (robust sigma estimator)
        median = np.median(abs_data)
        mad = np.median(np.abs(abs_data - median))
        if mad == 0:
            std = np.std(abs_data)
            z_scores = np.abs(abs_data - np.mean(abs_data)) / (std + 1e-10)
        else:
            z_scores = 0.6745 * (abs_data - median) / (mad + 1e-10)
        mask = z_scores > threshold
        return _with_common({"mask": mask, "median": float(median), "mad": float(mad)})
    
    elif method == "sigma":
        # Standard deviation sigma clipping
        mean = np.mean(abs_data)
        std = np.std(abs_data)
        z_scores = np.abs(abs_data - mean) / (std + 1e-10)
        mask = z_scores > threshold
        return _with_common({"mask": mask, "mean": float(mean), "std": float(std)})
    
    elif method == "iqr":
        # Interquartile range
        q25 = np.percentile(abs_data, 25)
        q75 = np.percentile(abs_data, 75)
        iqr = q75 - q25
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        mask = (abs_data < lower_bound) | (abs_data > upper_bound)
        return _with_common({"mask": mask, "q25": float(q25), "q75": float(q75), "iqr": float(iqr)})
    
    elif method == "percentile":
        # Fixed percentile threshold
        upper_threshold = np.percentile(abs_data, 100 - threshold)
        mask = abs_data > upper_threshold
        return _with_common({"mask": mask, "threshold_value": float(upper_threshold)})
    
    else:
        raise ValueError(f"Unknown RFI detection method: {method}")

def sigma_clip(data: np.ndarray, sigma: float = 3.0, maxiters: int = 5, cenfunc: str = "median") -> Dict[str, Any]:
    """
    Iterative sigma clipping for RFI detection.
    
    Args:
        data: Input data array
        sigma: Sigma threshold for clipping
        maxiters: Maximum number of iterations
        cenfunc: Centering function ('median' or 'mean')
    
    Returns:
        dict: Mask and statistics
    """
    abs_data = np.abs(data).flatten()
    mask = np.zeros_like(abs_data, dtype=bool)
    
    for _ in range(maxiters):
        if cenfunc == "median":
            center = np.median(abs_data[~mask])
        else:
            center = np.mean(abs_data[~mask])
        
        std = np.std(abs_data[~mask])
        
        # Find outliers
        new_mask = np.abs(abs_data - center) > (sigma * std)
        
        # Stop if no new outliers found
        if np.array_equal(new_mask, mask):
            break
        
        mask = new_mask
        
        # Stop if too much data is masked
        if np.sum(mask) > 0.9 * len(mask):
            break
    
    # Reshape mask to original data shape
    mask = mask.reshape(data.shape)
    
    return {
        "mask": mask,
        "count": int(np.sum(mask)),
        "percentage": float((np.sum(mask) / data.size) * 100),
        "final_center": float(center),
        "final_std": float(std),
        "iterations": _ + 1
    }

def threshold_mask(data: np.ndarray, threshold: float, mode: str = "absolute") -> np.ndarray:
    """
    Simple threshold masking for RFI detection.
    
    Args:
        data: Input data array
        threshold: Threshold value
        mode: 'absolute' or 'relative' to median
    
    Returns:
        np.ndarray: Boolean mask
    """
    abs_data = np.abs(data)
    
    if mode == "absolute":
        mask = abs_data > threshold
    elif mode == "relative":
        median = np.median(abs_data)
        mask = abs_data > (threshold * median)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return mask

def clean_rfi(data: np.ndarray, mask: np.ndarray, method: str = "interpolate", chunk_size: int = 500) -> np.ndarray:
    """
    Mitigate RFI with optional chunking for interpolation.
    """
    if data.ndim > 2 and method == "interpolate":
        flat_data = data.reshape(data.shape[0], -1)
        flat_mask = mask.reshape(mask.shape[0], -1)
        return _clean_rfi_core(flat_data, flat_mask, method).reshape(data.shape)

    if data.ndim >= 2 and data.shape[1] > chunk_size and method == "interpolate":
        # Process channels in chunks to save memory
        cleaned = data.copy()
        for i in range(0, data.shape[1], chunk_size):
            end = min(i + chunk_size, data.shape[1])
            cleaned[:, i:end] = _clean_rfi_core(data[:, i:end], mask[:, i:end], method)
        return cleaned
    
    return _clean_rfi_core(data, mask, method)

def _clean_rfi_core(data: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    cleaned = data.copy()
    if method == "zero":
        cleaned[mask] = 0
    elif method == "median":
        cleaned[mask] = np.median(data[~mask]) if np.any(~mask) else 0
    else: # interpolate
        if data.ndim == 2:
            for freq_chan in range(data.shape[1]):
                chan_data = data[:, freq_chan]
                chan_mask = mask[:, freq_chan]
                if np.any(chan_mask) and np.any(~chan_mask):
                    indices = np.arange(len(chan_data))
                    cleaned[:, freq_chan] = np.interp(indices, indices[~chan_mask], chan_data[~chan_mask])
        elif data.ndim == 1:
            indices = np.arange(len(data))
            if np.any(mask) and np.any(~mask):
                cleaned = np.interp(indices, indices[~mask], data[~mask])
    return cleaned

def save_rfi_metadata(ds_id: str, metadata: dict):
    """Persistence helper for RFI metadata."""
    from modules.core import DS_DIR, safe_write
    import json
    meta_path = DS_DIR / f"{ds_id}.rfi.json"
    safe_write(meta_path, json.dumps(metadata, indent=2))

def load_rfi_metadata(ds_id: str) -> dict:
    from modules.core import DS_DIR
    import json
    meta_path = DS_DIR / f"{ds_id}.rfi.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def render():
    """Render a practical RFI detection workflow over real loaded datasets."""
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
    from modules.core import add_history, sanitize_varname, save_dataset

    st.markdown('<div class="main-header">RFI Detection</div>', unsafe_allow_html=True)

    if not st.session_state.get("datasets"):
        st.warning("Upload a dataset before running RFI detection.")
        return

    ds_names = {k: v.get("name", k) for k, v in st.session_state.datasets.items()}
    ds_id = st.selectbox(
        "Select dataset",
        list(ds_names.keys()),
        format_func=lambda key: ds_names[key],
        key="rfi_dataset",
    )
    df = st.session_state.datasets[ds_id]["df"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    complex_cols = [c for c in df.columns if np.iscomplexobj(df[c].to_numpy())]
    selectable_cols = complex_cols + [c for c in numeric_cols if c not in complex_cols]

    if not selectable_cols:
        st.info("RFI detection needs numeric or complex-valued columns.")
        return

    default_cols = selectable_cols[: min(64, len(selectable_cols))]
    selected_cols = st.multiselect(
        "Frequency/channel columns",
        selectable_cols,
        default=default_cols,
        help="Rows are treated as time samples; selected columns are treated as channels.",
    )
    if not selected_cols:
        st.info("Select one or more channels to analyze.")
        return

    data = df[selected_cols].to_numpy()
    st.caption(f"Analysis matrix: {data.shape[0]:,} time samples x {data.shape[1]:,} channels")

    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Detection method", ["mad", "sigma", "iqr", "percentile"], index=0)
    threshold = c2.slider("Threshold", 1.0, 10.0, 3.5, 0.1)
    clean_method = c3.selectbox("Mitigation", ["interpolate", "median", "zero"], index=0)

    manual_channels = st.multiselect(
        "Manually mask channels",
        list(range(data.shape[1])),
        format_func=lambda i: str(selected_cols[i]),
        help="Optional manual mask for persistently contaminated channels.",
    )

    if st.button("Run RFI Analysis", type="primary"):
        result = detect_rfi(data, threshold=threshold, method=method)
        mask = result["mask"].copy()
        if manual_channels:
            mask[:, manual_channels] = True
            result["mask"] = mask
            result["count"] = int(np.sum(mask))
            result["percentage"] = float((np.sum(mask) / mask.size) * 100)

        cleaned = clean_rfi(data, mask, method=clean_method)
        st.session_state["rfi_last_result"] = result
        st.session_state["rfi_last_cleaned"] = cleaned
        st.session_state["rfi_last_columns"] = selected_cols
        add_history("RFI", f"{ds_names[ds_id]} masked {result['percentage']:.2f}%")

    result = st.session_state.get("rfi_last_result")
    cleaned = st.session_state.get("rfi_last_cleaned")
    last_cols = st.session_state.get("rfi_last_columns", selected_cols)

    if result is None:
        return

    st.success(f"Masked {result['count']:,} samples ({result['percentage']:.2f}%).")
    tabs = st.tabs(["Waterfall", "Mask", "Cleaned", "Metadata"])

    with tabs[0]:
        fig = go.Figure(go.Heatmap(z=np.log10(np.abs(data) + 1e-12), colorscale="Viridis"))
        fig.update_layout(xaxis_title="Channel", yaxis_title="Time sample", height=480)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        fig = go.Figure(go.Heatmap(z=result["mask"].astype(int), colorscale="Reds", showscale=False))
        fig.update_layout(xaxis_title="Channel", yaxis_title="Time sample", height=480)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        if cleaned is not None:
            fig = go.Figure(go.Heatmap(z=np.log10(np.abs(cleaned) + 1e-12), colorscale="Viridis"))
            fig.update_layout(xaxis_title="Channel", yaxis_title="Time sample", height=480)
            st.plotly_chart(fig, use_container_width=True)

            if st.button("Save Cleaned Dataset"):
                out_df = df.copy()
                out_df[last_cols] = cleaned
                out_name = f"{ds_names[ds_id]}_rfi_cleaned"
                out_id = sanitize_varname(out_name)
                save_dataset(out_id, {
                    "name": out_name,
                    "df": pd.DataFrame(out_df),
                    "format": st.session_state.datasets[ds_id].get("format", "processed"),
                    "source": st.session_state.datasets[ds_id].get("source", ds_names[ds_id]),
                    "metadata": {
                        "rfi_method": result.get("method"),
                        "rfi_threshold": result.get("threshold"),
                        "masked_percentage": result.get("percentage"),
                    },
                })
                st.session_state.datasets[out_id] = {
                    "name": out_name,
                    "df": pd.DataFrame(out_df),
                    "format": st.session_state.datasets[ds_id].get("format", "processed"),
                    "source": st.session_state.datasets[ds_id].get("source", ds_names[ds_id]),
                    "metadata": {
                        "derived_from": ds_id,
                        "rfi_method": result.get("method"),
                        "rfi_threshold": result.get("threshold"),
                        "masked_percentage": result.get("percentage"),
                    },
                }
                st.success(f"Saved {out_name}")

    with tabs[3]:
        st.json({k: v for k, v in result.items() if k != "mask"})
