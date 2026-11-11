import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import h5py
import os
import plotly.graph_objects as go
from pathlib import Path
from modules.core import (
    add_history, sanitize_varname, save_dataset, DATA_DIR
)
from modules import hirax_processing, hirax_analysis, experiment_tracker, observation_session, observatory_intelligence
from modules.dataset_selector import render_dataset_selector

# Ensure raw_hdf5 directory exists
RAW_H5_DIR = DATA_DIR / "raw_hdf5"
RAW_H5_DIR.mkdir(exist_ok=True, parents=True)


def _observatory_cache_key(session_records):
    payload = []
    for rec in session_records:
        payload.append(
            (
                getattr(rec, "file_name", ""),
                getattr(rec, "dataset_path", ""),
                getattr(rec, "time_order_key", None),
                tuple(getattr(rec, "visibility_shape", ())),
                tuple(getattr(rec, "polarization_labels", ())),
                tuple(getattr(rec, "baseline_labels", ())),
            )
        )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()

def render():
    st.markdown('<div class="main-header">🔭 HIRAX Data Inspector</div>', unsafe_allow_html=True)
    st.markdown("Specialized HDF5 explorer for HIRAX scientific data. Analyze dimensions, inspect metadata, and visualize waterfall plots.")

    # ── Dataset Selection (Session State) ───────────────────────────────────────
    st.markdown("---")
    selected_dataset = render_dataset_selector(workflow_type="hirax")
    
    if selected_dataset:
        st.session_state.hirax_selected_dataset = selected_dataset
        st.success(f"✅ Selected dataset: {selected_dataset}")
    
    st.markdown("---")

    # ── File Selection ────────────────────────────────────────────────────────
    st.sidebar.markdown("### 📁 Data Source")
    source_type = st.sidebar.radio("Source", ["Upload New HDF5", "Previously Uploaded"], key="hirax_source_type")

    h5_path = None

    if source_type == "Upload New HDF5":
        uploaded = st.sidebar.file_uploader("Upload HDF5 (.h5, .hdf5)", type=["h5", "hdf5"])
        if uploaded:
            # Save to raw_hdf5 dir for persistent access
            h5_path = RAW_H5_DIR / uploaded.name
            with open(h5_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.sidebar.success(f"Loaded: {uploaded.name}")
    else:
        # List files in RAW_H5_DIR
        h5_files = list(RAW_H5_DIR.glob("*.h5")) + list(RAW_H5_DIR.glob("*.hdf5"))
        if not h5_files:
            st.sidebar.info("No previously uploaded HDF5 files found.")
        else:
            selected_file = st.sidebar.selectbox("Select File", h5_files, format_func=lambda x: x.name)
            if selected_file:
                h5_path = selected_file

    if not h5_path:
        st.info("Please upload or select an HDF5 file to begin inspection.")
        # Show some helpful info about HIRAX data structures
        with st.expander("ℹ️ About HIRAX Data Structure", expanded=True):
            st.markdown("""
            HIRAX (Hydrogen Intensity and Real-time Analysis eXperiment) typically stores data in HDF5 with:
            - **vis**: Visibility data (often 4D: time, freq, baseline, polarization)
            - **gain**: Calibration gains
            - **freq**: Frequency channel information
            - **time**: Timestamp information
            - **index**: Mapping indices
            """)
        return

    # ── Inspection ────────────────────────────────────────────────────────────
    try:
        with h5py.File(h5_path, 'r') as f:
            # 1. Hierarchical Browser
            st.markdown(f"### 📂 File Structure: `{h5_path.name}`")
            
            def get_structure(group, prefix=""):
                items = []
                for k in group.keys():
                    path = f"{prefix}/{k}" if prefix else k
                    obj = group[k]
                    if isinstance(obj, h5py.Group):
                        items.append({"path": path, "type": "Group", "shape": "-", "dtype": "-"})
                        items.extend(get_structure(obj, path))
                    else:
                        items.append({
                            "path": path, 
                            "type": "Dataset", 
                            "shape": str(obj.shape), 
                            "dtype": str(obj.dtype)
                        })
                return items

            structure = get_structure(f)
            df_struct = pd.DataFrame(structure)
            
            # Show structure in an expander or table
            with st.expander("View Full Hierarchy", expanded=False):
                st.dataframe(df_struct, use_container_width=True)

            # 2. Dataset Selector
            dataset_paths = [item["path"] for item in structure if item["type"] == "Dataset"]
            selected_ds_path = st.selectbox("Select Dataset to Inspect", dataset_paths)

            if selected_ds_path:
                ds = f[selected_ds_path]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("#### Metadata")
                    st.write(f"**Shape:** {ds.shape}")
                    st.write(f"**Dtype:** {ds.dtype}")
                    if ds.attrs:
                        st.write("**Attributes:**")
                        for k, v in ds.attrs.items():
                            st.write(f"- `{k}`: {v}")
                    else:
                        st.write("*No attributes found*")

                with col2:
                    st.markdown("#### Dimensional Analysis")
                    dims = list(ds.shape)
                    st.write(f"Rank: {len(dims)}")
                    
                    # Heuristic for HIRAX dimensions
                    dim_labels = []
                    for i, d in enumerate(dims):
                        label = f"Dim {i}"
                        if d > 500 and d < 2048: # Often frequency
                            label = f"Dim {i} (Freq Channels)"
                        elif d > 1000: # Often time
                            label = f"Dim {i} (Time Samples)"
                        elif d == 2 or d == 4: # Often polarization
                            label = f"Dim {i} (Polarization)"
                        elif d > 4 and d < 500: # Often baselines
                            label = f"Dim {i} (Interferometric Baselines)"
                        dim_labels.append(label)
                    
                    st.write(dim_labels)
                    
                    # One-Click Full Analysis Shortcut
                    st.divider()
                    if st.button("🚀 Run Full HIRAX Analysis", type="primary", use_container_width=True):
                        st.session_state["_hirax_run_full_analysis"] = True
                        st.info("Choose or confirm a slice below. The analysis will run on that real data slice.")

                st.divider()

                # 3. Slicing & Preview
                st.markdown("#### 🔪 Slice Preview")
                
                slice_input = st.text_input("Enter Slice (Python syntax)", value="0, :, :", help="Example: 0, :, : or :, 10:20, :")
                
                try:
                    # Evaluate slice string safely
                    # We convert something like "0, :, :" to (0, slice(None), slice(None))
                    def parse_slice(s):
                        parts = s.split(",")
                        parsed = []
                        for p in parts:
                            p = p.strip()
                            if ":" in p:
                                sub_parts = p.split(":")
                                start = int(sub_parts[0]) if sub_parts[0] else None
                                stop = int(sub_parts[1]) if sub_parts[1] else None
                                step = int(sub_parts[2]) if len(sub_parts) > 2 else None
                                parsed.append(slice(start, stop, step))
                            else:
                                parsed.append(int(p))
                        return tuple(parsed)

                    s_obj = parse_slice(slice_input)
                    data_slice = ds[s_obj]
                    
                    st.write(f"Previewing slice shape: **{data_slice.shape}**")

                    if st.session_state.pop("_hirax_run_full_analysis", False):
                        with st.spinner("Executing HIRAX analysis on selected slice..."):
                            r_res = hirax_processing.detect_rfi(data_slice, threshold=3.5)
                            c_data = hirax_processing.clean_rfi(data_slice, r_res["mask"], method="interpolate")
                            cal_data = hirax_analysis.calibrate_signal(c_data, window_size=50)
                            p_res = hirax_analysis.compute_power_spectrum(cal_data, sampling_rate=1.0)

                            st.session_state["hirax_cleaned_data"] = c_data
                            st.session_state["hirax_calibrated_data"] = cal_data
                            st.session_state["hirax_power_spectrum"] = p_res
                            st.session_state["hirax_rfi_mask"] = r_res["mask"]

                            st.success("HIRAX analysis complete for the selected slice.")

                    if data_slice.ndim == 2:
                        # Offer Waterfall Plot
                        st.markdown("##### 🌊 Waterfall Plot")
                        
                        # Controls
                        c1, c2, c3 = st.columns(3)
                        cmap = c1.selectbox("Color Scale", ["Viridis", "Magma", "Inferno", "Plasma", "Cividis", "Blues", "Hot"])
                        log_scale = c2.checkbox("Log Scale Intensity", value=True)
                        show_grid = c3.checkbox("Show Grid", value=False)

                        plot_data = data_slice
                        if log_scale:
                            # Avoid log(0)
                            plot_data = np.log10(np.abs(plot_data) + 1e-10)

                        fig = go.Figure(data=go.Heatmap(
                            z=plot_data,
                            colorscale=cmap,
                            colorbar=dict(title="Intensity (log)" if log_scale else "Intensity")
                        ))
                        
                        fig.update_layout(
                            title=f"Waterfall: {selected_ds_path} [{slice_input}]",
                            xaxis_title="Time / Sample",
                            yaxis_title="Frequency / Channel",
                            template="plotly_dark" if st.session_state.get("dark_mode") else "plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

                        # ── RFI Analysis & Mitigation ─────────────────────────
                        st.markdown("#### 🛡️ RFI Analysis & Mitigation")
                        
                        rc1, rc2, rc3 = st.columns(3)
                        rfi_threshold = rc1.slider("RFI Threshold (z-score)", 1.0, 10.0, 3.5, 0.1)
                        detect_btn = rc2.button("🔍 Detect RFI", use_container_width=True)
                        clean_btn = rc3.button("✨ Clean Data", use_container_width=True)

                        # Detect RFI
                        if detect_btn:
                            with st.spinner("Analyzing signals for interference..."):
                                results = hirax_processing.detect_rfi(data_slice, threshold=rfi_threshold)
                                st.session_state["hirax_rfi_results"] = results
                                st.session_state["hirax_rfi_mask"] = results["mask"]
                                st.success(f"RFI Detection Complete: **{results['percentage']:.2f}%** of data affected.")
                        
                        # Clean Data
                        if clean_btn:
                            mask = st.session_state.get("hirax_rfi_mask")
                            if mask is None:
                                with st.spinner("Detecting RFI before cleaning..."):
                                    results = hirax_processing.detect_rfi(data_slice, threshold=rfi_threshold)
                                    mask = results["mask"]
                                    st.session_state["hirax_rfi_mask"] = mask

                            with st.spinner("Applying mitigation algorithms..."):
                                cleaned = hirax_processing.clean_rfi(data_slice, mask, method="interpolate")
                                metrics = hirax_processing.get_validation_metrics(data_slice, cleaned, mask)
                                st.session_state["hirax_cleaned_data"] = cleaned
                                st.session_state["hirax_rfi_metrics"] = metrics
                                
                                # Persist metadata
                                ds_id = sanitize_varname(f"{h5_path.stem}_{selected_ds_path.replace('/', '_')}")
                                hirax_processing.save_rfi_metadata(ds_id, {
                                    "threshold": rfi_threshold,
                                    "metrics": metrics,
                                    "slice": slice_input
                                })
                                st.success("Data Mitigation Applied & Results Persisted.")

                        # Comparative Visualization
                        if "hirax_rfi_mask" in st.session_state:
                            with st.expander("📊 RFI Diagnostics & Comparison", expanded=True):
                                diag_tabs = st.tabs(["Mask Overlay", "Before vs After", "Stats & Metrics"])
                                
                                with diag_tabs[0]:
                                    st.markdown("**RFI Mask Overlay** (Red pixels indicate interference)")
                                    mask_plot = np.zeros_like(data_slice, dtype=float)
                                    mask_plot[st.session_state["hirax_rfi_mask"]] = 1.0
                                    
                                    fig_mask = go.Figure(data=go.Heatmap(
                                        z=mask_plot,
                                        colorscale=[[0, "rgba(0,0,0,0)"], [1, "red"]],
                                        showscale=False
                                    ))
                                    fig_mask.update_layout(title="RFI Mask", template="plotly_dark")
                                    st.plotly_chart(fig_mask, use_container_width=True)

                                with diag_tabs[1]:
                                    if "hirax_cleaned_data" in st.session_state:
                                        st.markdown("**Before vs After Cleaning**")
                                        c_left, c_right = st.columns(2)
                                        
                                        with c_left:
                                            st.caption("Original (Dirty)")
                                            plot_orig = np.log10(np.abs(data_slice) + 1e-10) if log_scale else np.abs(data_slice)
                                            st.plotly_chart(go.Figure(go.Heatmap(z=plot_orig, colorscale=cmap, showscale=False)).update_layout(height=400, margin=dict(t=0,b=0,l=0,r=0)), use_container_width=True)
                                        
                                        with c_right:
                                            st.caption("Cleaned (Interpolated)")
                                            plot_clean = np.log10(np.abs(st.session_state["hirax_cleaned_data"]) + 1e-10) if log_scale else np.abs(st.session_state["hirax_cleaned_data"])
                                            st.plotly_chart(go.Figure(go.Heatmap(z=plot_clean, colorscale=cmap, showscale=False)).update_layout(height=400, margin=dict(t=0,b=0,l=0,r=0)), use_container_width=True)
                                    else:
                                        st.info("Clean data to see comparison.")

                                with diag_tabs[2]:
                                    results = st.session_state.get("hirax_rfi_results", {})
                                    metrics = st.session_state.get("hirax_rfi_metrics", {})
                                    if results:
                                        st.markdown("##### Detection Stats")
                                        sc1, sc2, sc3 = st.columns(3)
                                        sc1.metric("Interference Pixels", f"{results['count']:,}")
                                        sc2.metric("Data Contamination", f"{results['percentage']:.2f}%")
                                        sc3.metric("Threshold Used", f"{results.get('threshold', 3.5)}σ")
                                        
                                    if metrics:
                                        st.divider()
                                        st.markdown("##### Validation Metrics")
                                        m1, m2, m3 = st.columns(3)
                                        m1.metric("Variance Reduction", f"{metrics['var_reduction']:.4f}")
                                        m2.metric("Signal Preservation", f"{metrics['signal_preservation']:.2f}%")
                                        m3.metric("Noise level (Std Clean)", f"{metrics['std_clean']:.4f}")
                        
                        st.line_chart(data_slice)
                    else:
                        st.warning(f"Slice has {data_slice.ndim} dimensions. Waterfall plot requires 2D. Please adjust slice.")
                        st.write("First few values:")
                        st.write(data_slice[:5])

                    # 4. Export to UDIP
                    st.divider()
                    st.markdown("#### 📥 UDIP Integration")
                    
                    data_to_import = data_slice
                    import_label = "Original"
                    
                    if "hirax_cleaned_data" in st.session_state:
                        version = st.radio("Version to Import", ["Original", "Cleaned (RFI Mitigated)"], horizontal=True)
                        if version == "Cleaned (RFI Mitigated)":
                            data_to_import = st.session_state["hirax_cleaned_data"]
                            import_label = "Cleaned"

                    if st.button("📥 Import to UDIP Workspace", type="primary", use_container_width=True):
                        if data_to_import.ndim > 2:
                            # Flatten for DataFrame
                            flat_data = data_to_import.reshape(data_to_import.shape[0], -1)
                            df_new = pd.DataFrame(flat_data)
                        elif data_to_import.ndim == 2:
                            df_new = pd.DataFrame(data_to_import)
                        else:
                            df_new = pd.DataFrame({"value": data_to_import})
                        
                        new_name = f"{h5_path.stem}_{selected_ds_path.replace('/', '_')}_{import_label.lower()}_slice"
                        ds_id = sanitize_varname(new_name)
                        
                        save_dataset(ds_id, {
                            "name": new_name,
                            "df": df_new,
                            "format": "hdf5_slice",
                            "source_h5": str(h5_path),
                            "slice": slice_input,
                            "version": import_label
                        })
                        add_history("HIRAX", f"Imported {import_label.lower()} slice from {h5_path.name}")
                        st.success(f"Imported as dataset: {new_name}")
                        st.balloons()

                    # ── Signal Analysis ──────────────────────────────────────
                    st.divider()
                    st.markdown("#### 📉 Signal Analysis (FFT & Power)")
                    
                    ac1, ac2 = st.columns(2)
                    sampling_rate = ac1.number_input("Sampling Rate (Hz)", value=1.0, step=0.1)
                    drift_window = ac2.slider("Calibration Window", 10, 200, 50)
                    
                    ana_tabs = st.tabs(["Calibration", "FFT Spectrum", "Power Spectrum"])
                    
                    with ana_tabs[0]:
                        st.markdown("**Signal Calibration** (Drift Removal & Normalization)")
                        if st.button("🚀 Calibrate Signal", use_container_width=True):
                            with st.spinner("Calibrating..."):
                                data_to_ana = st.session_state.get("hirax_cleaned_data", data_slice)
                                calibrated = hirax_analysis.calibrate_signal(data_to_ana, window_size=drift_window)
                                st.session_state["hirax_calibrated_data"] = calibrated
                                st.success("Signal Calibrated (Baseline drift removed).")
                        
                        if "hirax_calibrated_data" in st.session_state:
                            st.caption("Calibrated Waterfall (Normalized)")
                            fig_cal = go.Figure(data=go.Heatmap(
                                z=st.session_state["hirax_calibrated_data"],
                                colorscale=cmap
                            ))
                            fig_cal.update_layout(height=400, template="plotly_dark")
                            st.plotly_chart(fig_cal, use_container_width=True)

                    with ana_tabs[1]:
                        st.markdown("**FFT Magnitude Spectrum**")
                        if st.button("📈 Compute FFT", use_container_width=True):
                            data_to_ana = st.session_state.get("hirax_calibrated_data", 
                                          st.session_state.get("hirax_cleaned_data", data_slice))
                            with st.spinner("Computing FFT..."):
                                fft_res = hirax_analysis.compute_fft(data_to_ana, sampling_rate=sampling_rate)
                                st.session_state["hirax_fft_result"] = fft_res
                        
                        if "hirax_fft_result" in st.session_state:
                            fft_res = st.session_state["hirax_fft_result"]
                            fig_fft = go.Figure()
                            # If 2D, plot mean magnitude
                            mag = np.mean(fft_res["magnitude"], axis=1) if fft_res["magnitude"].ndim == 2 else fft_res["magnitude"]
                            fig_fft.add_trace(go.Scatter(x=fft_res["frequencies"], y=mag, name="Magnitude"))
                            fig_fft.update_layout(title="FFT Magnitude Spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", template="plotly_dark")
                            st.plotly_chart(fig_fft, use_container_width=True)

                    with ana_tabs[2]:
                        st.markdown("**Power Spectrum Density**")
                        if st.button("🔥 Compute Power Spectrum", use_container_width=True):
                            data_to_ana = st.session_state.get("hirax_calibrated_data", 
                                          st.session_state.get("hirax_cleaned_data", data_slice))
                            with st.spinner("Computing Power..."):
                                p_res = hirax_analysis.compute_power_spectrum(data_to_ana, sampling_rate=sampling_rate)
                                st.session_state["hirax_power_spectrum"] = p_res
                        
                        if "hirax_power_spectrum" in st.session_state:
                            p_res = st.session_state["hirax_power_spectrum"]
                            fig_p = go.Figure()
                            fig_p.add_trace(go.Scatter(x=p_res["frequencies"], y=p_res["power_avg"], name="Power", line=dict(color="orange")))
                            fig_p.update_layout(title="Power Spectrum Density", xaxis_title="Frequency (Hz)", yaxis_title="Power", yaxis_type="log", template="plotly_dark")
                            st.plotly_chart(fig_p, use_container_width=True)
                            
                            # Before vs After Comparison if RFI cleaned
                            if "hirax_cleaned_data" in st.session_state:
                                with st.expander("🔄 Compare Dirty vs Cleaned Spectrum"):
                                    with st.spinner("Comparing..."):
                                        p_dirty = hirax_analysis.compute_power_spectrum(data_slice, sampling_rate=sampling_rate)
                                        fig_comp = go.Figure()
                                        fig_comp.add_trace(go.Scatter(x=p_dirty["frequencies"], y=p_dirty["power_avg"], name="Dirty (Original)", line=dict(color="red", dash="dash")))
                                        fig_comp.add_trace(go.Scatter(x=p_res["frequencies"], y=p_res["power_avg"], name="Cleaned", line=dict(color="green")))
                                        fig_comp.update_layout(title="Power Spectrum Comparison", xaxis_title="Frequency (Hz)", yaxis_title="Power (log)", yaxis_type="log", template="plotly_dark")
                                        st.plotly_chart(fig_comp, use_container_width=True)

                    # ── Interferometry Analysis ───────────────────────────────
                    st.divider()
                    st.markdown("#### 🔭 Interferometry Analysis")
                    
                    inter_tabs = st.tabs(["Visibility Maps", "Baseline Matrix", "UV Coverage", "Polarization"])
                    
                    with inter_tabs[0]:
                        st.markdown("**Visibility Amplitude & Phase**")
                        st.caption("Interferometric visibilities are complex numbers with amplitude and phase.")
                        
                        # Check if data is complex
                        if np.iscomplexobj(data_slice):
                            amp = np.abs(data_slice)
                            phase = np.angle(data_slice)
                            
                            viz_c1, viz_c2 = st.columns(2)
                            
                            with viz_c1:
                                st.markdown("##### Amplitude Map")
                                fig_amp = go.Figure(data=go.Heatmap(z=amp, colorscale="Viridis"))
                                fig_amp.update_layout(title="Visibility Amplitude", template="plotly_dark")
                                st.plotly_chart(fig_amp, use_container_width=True)
                            
                            with viz_c2:
                                st.markdown("##### Phase Map (radians)")
                                fig_phase = go.Figure(data=go.Heatmap(z=phase, colorscale="Cividis"))
                                fig_phase.update_layout(title="Visibility Phase", template="plotly_dark")
                                st.plotly_chart(fig_phase, use_container_width=True)
                            
                            # Statistics
                            st.metric("Mean Amplitude", f"{np.mean(amp):.4f}")
                            st.metric("Phase Range", f"{np.min(phase):.2f} to {np.max(phase):.2f} rad")
                        else:
                            st.info("Current data is real-valued. Visibility analysis requires complex-valued data.")
                            st.caption("Tip: Select a visibility dataset (e.g., 'vis' group) from the HDF5 structure.")
                    
                    with inter_tabs[1]:
                        st.markdown("**Baseline Correlation Matrix**")
                        st.caption("Shows correlation between different interferometric baselines.")
                        
                        # If data has baseline dimension, show baseline matrix
                        if data_slice.ndim >= 2:
                            # Compute correlation matrix across one dimension
                            if data_slice.ndim == 2:
                                corr_matrix = np.corrcoef(data_slice)
                            else:
                                # For higher dimensions, take mean across other dims
                                flat_data = data_slice.reshape(data_slice.shape[0], -1)
                                corr_matrix = np.corrcoef(flat_data)
                            
                            fig_corr = go.Figure(data=go.Heatmap(
                                z=corr_matrix,
                                colorscale="RdBu",
                                zmid=0,
                                colorbar=dict(title="Correlation")
                            ))
                            fig_corr.update_layout(
                                title="Baseline Correlation Matrix",
                                template="plotly_dark",
                                height=500
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            st.metric("Mean Correlation", f"{np.mean(np.abs(corr_matrix)):.4f}")
                        else:
                            st.info("Baseline matrix requires multi-dimensional data.")
                    
                    with inter_tabs[2]:
                        st.markdown("**UV-Plane Coverage**")
                        st.caption("Spatial frequency sampling pattern of the interferometer.")
                        
                        uv_candidates = [
                            item["path"] for item in structure
                            if item["type"] == "Dataset"
                            and any(token in item["path"].lower() for token in ("uvw", "uv", "baseline"))
                        ]

                        u_coords = v_coords = None
                        selected_uv_path = None
                        for uv_path in uv_candidates:
                            arr = np.asarray(f[uv_path][()])
                            if arr.dtype.names and {"u", "v"}.issubset(arr.dtype.names):
                                u_coords = np.asarray(arr["u"]).ravel()
                                v_coords = np.asarray(arr["v"]).ravel()
                                selected_uv_path = uv_path
                                break
                            if arr.ndim >= 2 and arr.shape[-1] >= 2:
                                flat_uv = arr.reshape(-1, arr.shape[-1])
                                u_coords = flat_uv[:, 0]
                                v_coords = flat_uv[:, 1]
                                selected_uv_path = uv_path
                                break

                        if u_coords is None or v_coords is None:
                            st.info("No UV coordinate dataset was found in this file. UV coverage needs real u/v or uvw coordinates from the observation metadata.")
                        else:
                            fig_uv = go.Figure()
                            fig_uv.add_trace(go.Scatter(
                                x=u_coords,
                                y=v_coords,
                                mode='markers',
                                marker=dict(size=7, color='cyan', line=dict(color='white', width=1)),
                                name='Baselines'
                            ))
                            fig_uv.update_layout(
                                title=f"UV-Plane Coverage: {selected_uv_path}",
                                xaxis_title="u (wavelengths)",
                                yaxis_title="v (wavelengths)",
                                template="plotly_dark",
                                height=500,
                                width=500
                            )
                            st.plotly_chart(fig_uv, use_container_width=True)

                            col_u1, col_u2 = st.columns(2)
                            col_u1.metric("Max u coordinate", f"{np.max(np.abs(u_coords)):.1f}")
                            col_u2.metric("Max v coordinate", f"{np.max(np.abs(v_coords)):.1f}")
                    
                    with inter_tabs[3]:
                        st.markdown("**Polarization Analysis**")
                        st.caption("Analysis of different polarization states (XX, XY, YX, YY).")
                        
                        # Check if data has polarization dimension
                        pol_dims = [d for d, label in enumerate(dims) if "Polarization" in str(label)]
                        
                        if pol_dims:
                            st.success(f"✅ Polarization dimension detected at axis {pol_dims[0]}")
                            st.write(f"Dimension size: {dims[pol_dims[0]]}")
                            
                            # If polarization dimension exists, allow selection
                            if dims[pol_dims[0]] >= 2:
                                pol_indices = list(range(dims[pol_dims[0]]))
                                pol_names = ["XX", "XY", "YX", "YY"][:dims[pol_dims[0]]]
                                
                                selected_pol = st.selectbox("Select Polarization", pol_indices, format_func=lambda i: pol_names[i])
                                
                                # Slice to show selected polarization
                                slice_pol = list(s_obj)
                                slice_pol[pol_dims[0]] = selected_pol
                                pol_data = ds[tuple(slice_pol)]
                                
                                st.markdown(f"##### Polarization {pol_names[selected_pol]}")
                                if pol_data.ndim >= 2:
                                    fig_pol = go.Figure(data=go.Heatmap(z=np.abs(pol_data), colorscale="Plasma"))
                                    fig_pol.update_layout(title=f"{pol_names[selected_pol]} Polarization", template="plotly_dark")
                                    st.plotly_chart(fig_pol, use_container_width=True)
                                else:
                                    st.line_chart(pol_data)
                        else:
                            st.info("No polarization dimension detected in current data.")
                            st.caption("HIRAX data typically includes polarization states XX, XY, YX, YY.")

                    # ── Signal Intelligence ──────────────────────────────────
                    st.divider()
                    st.markdown("#### 🔬 Signal Intelligence & Interpretation")
                    
                    ic1, ic2 = st.columns(2)
                    win_type = ic1.selectbox("Spectral Window", ["None", "Hann", "Hamming", "Blackman"])
                    peak_thresh = ic2.slider("Peak Threshold", 0.0, 1.0, 0.1)
                    
                    intel_tabs = st.tabs(["Interpretation", "Peak Registry", "Annotated Spectrum"])
                    
                    # Pre-processing for Intelligence
                    data_for_intel = st.session_state.get("hirax_calibrated_data", 
                                     st.session_state.get("hirax_cleaned_data", data_slice))
                    
                    if win_type != "None":
                        data_for_intel = hirax_analysis.apply_window(data_for_intel, window_type=win_type.lower())
                    
                    # Always re-compute for the latest settings
                    p_res_intel = hirax_analysis.compute_power_spectrum(data_for_intel, sampling_rate=sampling_rate)
                    peaks_res = hirax_analysis.detect_peaks(p_res_intel["power_avg"], p_res_intel["frequencies"], threshold=peak_thresh)
                    classification = hirax_analysis.classify_signal(p_res_intel["power_avg"])
                    
                    with intel_tabs[0]:
                        st.markdown("**Automated Interpretation**")
                        c_color = {"noise": "gray", "structured_signal": "green", "interference": "red"}.get(classification, "white")
                        st.markdown(f"Signal Class: <span style='color:{c_color}; font-weight:bold; font-size:24px;'>{classification['label'].upper().replace('_', ' ')}</span>", unsafe_allow_html=True)
                        st.progress(classification["confidence"], text=f"Confidence: {classification['confidence']*100:.1f}%")
                        
                        if classification["label"] == "interference":
                            st.error("⚠️ High Peak-to-Average Power Ratio (PAPR) detected. Signal likely contaminated by RFI.")
                        elif classification["label"] == "structured_signal":
                            st.success("✅ Structured signal features detected. High probability of astronomical sources or calibration markers.")
                        else:
                            st.info("ℹ️ Signal profile matches Gaussian noise. No strong narrowband features detected.")
                            
                        st.metric("Detected Spectral Peaks", peaks_res["count"])

                    with intel_tabs[1]:
                        st.markdown("**Detected Frequency Peaks**")
                        if peaks_res["peaks"]:
                            df_peaks = pd.DataFrame(peaks_res["peaks"])[["frequency", "power"]]
                            st.dataframe(df_peaks, use_container_width=True)
                        else:
                            st.write("No peaks detected with current threshold.")

                    with intel_tabs[2]:
                        st.markdown("**Annotated Power Spectrum**")
                        fig_intel = go.Figure()
                        fig_intel.add_trace(go.Scatter(x=p_res_intel["frequencies"], y=p_res_intel["power_avg"], name="Power", line=dict(color="cyan")))
                        
                        # Add peak markers
                        if peaks_res["peaks"]:
                            pk_freqs = [p["frequency"] for p in peaks_res["peaks"]]
                            pk_powers = [p["power"] for p in peaks_res["peaks"]]
                            fig_intel.add_trace(go.Scatter(
                                x=pk_freqs, y=pk_powers, 
                                mode='markers', name='Detected Peaks',
                                marker=dict(color='red', size=10, symbol='x')
                            ))
                            
                        fig_intel.update_layout(
                            xaxis_title="Frequency (Hz)", yaxis_title="Power (log)", 
                            yaxis_type="log", template="plotly_dark",
                            showlegend=True
                        )
                        st.plotly_chart(fig_intel, use_container_width=True)

                    # ── Research Workbench ───────────────────────────────────
                    st.divider()
                    st.markdown("#### 🔬 Research Workbench")
                    
                    work_tabs = st.tabs(["Export Data", "Batch Analysis", "Experiment History"])
                    
                    with work_tabs[0]:
                        st.markdown("**Scientific Export Tools**")
                        ex1, ex2 = st.columns(2)
                        if ex1.button("📊 Export Peaks (CSV)", use_container_width=True):
                            if peaks_res["peaks"]:
                                df_p = pd.DataFrame(peaks_res["peaks"])
                                csv_p = df_p.to_csv(index=False)
                                st.download_button("Download Peaks CSV", csv_p, f"{h5_path.stem}_peaks.csv", "text/csv")
                        
                        if ex2.button("📈 Export Spectrum (NumPy)", use_container_width=True):
                            # In streamlit we use bytes for download
                            import io
                            buf = io.BytesIO()
                            np.save(buf, p_res_intel["power_avg"])
                            st.download_button("Download Spectrum .npy", buf.getvalue(), f"{h5_path.stem}_spectrum.npy")

                    with work_tabs[1]:
                        st.markdown("**Batch Multi-File Analysis**")
                        batch_files = st.multiselect("Select HDF5 Files for Batch", 
                                                    list(RAW_H5_DIR.glob("*.h5")) + list(RAW_H5_DIR.glob("*.hdf5")),
                                                    format_func=lambda x: x.name)
                        if st.button("🚀 Run Batch Processing", type="primary", use_container_width=True):
                            if not batch_files:
                                st.warning("Please select at least one file.")
                            else:
                                with st.spinner(f"Batch processing {len(batch_files)} files..."):
                                    # Create a simple template
                                    template = {
                                        "execution_plan": [
                                            {"action": "load_dataset", "file_path": ""},
                                            {"action": "detect_rfi", "threshold": rfi_threshold},
                                            {"action": "clean_rfi", "method": "interpolate"},
                                            {"action": "power_spectrum", "sampling_rate": sampling_rate},
                                            {"action": "detect_peaks", "threshold": peak_thresh}
                                        ]
                                    }
                                    from modules.pipeline_runner import run_batch_pipeline
                                    agg_res = run_batch_pipeline([str(f) for f in batch_files], template)
                                    st.success(f"Batch analysis of {agg_res['total_files']} files complete.")
                                    
                                    # Plot aggregate spectrum
                                    if agg_res["average_power"].size > 0:
                                        st.markdown("##### 🌐 Aggregate Average Power Spectrum")
                                        fig_agg = go.Figure(go.Scatter(y=agg_res["average_power"], line=dict(color="lime")))
                                        fig_agg.update_layout(template="plotly_dark", height=300)
                                        st.plotly_chart(fig_agg, use_container_width=True)
                                    
                                    # Show consistency
                                    if agg_res["peak_consistency"]:
                                        st.markdown("##### 💎 Most Consistent Spectral Peaks (Hz)")
                                        st.write(agg_res["peak_consistency"])

                    with work_tabs[2]:
                        st.markdown("**Experiment Audit Trail**")
                        exps = experiment_tracker.list_experiments()
                        if not exps:
                            st.info("No experiment logs found.")
                        else:
                            for exp in exps[:10]: # Show last 10
                                with st.expander(f"📌 {exp['experiment_id']} - {exp['timestamp'][:16].replace('T', ' ')}"):
                                    # Display dataset name instead of cryptic ID
                                    ds_id = exp.get('dataset_id', '')
                                    ds_name = ds_id
                                    if ds_id in st.session_state.datasets:
                                        ds_name = st.session_state.datasets[ds_id].get('name', ds_id)
                                    st.write(f"**Dataset:** {ds_name}")
                                    st.write("**Steps:**")
                                    st.json(exp["pipeline"])
                                    if "results" in exp:
                                        st.write("**Outcome:**")
                                        st.write(exp["results"].get("final_summary", "Completed"))

                    with work_tabs[2]:
                        st.markdown("**Validation Workbench**")
                        from modules import hirax_validation
                        v1, v2 = st.columns(2)
                        inject_freq = v1.number_input("Injection Frequency (Hz)", value=12.5)
                        inject_amp = v2.number_input("Injection Amplitude", value=5.0)
                        
                        if st.button("🧪 Run Validation Test", use_container_width=True):
                            with st.spinner("Simulating & Validating..."):
                                # Generate synthetic data based on current slice shape
                                synth_data = hirax_validation.generate_synthetic_signal(
                                    data_slice.shape[0], data_slice.shape[1],
                                    signals=[{"freq": inject_freq, "amp": inject_amp}],
                                    noise_level=1.0
                                )
                                # Measure accuracy
                                score = hirax_validation.measure_detection_accuracy(synth_data, [inject_freq])
                                
                                st.metric("Detection Accuracy", f"{score['accuracy']*100:.1f}%")
                                if score['accuracy'] > 0:
                                    st.success("✅ Known signal correctly identified in synthetic noise.")
                                else:
                                    st.warning("❌ Known signal missed. Check threshold and windowing.")
                                    
                                # Plot synthetic result
                                p_synth = hirax_analysis.compute_power_spectrum(synth_data)
                                fig_synth = go.Figure(go.Scatter(x=p_synth["frequencies"], y=p_synth["power"]))
                                fig_synth.update_layout(title="Synthetic Test Spectrum", template="plotly_dark")
                                st.plotly_chart(fig_synth, use_container_width=True)

                    with st.expander("🕒 Temporal Session Engine", expanded=False):
                        st.markdown("Reconstruct a continuous observation timeline from multiple HDF5 files.")
                        session_files = st.multiselect(
                            "Select HDF5 files for session reconstruction",
                            list(RAW_H5_DIR.glob("*.h5")) + list(RAW_H5_DIR.glob("*.hdf5")),
                            format_func=lambda x: x.name,
                            key="hirax_session_files",
                        )
                        preferred_dataset_path = st.text_input(
                            "Preferred visibility dataset path (optional)",
                            value=st.session_state.get("hirax_session_dataset_path", ""),
                            key="hirax_session_dataset_path_input",
                        )
                        target_bins = st.number_input(
                            "Combined waterfall frequency bins",
                            min_value=16,
                            max_value=4096,
                            value=256,
                            step=16,
                            key="hirax_session_target_bins",
                        )

                        if st.button("🔬 Reconstruct Observation Sessions", use_container_width=True):
                            if not session_files:
                                st.warning("Please select at least one HDF5 file.")
                            else:
                                with st.spinner("Reconstructing observation timeline..."):
                                    session_result = observation_session.reconstruct_observation_sessions(
                                        [str(path) for path in session_files],
                                        preferred_dataset_path=preferred_dataset_path or None,
                                        target_freq_bins=int(target_bins),
                                    )
                                    st.session_state["hirax_observation_sessions"] = session_result
                                    st.success(f"Reconstructed {len(session_result['records'])} observation sessions.")

                        session_result = st.session_state.get("hirax_observation_sessions")
                        if session_result:
                            consistency = session_result.get("consistency_report", {})
                            fusion = session_result.get("weighted_fusion", {})
                            timeline = session_result.get("timeline")
                            combined_waterfall = session_result.get("combined_waterfall")
                            session_records = session_result.get("records", [])
                            cache_key = _observatory_cache_key(session_records)
                            observatory_cache = st.session_state.setdefault("hirax_observatory_cache", {})
                            if observatory_cache.get("cache_key") == cache_key:
                                observatory = observatory_cache.get("observatory", {})
                            else:
                                observatory = observatory_intelligence.build_observatory_session_intelligence(session_records)
                                observatory_cache["cache_key"] = cache_key
                                observatory_cache["observatory"] = observatory

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Sessions", f"{consistency.get('session_count', 0)}")
                            c2.metric("Consistent", "Yes" if consistency.get("is_consistent") else "No")
                            c3.metric("Ordered", "Yes" if consistency.get("ordered_times") else "No")
                            c4.metric("Fusion", "Ready" if fusion.get("available") else "Unavailable")

                            if observatory.get("confidence", {}).get("available"):
                                confidence = observatory["confidence"]
                                confidence_value = confidence.get("overall_confidence")
                                if confidence_value is not None:
                                    st.metric("Session confidence", f"{confidence_value:.3f}")
                                st.caption(
                                    f"Imaging reliability: {confidence.get('imaging_reliability_estimate', 0.0):.3f} | "
                                    f"Cosmology readiness: {confidence.get('cosmology_readiness_estimate', 0.0):.3f}"
                                )

                            if observatory.get("uv_tracks", {}).get("available"):
                                uv_tracks = observatory["uv_tracks"]
                                st.markdown("##### UV-track visualization")
                                st.dataframe(uv_tracks.get("uv_track_points"), use_container_width=True)
                                if isinstance(uv_tracks.get("uv_coverage_density"), np.ndarray) and uv_tracks["uv_coverage_density"].size:
                                    fig_uv = go.Figure(data=go.Heatmap(z=uv_tracks["uv_coverage_density"], colorscale="Viridis"))
                                    fig_uv.update_layout(title="UV Coverage Density", template="plotly_dark", height=420)
                                    st.plotly_chart(fig_uv, use_container_width=True)

                            if observatory.get("closure_explorer", {}).get("available"):
                                closure_explorer = observatory["closure_explorer"]
                                st.markdown("##### Closure triangle explorer")
                                triangle_browser = closure_explorer.get("triangle_browser")
                                if isinstance(triangle_browser, pd.DataFrame) and not triangle_browser.empty:
                                    st.dataframe(triangle_browser, use_container_width=True)
                                phase_preview = closure_explorer.get("closure_phase_heatmap")
                                if isinstance(phase_preview, np.ndarray) and phase_preview.size:
                                    fig_phase = go.Figure(data=go.Heatmap(z=phase_preview, colorscale="RdBu", zmid=0))
                                    fig_phase.update_layout(title="Closure Phase Heatmap", template="plotly_dark", height=420)
                                    st.plotly_chart(fig_phase, use_container_width=True)
                                amp_preview = closure_explorer.get("closure_amplitude_heatmap")
                                if isinstance(amp_preview, np.ndarray) and amp_preview.size:
                                    fig_amp = go.Figure(data=go.Heatmap(z=amp_preview, colorscale="Cividis"))
                                    fig_amp.update_layout(title="Closure Amplitude Heatmap", template="plotly_dark", height=420)
                                    st.plotly_chart(fig_amp, use_container_width=True)

                            if observatory.get("calibration_maps", {}).get("available"):
                                calibration_maps = observatory["calibration_maps"]
                                st.markdown("##### Calibration stability maps")
                                st.write(calibration_maps.get("session_quality_summary", {}))
                                coherence = calibration_maps.get("coherence_ranking")
                                if isinstance(coherence, pd.DataFrame) and not coherence.empty:
                                    st.dataframe(coherence, use_container_width=True)
                                gain_map = calibration_maps.get("gain_stability_map")
                                if isinstance(gain_map, np.ndarray) and gain_map.size:
                                    fig_gain = go.Figure(data=go.Heatmap(z=gain_map, colorscale="Viridis"))
                                    fig_gain.update_layout(title="Gain Stability Map", template="plotly_dark", height=420)
                                    st.plotly_chart(fig_gain, use_container_width=True)
                                phase_map = calibration_maps.get("phase_drift_map")
                                if isinstance(phase_map, np.ndarray) and phase_map.size:
                                    fig_phase_drift = go.Figure(data=go.Heatmap(z=phase_map, colorscale="RdBu", zmid=0))
                                    fig_phase_drift.update_layout(title="Phase Drift Map", template="plotly_dark", height=420)
                                    st.plotly_chart(fig_phase_drift, use_container_width=True)
                                bandpass_map = calibration_maps.get("bandpass_evolution_map")
                                if isinstance(bandpass_map, np.ndarray) and bandpass_map.size:
                                    fig_bandpass = go.Figure(data=go.Heatmap(z=bandpass_map, colorscale="Plasma"))
                                    fig_bandpass.update_layout(title="Bandpass Evolution Map", template="plotly_dark", height=420)
                                    st.plotly_chart(fig_bandpass, use_container_width=True)

                            if observatory.get("persistent_rfi", {}).get("available"):
                                persistent_rfi = observatory["persistent_rfi"]
                                st.markdown("##### Persistent RFI heatmaps")
                                st.write(persistent_rfi.get("frequency_occupancy_summary", {}))
                                ranking = persistent_rfi.get("baseline_contamination_ranking")
                                if isinstance(ranking, pd.DataFrame) and not ranking.empty:
                                    st.dataframe(ranking, use_container_width=True)
                                persistent_map = persistent_rfi.get("persistent_contamination_map")
                                if isinstance(persistent_map, np.ndarray) and persistent_map.size:
                                    fig_persistent = go.Figure(data=go.Heatmap(z=persistent_map, colorscale="Magma"))
                                    fig_persistent.update_layout(title="Persistent Channel Occupancy", template="plotly_dark", height=420)
                                    st.plotly_chart(fig_persistent, use_container_width=True)
                                time_freq = persistent_rfi.get("time_frequency_evolution")
                                if isinstance(time_freq, np.ndarray) and time_freq.size:
                                    fig_time_freq = go.Figure(data=go.Heatmap(z=time_freq, colorscale="Inferno"))
                                    fig_time_freq.update_layout(title="Time-Frequency RFI Evolution", template="plotly_dark", height=420)
                                    st.plotly_chart(fig_time_freq, use_container_width=True)

                            if observatory.get("uv_aware_fusion", {}).get("available"):
                                uv_fusion = observatory["uv_aware_fusion"]
                                st.markdown("##### UV-aware session fusion")
                                st.write({
                                    "compatible_records": uv_fusion.get("compatible_records", []),
                                    "rejected_records": uv_fusion.get("rejected_records", []),
                                    "preserves_time_order": uv_fusion.get("preserves_time_order"),
                                    "preserves_uv_coordinates": uv_fusion.get("preserves_uv_coordinates"),
                                    "preserves_polarization_products": uv_fusion.get("preserves_polarization_products"),
                                    "baseline_match_mode": uv_fusion.get("baseline_match_mode"),
                                })

                            if observatory.get("earth_rotation", {}).get("available"):
                                earth = observatory["earth_rotation"]
                                metrics = earth.get("uv_completeness_metrics", {})
                                st.markdown("##### Earth-rotation uv synthesis")
                                st.write({
                                    "n_uv_points": metrics.get("n_uv_points", 0),
                                    "unique_uv_points": metrics.get("unique_uv_points", 0),
                                    "coverage_symmetry": metrics.get("coverage_symmetry", 0.0),
                                    "beam_quality_score": metrics.get("beam_quality_score", 0.0),
                                })
                                trajectory = earth.get("trajectory")
                                if isinstance(trajectory, pd.DataFrame) and not trajectory.empty:
                                    st.dataframe(trajectory, use_container_width=True)

                            if observatory.get("calibration_drift", {}).get("available"):
                                drift = observatory["calibration_drift"]
                                st.markdown("##### Calibration drift analysis")
                                drift_timeline = drift.get("drift_timeline")
                                if isinstance(drift_timeline, pd.DataFrame) and not drift_timeline.empty:
                                    st.dataframe(drift_timeline, use_container_width=True)

                            if observatory.get("closure_timeline", {}).get("available"):
                                closure = observatory["closure_timeline"]
                                st.markdown("##### Closure timeline diagnostics")
                                closure_timeline = closure.get("closure_timeline")
                                if isinstance(closure_timeline, pd.DataFrame) and not closure_timeline.empty:
                                    st.dataframe(closure_timeline, use_container_width=True)

                            if observatory.get("rfi_evolution", {}).get("available"):
                                rfi = observatory["rfi_evolution"]
                                st.markdown("##### RFI evolution analysis")
                                session_rfi = rfi.get("session_rfi_summary")
                                if isinstance(session_rfi, pd.DataFrame) and not session_rfi.empty:
                                    st.dataframe(session_rfi, use_container_width=True)

                            if session_result.get("warnings"):
                                st.warning("; ".join(session_result["warnings"][:3]))
                            if session_result.get("errors"):
                                st.error("; ".join(session_result["errors"][:3]))

                            if isinstance(timeline, pd.DataFrame) and not timeline.empty:
                                st.markdown("##### Temporal visibility evolution")
                                st.dataframe(timeline, use_container_width=True)

                            if isinstance(combined_waterfall, np.ndarray) and combined_waterfall.size:
                                st.markdown("##### Combined waterfall analysis")
                                fig_temporal = go.Figure(
                                    data=go.Heatmap(
                                        z=np.log10(np.abs(combined_waterfall) + 1e-10),
                                        colorscale="Viridis",
                                    )
                                )
                                fig_temporal.update_layout(
                                    title="Combined Observation Waterfall",
                                    xaxis_title="Frequency Bin",
                                    yaxis_title="Session / Time Order",
                                    template="plotly_dark",
                                    height=500,
                                )
                                st.plotly_chart(fig_temporal, use_container_width=True)

                            if fusion.get("fused_visibility") is not None:
                                st.markdown("##### Weighted fusion diagnostics")
                                fused_visibility = fusion["fused_visibility"]
                                fused_stats = hirax_processing.get_validation_metrics(np.abs(fused_visibility), np.abs(fused_visibility), np.zeros_like(np.abs(fused_visibility), dtype=bool)) if hasattr(hirax_processing, "get_validation_metrics") else {}
                                st.write({
                                    "weights": fusion.get("weights", []),
                                    "visible_shape_matched": fusion.get("visible_shape_matched"),
                                    "fused_visibility_shape": list(fused_visibility.shape),
                                    "fused_waterfall_shape": fusion.get("fused_waterfall_shape"),
                                    "reason": fusion.get("reason"),
                                })
                                if hasattr(fused_visibility, "shape") and fused_visibility.ndim >= 2:
                                    fig_fused = go.Figure(data=go.Heatmap(z=np.abs(fused_visibility), colorscale="Cividis"))
                                    fig_fused.update_layout(title="Weighted Fused Visibility Amplitude", template="plotly_dark", height=450)
                                    st.plotly_chart(fig_fused, use_container_width=True)

                            if observatory.get("diagnostics", {}).get("available"):
                                diagnostics = observatory["diagnostics"]
                                st.markdown("##### Performance diagnostics")
                                st.write({
                                    "cache_signature": diagnostics.get("cache_signature"),
                                    "estimated_geometry_reuse": diagnostics.get("estimated_geometry_reuse"),
                                    "estimated_frequency_reuse": diagnostics.get("estimated_frequency_reuse"),
                                })
                                hotspots = diagnostics.get("hotspots")
                                if isinstance(hotspots, pd.DataFrame):
                                    st.dataframe(hotspots, use_container_width=True)
                                elif isinstance(hotspots, list) and hotspots:
                                    st.dataframe(pd.DataFrame(hotspots), use_container_width=True)
                                repeated = diagnostics.get("repeated_computation")
                                if isinstance(repeated, list) and repeated:
                                    st.dataframe(pd.DataFrame(repeated), use_container_width=True)

                            if observatory.get("resilience", {}).get("available"):
                                resilience = observatory["resilience"]
                                st.markdown("##### Malformed dataset resilience")
                                st.write({"quarantined_count": resilience.get("quarantined_count", 0)})
                                resilience_rows = resilience.get("rows")
                                if isinstance(resilience_rows, pd.DataFrame) and not resilience_rows.empty:
                                    st.dataframe(resilience_rows, use_container_width=True)

                            export_preview = observatory.get("session_export_preview", {})
                            if export_preview:
                                st.markdown("##### Session intelligence export")
                                st.write(export_preview)
                                csv_export = export_preview.get("csv")
                                json_export = export_preview.get("json")
                                if csv_export:
                                    st.download_button("⬇️ Download session intelligence CSV", Path(csv_export).read_bytes(), file_name=Path(csv_export).name, mime="text/csv")
                                if json_export:
                                    st.download_button("⬇️ Download session intelligence JSON", Path(json_export).read_bytes(), file_name=Path(json_export).name, mime="application/json")

                            export_dir = Path("/tmp") / "udip_hirax_exports"
                            try:
                                export_paths = observatory_intelligence.export_session_intelligence(
                                    observatory,
                                    export_dir,
                                    stem=f"{Path(h5_path).stem}_session_intelligence",
                                )
                                csv_bytes = Path(export_paths["csv"]).read_bytes()
                                json_bytes = Path(export_paths["json"]).read_bytes()
                                st.download_button(
                                    "⬇️ Download session intelligence CSV",
                                    csv_bytes,
                                    file_name=f"{Path(h5_path).stem}_session_intelligence.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                )
                                st.download_button(
                                    "⬇️ Download session intelligence JSON",
                                    json_bytes,
                                    file_name=f"{Path(h5_path).stem}_session_intelligence.json",
                                    mime="application/json",
                                    use_container_width=True,
                                )
                            except Exception as exc:
                                st.warning(f"Session intelligence export unavailable: {exc}")

                            if observatory.get("guidance"):
                                guidance = observatory["guidance"]
                                st.markdown("##### Scientific interpretation")
                                st.caption(guidance.get("uv_coverage", ""))
                                st.caption(guidance.get("closure_stability", ""))
                                st.caption(guidance.get("calibration_drift", ""))
                                st.caption(guidance.get("rfi_occupancy", ""))
                                st.caption(guidance.get("confidence_score", ""))
                                st.caption(guidance.get("imaging_readiness", ""))
                                st.caption(guidance.get("cosmology_readiness", ""))

                            if observatory.get("reproducibility", {}).get("available"):
                                reproducibility = observatory["reproducibility"]
                                st.markdown("##### Reproducibility diagnostics")
                                st.write({"dataset_signature": reproducibility.get("dataset_signature"), "cache_safe": reproducibility.get("cache_safe")})

                            if observatory.get("trust_checks", {}).get("available"):
                                trust_checks = observatory["trust_checks"]
                                st.markdown("##### Observatory consistency checks")
                                if trust_checks.get("warnings"):
                                    st.warning("; ".join(trust_checks.get("warnings", [])[:5]))

                            if observatory.get("export_integrity", {}).get("available"):
                                export_integrity = observatory["export_integrity"]
                                st.markdown("##### Export integrity")
                                st.write({k: export_integrity.get(k) for k in ["integrity_hash", "metadata_integrity", "timestamp_preserved", "baseline_provenance_preserved", "frequency_axis_preserved", "polarization_labels_preserved"]})

                            if observatory.get("stress", {}).get("available"):
                                stress = observatory["stress"]
                                st.markdown("##### Stress validation")
                                st.write({"session_count": stress.get("session_count"), "long_waterfall_rows": stress.get("long_waterfall_rows")})

                            if observatory.get("export_signature_summary", {}).get("available"):
                                st.markdown("##### Export reproducibility signature")
                                st.write(observatory["export_signature_summary"])

                except Exception as e:
                    st.error(f"Invalid slice or data error: {e}")

    except Exception as e:
        st.error(f"Error reading HDF5 file: {e}")

if __name__ == "__main__":
    render()
