"""
UDIP v3 — udip_pages/data_pulse.py
Proactive Monitoring Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from modules.data_pulse import pulse_check_all, detect_outliers_zscore, calculate_dataset_drift
from modules.core import list_snapshots, restore_snapshot, log
from modules.state_manager import get_page_memory

def render():
    st.markdown('<div class="main-header">💓 Data Pulse</div>', unsafe_allow_html=True)
    st.markdown("Proactive monitoring of dataset health, anomalies, and statistical drift.")

    datasets = st.session_state.get("datasets", {})
    if not datasets:
        st.warning("Upload datasets to enable proactive monitoring.")
        return

    # ── Performance & Sampling Controls ────────────────────────────────────────
    _mem = get_page_memory("pulse")
    with st.sidebar:
        st.markdown("### ⚙️ Performance Controls")
        st.info("Large datasets can be scanned in segments to preserve UI responsiveness.")
        use_sampling = st.toggle(
            "Enable Sampling Mode",
            value=_mem.get("use_sampling", True),
            help="Scan a representative sample instead of the full dataset for speed."
        )
        _mem.set("use_sampling", use_sampling)

        _sample_options = [1000, 5000, 10000, 50000, 100000, 500000]
        _saved_sample = _mem.get("sample_size", 10000)
        _sample_val = _saved_sample if _saved_sample in _sample_options else 10000
        sample_size = st.select_slider(
            "Sample Size (rows)",
            options=_sample_options,
            value=_sample_val,
            disabled=not use_sampling
        )
        _mem.set("sample_size", sample_size)

        max_cols = st.slider("Max Columns to Scan", 5, 100, _mem.get("max_cols", 20))
        _mem.set("max_cols", max_cols)

        pulse_settings = {
            "sample_size": sample_size if use_sampling else None,
            "max_cols": max_cols
        }


    # ── Overview Stats ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    
    with st.spinner("Scanning datasets for anomalies..."):
        alerts = pulse_check_all(datasets, settings=pulse_settings)
    
    col1.metric("Active Monitors", len(datasets))
    col2.metric("Anomalies Detected", len(alerts))
    col3.metric("Critical Alerts", len([a for a in alerts if a['severity'] == 'High']))
    col4.metric("Engine Status", "Healthy", delta="Online")

    # ── Alerts Panel ──────────────────────────────────────────────────────────
    st.subheader("🚨 Real-time Alerts")
    if alerts:
        for alert in alerts:
            with st.expander(f"{alert['severity']}: {alert['message']} in **{alert['ds_name']}**", expanded=(alert['severity']=='High')):
                st.write(f"**Dataset:** {alert['ds_name']} (ID: `{alert['ds_id']}`)")
                st.write(f"**Column:** `{alert['column']}`")
                st.write(f"**Detection Method:** Z-Score (Threshold: 3.0)")
                
                # Show outlier samples
                df = datasets[alert['ds_id']]["df"]
                outliers = detect_outliers_zscore(df, alert['column'], sample_size=pulse_settings.get("sample_size"))
                st.dataframe(outliers.head(10), use_container_width=True)
                
                c1, c2 = st.columns(2)
                if c1.button("🧠 AI RCA (Root Cause Analysis)", key=f"rca_{alert['ds_id']}_{alert['column']}"):
                    st.session_state.active_page = "🤖 AI Assistant"
                    st.session_state._ai_trigger = f"I detected {len(outliers)} anomalies in the column '{alert['column']}' of dataset '{alert['ds_name']}'. Please perform a root cause analysis and explain if these are likely errors or meaningful insights."
                    st.rerun()
                if c2.button("🧹 Clean Anomalies", key=f"clean_{alert['ds_id']}_{alert['column']}"):
                    st.session_state.active_page = "⚒️ Data Forge"
                    st.rerun()
    else:
        st.success("No statistical anomalies detected in active datasets.")

    st.divider()

    # ── Drift Monitoring ──────────────────────────────────────────────────────
    st.subheader("📉 Statistical Drift Analysis")
    st.markdown("Compare your current data against historical snapshots to detect changing patterns.")
    
    ds_id = st.selectbox("Select Dataset to Check Drift", list(datasets.keys()), 
                         format_func=lambda k: datasets[k]["name"])
    
    snapshots = list_snapshots(ds_id)
    if not snapshots:
        st.info("No snapshots found for this dataset. Create a snapshot in 'Data Forge' or 'Upload' to enable drift detection.")
        if st.button("📸 Create Baseline Snapshot Now"):
            from modules.core import save_snapshot
            save_snapshot(ds_id, datasets[ds_id], reason="Manual Baseline")
            st.success("Snapshot created! You can now monitor drift against this baseline.")
            st.rerun()
    else:
        # Comparison logic
        col_drift_1, col_drift_2 = st.columns(2)
        with col_drift_1:
            baseline_snap = st.selectbox("Baseline Snapshot (Old)", 
                                       options=[s['snap_id'] for s in snapshots],
                                       format_func=lambda sid: next(s['timestamp'] for s in snapshots if s['snap_id']==sid))
        
        with col_drift_2:
            target_col = st.selectbox("Column to Monitor", datasets[ds_id]["df"].columns)

        if st.button("🔍 Analyze Drift"):
            with st.spinner("Calculating statistical divergence..."):
                df_old, err = restore_snapshot(ds_id, baseline_snap)
                if df_old is not None:
                    df_new = datasets[ds_id]["df"]
                    drift_res = calculate_dataset_drift(df_old, df_new, target_col)
                    
                    if "error" in drift_res:
                        st.error(f"Drift check failed: {drift_res['error']}")
                    else:
                        score = drift_res['score']
                        st.markdown(f"### Drift Score: **{score:.4f}**")
                        
                        if drift_res['is_drifted']:
                            st.error(f"⚠️ **Significant Drift Detected!** Method: {drift_res['method']}")
                            st.markdown("The statistical distribution of this column has changed meaningfully since the baseline snapshot.")
                        else:
                            st.success(f"✅ **No Significant Drift.** Method: {drift_res['method']}")
                            st.markdown("The data distribution remains stable compared to the baseline.")
                        
                        # Visualize distributions
                        import plotly.figure_factory as ff
                        import plotly.graph_objects as go
                        
                        # Limit sample size for plotting if needed
                        old_sample = df_old[target_col].dropna()
                        new_sample = df_new[target_col].dropna()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=old_sample, name="Baseline", opacity=0.6))
                        fig.add_trace(go.Histogram(x=new_sample, name="Current", opacity=0.6))
                        fig.update_layout(barmode='overlay', title=f"Distribution Shift: {target_col}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Could not restore snapshot: {err}")

    # ── Configuration ──────────────────────────────────────────────────────────
    with st.expander("⚙️ Pulse Configuration", expanded=False):
        st.slider("Anomaly Sensitivity (Z-Score Threshold)", 2.0, 5.0, 3.0, 0.1)
        st.checkbox("Enable Automated Drift Alerts (Beta)", value=False)
        st.multiselect("Exclude Columns from Monitoring", datasets[ds_id]["df"].columns.tolist())
