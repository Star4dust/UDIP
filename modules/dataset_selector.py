"""
UDIP v3 — modules/dataset_selector.py
Dataset selector component for context-aware workflows.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def render_dataset_selector(workflow_type: str = "general") -> Optional[str]:
    """
    Render dataset selector dropdown for workflow-specific dataset selection.
    
    Args:
        workflow_type: Type of workflow ('hirax', 'analysis', 'reporting', etc.)
    
    Returns:
        str: Selected dataset name, or None if no dataset selected
    """
    if not st.session_state.datasets:
        st.info("No datasets loaded. Upload data first to proceed.")
        return None
    
    # Get dataset names
    dataset_names = list(st.session_state.datasets.keys())
    
    if not dataset_names:
        st.info("No datasets available in session state.")
        return None
    
    # Select dataset
    st.markdown("### 📊 Select Dataset")
    selected_dataset = st.selectbox(
        "Choose dataset for analysis",
        dataset_names,
        key=f"dataset_selector_{workflow_type}"
    )
    
    if not selected_dataset:
        return None
    
    # Show dataset preview
    dataset_info = st.session_state.datasets[selected_dataset]
    df = dataset_info["df"]
    
    # Display dataset metadata
    st.markdown("#### Dataset Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with col3:
        st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Show column names
    st.markdown("#### Columns")
    st.write(list(df.columns))
    
    # Show data types
    st.markdown("#### Data Types")
    st.write(df.dtypes)
    
    # Compatibility check based on workflow
    if workflow_type == "hirax":
        check_hirax_compatibility(df)
    elif workflow_type == "analysis":
        check_analysis_compatibility(df)
    
    return selected_dataset


def check_hirax_compatibility(df: pd.DataFrame):
    """
    Check if dataset is compatible with HIRAX analysis.
    
    Args:
        df: Dataset to check
    """
    st.markdown("#### HIRAX Compatibility Check")
    
    # Check for numeric columns (required for spectral analysis)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        st.success("✅ Dataset contains numeric columns suitable for spectral analysis")
        st.write(f"Found {len(numeric_cols)} numeric columns: {numeric_cols[:5]}...")
    else:
        st.warning("⚠️ Dataset may not be suitable for HIRAX analysis (needs numeric columns)")
    
    # Check for signal-like patterns
    if len(numeric_cols) > 0:
        st.info(f"📈 Ready for signal processing with {len(numeric_cols)} signal channels")
    else:
        st.error("❌ No numeric columns found - cannot perform HIRAX analysis")


def check_analysis_compatibility(df: pd.DataFrame):
    """
    Check if dataset is compatible with general analysis.
    
    Args:
        df: Dataset to check
    """
    st.markdown("#### Analysis Compatibility Check")
    
    # Check dataset size
    if df.shape[0] > 1000:
        st.success("✅ Dataset size suitable for comprehensive analysis")
    elif df.shape[0] > 100:
        st.info("ℹ️ Dataset size moderate - suitable for basic analysis")
    else:
        st.warning("⚠️ Dataset small - limited analysis possible")
    
    # Check for missing values
    missing_pct = (df.isnull().sum().sum() / df.size) * 100
    if missing_pct < 5:
        st.success("✅ Low missing data percentage")
    elif missing_pct < 20:
        st.info(f"ℹ️ Moderate missing data: {missing_pct:.1f}%")
    else:
        st.warning(f"⚠️ High missing data: {missing_pct:.1f}%")
