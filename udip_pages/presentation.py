"""
UDIP v3 — udip_pages/presentation.py
Full-screen interactive data storytelling.
"""

import streamlit as st
import pandas as pd
from modules.state_manager import get_state_manager

def render():
    state_mgr = get_state_manager()
    artifacts = state_mgr.get_pinned_items()

    if not artifacts:
        st.warning("No pinned items to present. Pin insights from other pages first.")
        return

    # ── Presentation Layout ──────────────────────────────────────────────────
    st.markdown('<div class="main-header">🎬 Presentation Mode</div>', unsafe_allow_html=True)
    
    # Global Filters in Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Global Filters")
        st.markdown("Apply filters to all data-aware charts in this presentation.")
        
        datasets = st.session_state.get("datasets", {})
        if datasets:
            active_ds = st.selectbox("Base Dataset for Filtering", list(datasets.keys()), 
                                    format_func=lambda k: datasets[k]["name"])
            df = datasets[active_ds]["df"]
            
            filter_col = st.selectbox("Filter Column", df.columns)
            if pd.api.types.is_numeric_dtype(df[filter_col]):
                min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
                st.slider("Range", min_val, max_val, (min_val, max_val))
            else:
                st.multiselect("Values", df[filter_col].unique())
        else:
            st.caption("No datasets loaded for global filtering.")

    # Render Slides
    for idx, item in enumerate(artifacts):
        title = item.get('title', 'Slide')
        type_ = item.get('type', 'unknown')
        data  = item.get('data', {})
        
        st.markdown(f"## {title}")
        
        if type_ == 'chart':
            try:
                import plotly.io as pio
                fig_json = data.get('plotly_json')
                if fig_json:
                    fig = pio.from_json(fig_json)
                    fig.update_layout(margin=dict(t=50, b=50, l=50, r=50))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error rendering chart: {e}")
        elif type_ == 'analysis' or type_ == 'insight':
            st.markdown(data.get('content', ''))
        elif type_ == 'code':
            st.code(data.get('code', ''), language="python")
        
        st.divider()

    st.markdown("""
    <style>
        .stApp {
            # background: #ffffff !important;
        }
        .main-header {
            text-align: center;
            border: none;
            box-shadow: none;
        }
    </style>
    """, unsafe_allow_html=True)
