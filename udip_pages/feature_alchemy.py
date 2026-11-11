"""
UDIP v3 — udip_pages/feature_alchemy.py
AI-driven and manual feature engineering.
"""

import streamlit as st
import pandas as pd
from modules.ai_engine import suggest_features, build_dataset_context
from modules.core import add_history, log
from modules.state_manager import get_page_memory

TRANSFORMS = [
    "Log Transform", "Square Root", "One-Hot Encoding",
    "Standard Scaling", "Min-Max Scaling", "Lag Features"
]

def render():
    st.markdown('<div class="main-header">🔬 Feature Alchemy</div>', unsafe_allow_html=True)
    st.markdown("Transform your data into gold using AI-driven feature engineering and manual alchemy.")

    datasets = st.session_state.get("datasets", {})
    if not datasets:
        st.warning("Upload a dataset to start alchemy.")
        return

    _mem = get_page_memory("alchemy")

    # Restore previously selected dataset
    available_ds = list(datasets.keys())
    saved_ds = _mem.get("ds_id")
    default_idx = available_ds.index(saved_ds) if saved_ds in available_ds else 0

    ds_id = st.selectbox(
        "Select Dataset for Alchemy",
        available_ds,
        format_func=lambda k: datasets[k]["name"],
        index=default_idx,
    )
    _mem.set("ds_id", ds_id)

    df = datasets[ds_id]["df"]

    tab1, tab2 = st.tabs(["🔮 AI Suggestion", "🛠️ Manual Alchemy"])

    with tab1:
        st.subheader("AI Feature Oracle")
        st.markdown("Let the AI analyze your data and suggest powerful new columns.")

        if st.button("🔮 Consult the Oracle", type="primary"):
            with st.spinner("Analyzing data patterns..."):
                ctx = build_dataset_context(selected_dataset_id=ds_id)
                suggestions = suggest_features(df, ctx)
                st.markdown("### Suggestions from the Oracle")
                st.markdown(suggestions)

                if st.button("🚀 Apply to Code Editor"):
                    st.session_state.editor_code = suggestions
                    st.session_state.active_page = "🐍 Code Editor"
                    st.rerun()

    with tab2:
        st.subheader("Standard Transformations")
        col1, col2 = st.columns(2)

        with col1:
            # Restore saved transform and column
            saved_transform = _mem.get("transform", TRANSFORMS[0])
            t_idx = TRANSFORMS.index(saved_transform) if saved_transform in TRANSFORMS else 0
            transform = st.selectbox("Transformation Type", TRANSFORMS, index=t_idx)
            _mem.set("transform", transform)

            col_options = list(df.columns)
            saved_col = _mem.get("target_col")
            c_idx = col_options.index(saved_col) if saved_col in col_options else 0
            target_col = st.selectbox("Target Column", col_options, index=c_idx)
            _mem.set("target_col", target_col)

        with col2:
            st.info("Manual transformation logic will generate code you can run in the editor.")

        if st.button("✨ Generate Alchemy Code"):
            code = f"# Manual Alchemy: {transform} on {target_col}\n"
            if transform == "Log Transform":
                code += f"import numpy as np\ndf['{target_col}_log'] = np.log1p(df['{target_col}'])\n"
            elif transform == "Square Root":
                code += f"import numpy as np\ndf['{target_col}_sqrt'] = np.sqrt(df['{target_col}'].clip(0))\n"
            elif transform == "One-Hot Encoding":
                code += f"df = pd.get_dummies(df, columns=['{target_col}'])\n"
            elif transform == "Standard Scaling":
                code += (f"from sklearn.preprocessing import StandardScaler\n"
                         f"scaler = StandardScaler()\n"
                         f"df['{target_col}_scaled'] = scaler.fit_transform(df[['{target_col}']]).flatten()\n")
            elif transform == "Min-Max Scaling":
                code += (f"from sklearn.preprocessing import MinMaxScaler\n"
                         f"scaler = MinMaxScaler()\n"
                         f"df['{target_col}_minmax'] = scaler.fit_transform(df[['{target_col}']]).flatten()\n")
            elif transform == "Lag Features":
                code += (f"for lag in [1, 2, 3]:\n"
                         f"    df[f'{target_col}_lag{{lag}}'] = df['{target_col}'].shift(lag)\n")

            st.session_state.editor_code = code
            st.session_state.active_page = "🐍 Code Editor"
            st.rerun()
