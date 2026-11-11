import streamlit as st
import pandas as pd
import uuid
from modules.core import sanitize_varname, save_dataset, add_history

def render():
    st.markdown('<div class="main-header">🔗 Merge Datasets</div>', unsafe_allow_html=True)

    if len(st.session_state.datasets) < 2:
        st.warning("⚠️ Upload at least 2 datasets to merge.")
        st.stop()

    ds_names = {k: v["name"] for k, v in st.session_state.datasets.items()}
    
    col1, col2 = st.columns(2)
    with col1:
        ds_left = st.selectbox("Left Dataset (Base)", list(ds_names.keys()), format_func=lambda k: ds_names[k])
    with col2:
        ds_right = st.selectbox("Right Dataset (Join)", list(ds_names.keys()), 
                                format_func=lambda k: ds_names[k], index=1 if len(ds_names)>1 else 0)

    if ds_left == ds_right:
        st.error("Select two different datasets.")
        st.stop()

    df_l = st.session_state.datasets[ds_left]["df"]
    df_r = st.session_state.datasets[ds_right]["df"]

    # Auto-find common columns
    common = list(set(df_l.columns) & set(df_r.columns))
    
    c1, c2, c3 = st.columns(3)
    join_type = c1.selectbox("Join Type", ["inner", "left", "right", "outer"])
    left_on = c2.selectbox("Left Key", df_l.columns, index=df_l.columns.get_loc(common[0]) if common else 0)
    right_on = c3.selectbox("Right Key", df_r.columns, index=df_r.columns.get_loc(common[0]) if common else 0)

    out_name = st.text_input("New dataset name", value=f"merged_{ds_names[ds_left]}_{ds_names[ds_right]}")

    if st.button("▶️ Execute Merge", type="primary", width="stretch"):
        with st.spinner("Merging..."):
            from modules.execution_layer import merge_datasets
            res_obj = merge_datasets(df_l, df_r, how=join_type, left_on=left_on, right_on=right_on)
            
            if res_obj["success"]:
                res = res_obj["output"]
                ds_id = str(uuid.uuid4())[:8]
                entry = {
                    "name": out_name,
                    "df": res,
                    "format": "merged",
                    "source": "merge_ui",
                    "varname": sanitize_varname(out_name),
                }
                st.session_state.datasets[ds_id] = entry
                save_dataset(ds_id, entry)
                add_history("Merge", f"{out_name} ({len(res):,} rows)")
                st.success(f"✅ Merged! New dataset '{out_name}' has {len(res):,} rows.")
                st.dataframe(res.head(50), width="stretch")
            else:
                st.error(f"Merge failed: {res_obj['error']}")
