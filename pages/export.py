import streamlit as st
from pathlib import Path
from modules.core import df_to_bytes, EXT, MIME, fmt_bytes, add_history, log
from modules.notifications import notify_task_complete

def render():
    st.markdown('<div class="main-header">💾 Export Dataset</div>', unsafe_allow_html=True)

    if not st.session_state.datasets:
        st.warning("⚠️ No datasets loaded. Upload or merge some data first.")
        st.stop()

    ds_names = {k: v["name"] for k, v in st.session_state.datasets.items()}
    chosen_id = st.selectbox("Select dataset to export",
                             list(ds_names.keys()),
                             format_func=lambda k: ds_names[k])
    df_exp = st.session_state.datasets[chosen_id]["df"]

    st.markdown(f"**{df_exp.shape[0]:,} rows × {df_exp.shape[1]} columns**")
    st.dataframe(df_exp.head(5), width="stretch")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fmt = st.radio("Export format", ["CSV", "Excel", "Parquet", "JSON"],
                       horizontal=True)
        fname = st.text_input("File name", value=Path(ds_names[chosen_id]).stem)
    with col2:
        st.markdown("**Format info**")
        fmt_info = {
            "CSV":     "✅ Universal format · Works with Power BI, Excel, Tableau, any tool",
            "Excel":   "✅ .xlsx format · Opens directly in Microsoft Excel",
            "Parquet": "✅ Fast columnar format · Great for large datasets",
            "JSON":    "✅ API-ready format · Records-oriented",
        }
        st.info(fmt_info[fmt])

    fmt_key = fmt.lower()
    file_bytes = df_to_bytes(df_exp, fmt_key)
    file_name  = fname + EXT[fmt_key]
    mime       = MIME[fmt_key]

    dl_clicked = st.download_button(
        label=f"⬇️ Download {file_name}",
        data=file_bytes,
        file_name=file_name,
        mime=mime,
        type="primary",
        width="stretch",
    )
    if dl_clicked:
        size_str = fmt_bytes(len(file_bytes))
        st.success(f"✅ Download started: **{file_name}** ({size_str})")
        add_history("Export", f"{file_name} — {len(df_exp):,} rows — {size_str}")
        log.info(f"Exported: {file_name} ({size_str})")
        notify_task_complete("Export", f"{file_name} ({size_str})")

    st.markdown("---")
    st.markdown("### 📊 Using with Power BI")
    st.markdown("""
    1. Export as **CSV** or **Parquet**.
    2. Open Power BI Desktop.
    3. Click 'Get Data' -> 'Text/CSV' (for CSV) or 'More...' -> 'Parquet' (for Parquet).
    4. Select your downloaded file.
    5. Your UDIP-cleaned data is ready for downstream analysis.
    """)
