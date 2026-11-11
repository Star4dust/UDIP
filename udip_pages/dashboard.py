import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""
UDIP v3 — pages/dashboard.py
Pinned charts dashboard — charts persist across sessions.
"""

import streamlit as st
import pandas as pd
import json
from modules.core import (
    log, add_history, save_dashboard, fmt_bytes, get_memory_mb,
    APP_VERSION, DATA_DIR
)
from modules.state_manager import get_state_manager
from datetime import datetime

def render():
    st.markdown('<div class="main-header">📌 Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Pin your most important charts here — they reload automatically every session.")

    state_mgr = get_state_manager()
    pins = st.session_state.get("dashboard_pins", [])
    datasets = st.session_state.get("datasets", {})

    # ── Top stats bar ──────────────────────────────────────────────────────────
    total_rows = sum(v["df"].shape[0] for v in datasets.values())
    mem_mb     = get_memory_mb()
    n_ds       = len(datasets)
    n_hist     = len(st.session_state.get("history", []))

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Datasets loaded",  n_ds)
    c2.metric("Total rows",       f"{total_rows:,}")
    c3.metric("Memory used",      f"{mem_mb:.0f} MB")
    c4.metric("Actions logged",   n_hist)
    c5.metric("App version",      f"v{APP_VERSION}")

    st.markdown(
        f"""
        <div class="udip-panel">
            <div class="udip-panel__eyebrow">Pinned Intelligence</div>
            <div class="udip-panel__title">Keep your highest-signal visuals one click away</div>
            <div class="udip-panel__text">
                Dashboard pins turn one-off exploration into a persistent analyst cockpit. Track the charts you keep
                revisiting, then use this page as the fastest health-check for the workspace.
            </div>
            <div class="udip-stat-strip">
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{len(pins)}</div>
                    <div class="udip-stat-tile__label">Pinned visuals</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{n_ds}</div>
                    <div class="udip-stat-tile__label">Active datasets</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{total_rows:,}</div>
                    <div class="udip-stat-tile__label">Rows in memory</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Pin a new chart ────────────────────────────────────────────────────────
    with st.expander("📌 Pin a new chart to this dashboard", expanded=len(pins)==0):
        if not datasets:
            st.info("Upload some datasets first, then pin charts here.")
        else:
            try:
                import plotly.express as px
            except ImportError:
                st.error("Plotly not installed. Restart the app."); return

            col1, col2 = st.columns(2)
            ds_id   = col1.selectbox("Dataset", list(datasets.keys()),
                                     format_func=lambda k: datasets[k]["name"], key="pin_ds")
            df      = datasets[ds_id]["df"]
            num_cols= df.select_dtypes(include="number").columns.tolist()
            cat_cols= df.select_dtypes(include=["object","category"]).columns.tolist()

            chart_type = col2.selectbox("Chart type", [
                "Bar chart","Line chart","Pie chart","Histogram",
                "Scatter plot","Box plot","Heatmap","Metric card"
            ], key="pin_chart_type")

            title = st.text_input("Chart title", value=f"{chart_type} — {datasets[ds_id]['name']}", key="pin_title")

            pin_config = {"ds_id": ds_id, "ds_name": datasets[ds_id]["name"],
                          "chart_type": chart_type, "title": title}

            if chart_type in ["Bar chart","Line chart","Pie chart","Box plot"]:
                cx,cy,cz = st.columns(3)
                pin_config["x"] = cx.selectbox("X / Category", cat_cols or df.columns.tolist(), key="pin_x")
                pin_config["y"] = cy.selectbox("Y / Value",    num_cols or df.columns.tolist(), key="pin_y")
                pin_config["agg"] = cz.selectbox("Aggregation", ["sum","mean","count","max","min"], key="pin_agg")
            elif chart_type == "Histogram":
                pin_config["x"] = st.selectbox("Column", num_cols, key="pin_hx")
            elif chart_type == "Scatter plot":
                cx,cy = st.columns(2)
                pin_config["x"] = cx.selectbox("X", num_cols, key="pin_sx")
                pin_config["y"] = cy.selectbox("Y", num_cols[1:] if len(num_cols)>1 else num_cols, key="pin_sy")
            elif chart_type == "Heatmap":
                pin_config["cols"] = st.multiselect("Columns", num_cols, default=num_cols[:8], key="pin_hm")
            elif chart_type == "Metric card":
                pin_config["col"] = st.selectbox("Metric column", num_cols, key="pin_mc")
                pin_config["func"] = st.selectbox("Function", ["sum","mean","max","min","count"], key="pin_mf")

            if st.button("📌 Pin this chart", type="primary"):
                import uuid
                pin_config["id"]      = str(uuid.uuid4())[:8]
                pin_config["pinned"]  = datetime.now().strftime("%Y-%m-%d %H:%M")
                pins.append(pin_config)
                st.session_state.dashboard_pins = pins
                save_dashboard(pins)
                add_history("Dashboard pin", f"Pinned: {title}")
                st.success(f"✅ '{title}' pinned to dashboard!")
                st.rerun()

    st.divider()

    # ── Render pinned charts ───────────────────────────────────────────────────
    if not pins:
        st.info("📌 No charts pinned yet. Use the panel above to pin your first chart.")
        return

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.error("Plotly not installed."); return

    # Render in 2-column grid
    for row_start in range(0, len(pins), 2):
        cols = st.columns(2)
        for ci, pin in enumerate(pins[row_start:row_start+2]):
            with cols[ci]:
                # Get dataset — may have been deleted
                ds_id = pin.get("ds_id","")
                if ds_id not in datasets:
                    st.warning(f"Dataset '{pin.get('ds_name')}' no longer loaded.")
                    if st.button("🗑️ Remove", key=f"rm_{pin['id']}"):
                        pins = [p for p in pins if p["id"] != pin["id"]]
                        st.session_state.dashboard_pins = pins
                        save_dashboard(pins)
                        st.rerun()
                    continue

                df = datasets[ds_id]["df"]
                ct = pin["chart_type"]
                title = pin["title"]

                try:
                    fig = None
                    _metric_val = None  # tracked separately for Metric card pins
                    if ct == "Bar chart":
                        agg_fn = getattr(df.groupby(pin["x"])[pin["y"]], pin["agg"])
                        df_p = agg_fn().nlargest(20).reset_index()
                        fig = px.bar(df_p, x=pin["x"], y=pin["y"], title=title, color=pin["x"])
                    elif ct == "Line chart":
                        df_p = df.groupby(pin["x"])[pin["y"]].agg(pin["agg"]).reset_index()
                        fig = px.line(df_p, x=pin["x"], y=pin["y"], title=title, markers=True)
                    elif ct == "Pie chart":
                        df_p = df.groupby(pin["x"])[pin["y"]].sum().nlargest(10).reset_index()
                        fig = px.pie(df_p, names=pin["x"], values=pin["y"], title=title)
                    elif ct == "Histogram":
                        fig = px.histogram(df, x=pin["x"], title=title, nbins=30)
                    elif ct == "Scatter plot":
                        if len(df) > 100_000:
                            fig = px.density_heatmap(
                                df, x=pin["x"], y=pin["y"],
                                nbinsx=50, nbinsy=50,
                                color_continuous_scale="Viridis",
                                title=title + " (full data density)"
                            )
                        else:
                            fig = px.scatter(df, x=pin["x"], y=pin["y"],
                                             title=title, opacity=0.6, trendline="ols")
                    elif ct == "Box plot":
                        fig = px.box(df, y=pin["y"], title=title, points="outliers")
                    elif ct == "Heatmap":
                        cols_h = [c for c in pin.get("cols",[]) if c in df.columns]
                        if cols_h:
                            corr = df[cols_h].corr().round(2)
                            fig  = px.imshow(corr, title=title, color_continuous_scale="RdBu_r",
                                             zmin=-1, zmax=1, text_auto=True)
                    elif ct == "Metric card":
                        col_m = pin["col"]
                        func  = pin["func"]
                        if col_m in df.columns:
                            _metric_val = getattr(df[col_m], func)()
                            st.metric(label=title, value=f"{_metric_val:,.2f}")

                    if fig:
                        fig.update_layout(
                            template="plotly_white",
                            height=320,
                            margin=dict(t=40,b=20,l=10,r=10),
                            showlegend=False,
                            font=dict(family="Aptos, Segoe UI, sans-serif"),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, width="stretch")

                except Exception as e:
                    st.error(f"Chart error: {e}")

                # Action bar — info | send to report studio | remove
                col_info, col_send, col_rm = st.columns([2, 1.4, 0.7])
                col_info.caption(f"📊 {pin.get('ds_name')} · Pinned {pin.get('pinned','')}")

                if col_send.button("📖 → Report Studio", key=f"send_{pin['id']}",
                                   help="Send this chart to Report Studio"):
                    try:
                        if fig is not None:
                            state_mgr.pin_item(
                                "chart",
                                {
                                    "chart_type": ct,
                                    "dataset":    pin.get("ds_name", ""),
                                    "plotly_json": fig.to_json(),
                                },
                                title,
                            )
                        elif _metric_val is not None:
                            state_mgr.pin_item(
                                "kpi",
                                {
                                    "label": pin.get("title", title),
                                    "value": f"{_metric_val:,.2f}",
                                    "delta": "",
                                    "note":  f"Live from dataset: {pin.get('ds_name', '')}",
                                    "tags":  [],
                                },
                                title,
                            )
                        else:
                            state_mgr.pin_item(
                                "chart",
                                {"chart_type": ct, "dataset": pin.get("ds_name", "")},
                                title,
                            )
                        st.success(f"📖 '{title}' sent to Report Studio!")
                    except Exception as _e:
                        st.error(f"Could not send to Report Studio: {_e}")

                if col_rm.button("🗑️", key=f"unpin_{pin['id']}", help="Remove from dashboard"):
                    pins = [p for p in pins if p["id"] != pin["id"]]
                    st.session_state.dashboard_pins = pins
                    save_dashboard(pins)
                    st.rerun()

    # ── Refresh button ─────────────────────────────────────────────────────────
    st.divider()
    col_ref, col_clear = st.columns([1,4])
    if col_ref.button("🔄 Refresh all charts"):
        st.rerun()
    if col_clear.button("🗑️ Clear all pins"):
        st.session_state.dashboard_pins = []
        save_dashboard([])
        st.rerun()
