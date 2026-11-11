"""
UDIP v3 — udip_pages/visualize.py
Professional Data Visualization Studio — 50+ chart types.
Every chart type a data analyst uses, with professional formatting.
"""

import streamlit as st
import pandas as pd
import numpy as np
from modules.core import add_history, save_chat, sanitize_varname
from modules.operator_actions import render_operator_handoff
from modules.visualizations import create_advanced_chart
from modules.state_manager import get_state_manager
from modules.workflow_handoff import render_handoff_actions, trigger_charts_created_handoff, HandoffType

# Color palettes for visualizations
PALETTES = {
    "Default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
    "Viridis": ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6dcd59", "#b4de2c", "#fde725"],
    "Plasma": ["#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786", "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"],
    "Professional": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#5D3A3A", "#8B7E74", "#B8B8B8", "#6A7F60", "#4A5D4F"],
    "Earth": ["#8B4513", "#A0522D", "#CD853F", "#DEB887", "#F4A460", "#D2691E", "#BC8F8F", "#F5DEB3", "#D2B48C", "#C4A484"],
    "Ocean": ["#003366", "#004080", "#004d99", "#0059b3", "#0066cc", "#0073e6", "#0080ff", "#1a8cff", "#3399ff", "#4da6ff"],
    "Sunset": ["#FF6B6B", "#FF8E72", "#FFB347", "#FFD93D", "#6BCB77", "#4D96FF", "#9B59B6", "#E74C3C", "#3498DB", "#2ECC71"]
}


def _save_viz_selection(key, value):
    """Save a visualization selection to session state."""
    state_mgr = get_state_manager()
    viz_state = state_mgr.get_page_state("visualize")
    viz_state["selections"][key] = value
    state_mgr.save_page_state("visualize", viz_state)


def _get_viz_selection(key, default=None):
    """Get a saved visualization selection from session state."""
    state_mgr = get_state_manager()
    viz_state = state_mgr.get_page_state("visualize")
    return viz_state.get("selections", {}).get(key, default)


def render():
    st.markdown('<div class="main-header">📊 Data Visualization Studio</div>', unsafe_allow_html=True)
    st.markdown("Professional chart library — **50+ chart types** covering every visualization a data analyst needs. Tableau-quality charts powered by Plotly.")
    render_operator_handoff("📊 Visualize")

    if not st.session_state.datasets:
        st.warning("Upload a dataset first to start visualizing.")
        st.stop()

    # Dataset selection with human-readable names
    ds_names = {k: v["name"] for k, v in st.session_state.datasets.items()}
    chosen_id = st.selectbox("Select dataset for visualization",
                             list(ds_names.keys()),
                             format_func=lambda k: ds_names[k])
    df_full = st.session_state.datasets[chosen_id]["df"]
    ds_name = ds_names[chosen_id]

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        st.error("Plotly not installed. Restart the app.")
    col_theme, col_palette = st.columns([1, 1])
    with col_theme:
        theme = st.selectbox("Theme", ["plotly_white", "simple_white", "seaborn", "ggplot2", "plotly", "plotly_dark"], key="viz_theme")
    with col_palette:
        palette_name = st.selectbox("Color palette", list(PALETTES.keys()), key="viz_palette")

    colors = PALETTES[palette_name]
    sample_label = "Rendering from the full dataset for accurate chart computation"

    # Chart type selection
    st.markdown("### Select Chart Type")
    chart_categories = {
        "Distribution": ["Histogram", "Box plot", "Violin plot", "KDE plot", "Density heatmap"],
        "Comparison": ["Bar chart", "Grouped bar chart", "Stacked bar chart", "Horizontal bar chart", "Lollipop chart"],
        "Relationship": ["Scatter plot", "Bubble chart", "Line chart", "Scatter matrix", "Correlation heatmap"],
        "Time Series": ["Line chart", "Area chart", "Time series with trend line", "Rolling average chart"],
        "Categorical": ["Pie chart", "Donut chart", "Treemap", "Sunburst chart", "Bar chart"],
        "Geospatial": ["Scatter map", "Choropleth map"]
    }
    
    selected_category = st.selectbox("Chart Category", list(chart_categories.keys()))
    chart_type = st.selectbox("Chart Type", chart_categories[selected_category])

    st.markdown("---")
    
    # Get column information for the selected dataset
    all_cols = df_full.columns.tolist()
    num_cols = df_full.select_dtypes(include="number").columns.tolist()
    cat_cols = df_full.select_dtypes(include="object").columns.tolist()
    date_cols = df_full.select_dtypes(include=["datetime64"]).columns.tolist()
    
    # Set df for chart rendering compatibility
    df = df_full
    
    st.markdown(
        f"""
        <div class="udip-panel">
            <div class="udip-panel__eyebrow">Visualization Studio</div>
            <div class="udip-panel__title">{ds_name}</div>
            <div class="udip-panel__text">
                Design chart-first stories with dataset-aware controls, professional palettes, and direct handoff to the AI assistant
                when you want interpretation instead of raw plots.
            </div>
            <div class="udip-stat-strip">
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{len(df_full):,}</div>
                    <div class="udip-stat-tile__label">Rows available</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{len(all_cols)}</div>
                    <div class="udip-stat-tile__label">Columns</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{len(num_cols)}</div>
                    <div class="udip-stat-tile__label">Numeric fields</div>
                </div>
                <div class="udip-stat-tile">
                    <div class="udip-stat-tile__value">{len(cat_cols)}</div>
                    <div class="udip-stat-tile__label">Categorical fields</div>
                </div>
            </div>
            <div class="udip-pill-row">
                <div class="udip-pill udip-pill--accent">{theme}</div>
                <div class="udip-pill">{palette_name}</div>
                <div class="udip-pill udip-pill--warm">{sample_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Chart rendering helper ────────────────────────────────────────────────
    def show_chart(fig, height=550):
        """Apply professional formatting and display chart."""
        dark_mode_chart = theme == "plotly_dark"
        axis_font_color = "#f8fafc" if dark_mode_chart else "#111111"
        chart_paper_bg = "#0f172a" if dark_mode_chart else "rgba(0,0,0,0)"
        chart_plot_bg = "#111827" if dark_mode_chart else "rgba(0,0,0,0)"
        fig.update_layout(
            template=theme,
            font=dict(family="Aptos, Segoe UI, sans-serif", size=13, color=axis_font_color),
            margin=dict(t=60, b=40, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            colorway=colors,
            height=height,
            hoverlabel=dict(font_size=12, font_family="Aptos, Segoe UI, sans-serif"),
            paper_bgcolor=chart_paper_bg,
            plot_bgcolor=chart_plot_bg,
        )
        fig.update_traces(selector=dict(type="parcoords"),
                          labelfont=dict(color=axis_font_color, size=12),
                          tickfont=dict(color=axis_font_color, size=11))
        # Professional axis formatting
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.1)",
            showline=True, linewidth=1, linecolor="rgba(128,128,128,0.3)",
            tickformat=",.0f" if theme == "plotly_dark" else None,
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.1)",
            showline=True, linewidth=1, linecolor="rgba(128,128,128,0.3)",
            tickformat=",.0f" if theme == "plotly_dark" else None,
        )

        try:
            # Performance Guard: Check number of categories if plot has color or many bars
            if hasattr(fig, "data"):
                total_traces = len(fig.data)
                if total_traces > 100:
                    st.warning(f"⚠️ High complexity plot: {total_traces} traces detected. Interaction might be sluggish.")
            
            st.plotly_chart(fig, width="stretch", config={
                "toImageButtonOptions": {"format": "png", "filename": f"UDIP_{chart_type.replace(' ','_')}",
                                          "height": 800, "width": 1200, "scale": 2},
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
            }, theme="streamlit")  # Use streamlit theme for proper dark/light mode
        except Exception as e:
            st.error(f"🎨 Visualization Error: This chart type failed to render with the current data selection. Try changing the X or Y axis.\n\nDetails: {e}")
        add_history("Visualization", f"{chart_type} — {ds_name}")

        # AI interpretation and Pin buttons — stable key avoids collision when fig is None
        import hashlib as _hl
        _hash = _hl.md5(f"{chart_type}{ds_name}".encode()).hexdigest()[:8]
        _btn_key = "ai_plot_" + _hash
        _pin_key = "pin_plot_" + _hash
        
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button(f"🤖 How do I interpret this {chart_type}?", key=_btn_key, width="stretch"):
                # Build a richer prompt with actual dataset context
                try:
                    _col_info = ", ".join(df.columns.tolist()[:10])
                    _shape = f"{len(df):,} rows × {len(df.columns)} cols"
                    _num_cols = df.select_dtypes(include="number").columns.tolist()[:5]
                    _stats = ""
                    if _num_cols:
                        _stats = "\nKey numeric ranges: " + ", ".join(
                            f"{c}: {df[c].min():.2g}–{df[c].max():.2g}" for c in _num_cols
                        )
                except Exception:
                    _col_info = "various columns"
                    _shape = "unknown size"
                    _stats = ""

                prompt_text = (
                    f"I just generated a {chart_type} using the dataset '{ds_name}' ({_shape}).\n"
                    f"Columns available: {_col_info}{_stats}\n\n"
                    f"Please explain:\n"
                    f"1. How to read and interpret a {chart_type}\n"
                    f"2. What patterns or trends I should look for in my specific data\n"
                    f"3. What anomalies or outliers might indicate a problem\n"
                    f"4. What follow-up analysis would you recommend based on this chart?"
                )

                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                st.session_state.chat_history.append({"role": "user", "content": prompt_text})
                from modules.core import save_chat
                save_chat(st.session_state.chat_history)
                st.session_state._ai_trigger = prompt_text
                st.session_state.active_page = "🤖 AI Assistant"
                st.rerun()
                
        with c2:
            if st.button("📌 Pin Chart", key=_pin_key, width="stretch"):
                try:
                    fig_json = fig.to_json()
                    chart_data = {
                        "chart_type": chart_type,
                        "dataset": ds_name,
                        "config": {
                            "x": x if 'x' in locals() else None,
                            "y": y if 'y' in locals() else None,
                            "agg": agg if 'agg' in locals() else None,
                        },
                        "plotly_json": fig_json
                    }
                    state_mgr.pin_item("chart", chart_data, f"{chart_type} - {ds_name}")
                    st.success("📌 Chart pinned! Access from sidebar.")
                except Exception as e:
                    st.error(f"Could not pin chart: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # CHART IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════════════════════
    try:
        # ── DISTRIBUTION & COMPARISON ─────────────────────────────────────────
        if chart_type == "Bar chart":
            c1, c2, c3 = st.columns(3)
            x_options = cat_cols or all_cols
            x_default = _get_viz_selection("bar_x")
            x_idx = x_options.index(x_default) if x_default in x_options else 0
            x   = c1.selectbox("Category (X)", x_options, index=x_idx)
            
            y_options = num_cols or all_cols
            y_default = _get_viz_selection("bar_y")
            y_idx = y_options.index(y_default) if y_default in y_options else 0
            y   = c2.selectbox("Value (Y)", y_options, index=y_idx)
            
            agg_options = ["sum","mean","count","max","min","median"]
            agg_default = _get_viz_selection("bar_agg", "sum")
            agg_idx = agg_options.index(agg_default) if agg_default in agg_options else 0
            agg = c3.selectbox("Aggregation", agg_options, index=agg_idx)
            
            _save_viz_selection("bar_x", x)
            _save_viz_selection("bar_y", y)
            _save_viz_selection("bar_agg", agg)
            
            # Cardinality check
            unique_vals = df[x].nunique()
            if unique_vals > 200:
                st.warning(f"⚠️ Column '{x}' has {unique_vals} unique values. Showing top 40 for performance.")
                top_n = st.slider("Show top N bars", 5, 100, 40)
            else:
                top_n = st.slider("Show top N bars", 5, 200, 20)
                
            df_p = df.groupby(x)[y].agg(agg).nlargest(top_n).reset_index()
            fig = px.bar(df_p, x=x, y=y, color=x, title=f"{agg.title()} of {y} by {x}",
                         text_auto=".2s", color_discrete_sequence=colors)
            fig.update_traces(textposition="outside")
            show_chart(fig)

        elif chart_type == "Grouped bar chart":
            c1, c2, c3 = st.columns(3)
            x     = c1.selectbox("X axis",    cat_cols or all_cols, key="gbx")
            y     = c2.selectbox("Y (value)", num_cols or all_cols, key="gby")
            color = c3.selectbox("Group by",  [c for c in cat_cols if c != x] or cat_cols, key="gbc")
            agg   = st.selectbox("Aggregation", ["sum","mean","count"], key="gba")
            df_p  = df.groupby([x, color])[y].agg(agg).reset_index()
            fig   = px.bar(df_p, x=x, y=y, color=color, barmode="group",
                           title=f"{agg.title()} of {y} by {x} grouped by {color}",
                           text_auto=".2s", color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Stacked bar chart":
            c1, c2, c3 = st.columns(3)
            x     = c1.selectbox("X axis",    cat_cols or all_cols, key="sbx")
            y     = c2.selectbox("Y (value)", num_cols or all_cols, key="sby")
            color = c3.selectbox("Stack by",  [c for c in cat_cols if c != x] or cat_cols, key="sbc")
            agg   = st.selectbox("Aggregation", ["sum","mean","count"], key="sba")
            df_p  = df.groupby([x, color])[y].agg(agg).reset_index()
            fig   = px.bar(df_p, x=x, y=y, color=color, barmode="stack",
                           title=f"Stacked: {agg.title()} of {y} by {x}",
                           color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Horizontal bar chart":
            c1, c2, c3 = st.columns(3)
            x   = c1.selectbox("Category", cat_cols or all_cols, key="hbx")
            y   = c2.selectbox("Value",    num_cols or all_cols, key="hby")
            agg = c3.selectbox("Aggregation", ["sum","mean","count","max","min"], key="hba")
            top_n = st.slider("Top N", 5, 50, 15, key="hbn")
            df_p = df.groupby(x)[y].agg(agg).nlargest(top_n).reset_index()
            fig = px.bar(df_p, y=x, x=y, orientation="h",
                         color=x, title=f"{agg.title()} of {y} by {x}",
                         text_auto=".2s", color_discrete_sequence=colors)
            fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            show_chart(fig)

        elif chart_type == "Lollipop chart":
            c1, c2, c3 = st.columns(3)
            x   = c1.selectbox("Category", cat_cols or all_cols, key="lolx")
            y   = c2.selectbox("Value",    num_cols or all_cols, key="loly")
            agg = c3.selectbox("Aggregation", ["sum","mean","count","max","min"], key="lola")
            top_n = st.slider("Top N", 5, 40, 15, key="loln")
            df_p = df.groupby(x)[y].agg(agg).nlargest(top_n).reset_index().sort_values(y)
            fig = go.Figure()
            for i, row in df_p.iterrows():
                fig.add_trace(go.Scatter(x=[0, row[y]], y=[row[x], row[x]],
                    mode="lines", line=dict(color=colors[0], width=2), showlegend=False))
            fig.add_trace(go.Scatter(x=df_p[y], y=df_p[x], mode="markers",
                marker=dict(size=12, color=colors[1], line=dict(width=2, color=colors[0])),
                name=y))
            fig.update_layout(title=f"Lollipop: {agg.title()} of {y} by {x}",
                              xaxis_title=y, yaxis_title=x)
            show_chart(fig)

        elif chart_type == "Dumbbell chart":
            if len(num_cols) < 2:
                st.warning("Need at least 2 numeric columns for a dumbbell chart (e.g., before/after values).")
            else:
                c1, c2, c3 = st.columns(3)
                cat  = c1.selectbox("Category", cat_cols or all_cols, key="dbc")
                val1 = c2.selectbox("Value 1 (left point)",  num_cols, key="dbv1")
                val2 = c3.selectbox("Value 2 (right point)", [c for c in num_cols if c != val1] or num_cols, key="dbv2")
                top_n = st.slider("Top N rows", 5, 40, 15, key="dbn")
                df_p = df.groupby(cat)[[val1, val2]].mean().head(top_n).reset_index().sort_values(val1)
                fig = go.Figure()
                for _, row in df_p.iterrows():
                    fig.add_trace(go.Scatter(x=[row[val1], row[val2]], y=[row[cat], row[cat]],
                        mode="lines", line=dict(color="gray", width=3), showlegend=False))
                fig.add_trace(go.Scatter(x=df_p[val1], y=df_p[cat], mode="markers",
                    marker=dict(size=12, color=colors[0]), name=val1))
                fig.add_trace(go.Scatter(x=df_p[val2], y=df_p[cat], mode="markers",
                    marker=dict(size=12, color=colors[1]), name=val2))
                fig.update_layout(title=f"Dumbbell: {val1} vs {val2} by {cat}",
                                  xaxis_title="Value")
                show_chart(fig)

        elif chart_type == "Histogram":
            c1, c2, c3 = st.columns(3)
            col   = c1.selectbox("Column",    num_cols or all_cols)
            bins  = c2.slider("Bins", 10, 200, 40)
            color = c3.selectbox("Color by",  ["None"] + cat_cols)
            norm  = st.selectbox("Normalize", ["count","percent","probability","density"])
            fig = px.histogram(df, x=col, nbins=bins,
                               color=None if color=="None" else color,
                               histnorm=None if norm=="count" else norm,
                               title=f"Distribution of {col}", marginal="box",
                               color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Box plot":
            c1, c2, c3 = st.columns(3)
            y      = c1.selectbox("Value",    num_cols)
            x      = c2.selectbox("Group by", ["None"] + cat_cols)
            color  = c3.selectbox("Color by", ["None"] + cat_cols)
            points = st.selectbox("Show points", ["outliers","all","none"])
            fig = px.box(df, y=y,
                         x=None if x=="None" else x,
                         color=None if color=="None" else color,
                         points=points, title=f"Box plot of {y}", notched=True,
                         color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Violin plot":
            c1, c2 = st.columns(2)
            y = c1.selectbox("Value",    num_cols)
            x = c2.selectbox("Group by", ["None"] + cat_cols)
            fig = px.violin(df, y=y, x=None if x=="None" else x,
                            box=True, points="outliers",
                            title=f"Violin plot of {y}",
                            color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Strip / dot plot":
            c1, c2 = st.columns(2)
            y = c1.selectbox("Value",    num_cols)
            x = c2.selectbox("Group by", ["None"] + cat_cols)
            color = st.selectbox("Color by", ["None"] + cat_cols, key="strip_c")
            fig = px.strip(df, y=y, x=None if x=="None" else x,
                           color=None if color=="None" else color,
                           title=f"Strip plot of {y}",
                           color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Error bar chart":
            if len(num_cols) < 1:
                st.warning("Need numeric columns.")
            else:
                c1, c2 = st.columns(2)
                x = c1.selectbox("Category", cat_cols or all_cols)
                y = c2.selectbox("Value",    num_cols)
                df_p = df.groupby(x)[y].agg(["mean","std"]).reset_index()
                df_p.columns = [x, "mean", "std"]
                fig = px.bar(df_p, x=x, y="mean", error_y="std",
                             title=f"Mean ± Std of {y} by {x}",
                             labels={"mean": f"Mean {y}"},
                             color_discrete_sequence=colors)
                show_chart(fig)

        elif chart_type == "Funnel chart":
            c1, c2 = st.columns(2)
            x = c1.selectbox("Stage/Category", cat_cols or all_cols)
            y = c2.selectbox("Value",          num_cols or all_cols)
            agg = st.selectbox("Aggregation", ["sum","mean","count"])
            df_p = df.groupby(x)[y].agg(agg).reset_index().sort_values(y, ascending=False)
            fig = px.funnel(df_p, x=y, y=x, title=f"Funnel: {y} by {x}",
                            color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Ridgeline / Joy plot":
            col = st.selectbox("Numeric column", num_cols, key="ridge_col")
            group = st.selectbox("Group by (categories)", cat_cols, key="ridge_grp")
            if col and group:
                from scipy.stats import gaussian_kde
                groups = df[group].value_counts().head(12).index.tolist()
                fig = go.Figure()
                for i, g in enumerate(groups):
                    data = df[df[group] == g][col].dropna()
                    if len(data) < 5:
                        continue
                    try:
                        kde = gaussian_kde(data)
                        x_range = np.linspace(data.min(), data.max(), 200)
                        y_vals = kde(x_range)
                        fig.add_trace(go.Scatter(
                            x=x_range, y=y_vals + i * y_vals.max() * 0.8,
                            fill="tozeroy" if i == 0 else "tonexty",
                            mode="lines", name=str(g),
                            line=dict(color=colors[i % len(colors)], width=1.5),
                        ))
                    except Exception:
                        continue
                fig.update_layout(title=f"Ridgeline: {col} by {group}",
                                  xaxis_title=col, yaxis_title="Density (offset)",
                                  showlegend=True)
                show_chart(fig, height=600)

        # ── TRENDS & TIME SERIES ──────────────────────────────────────────────
        elif chart_type == "Line chart":
            c1, c2 = st.columns(2)
            x = c1.selectbox("X axis (time/order)", date_cols + num_cols + cat_cols or all_cols)
            y = c2.selectbox("Y axis", num_cols)
            color = st.selectbox("Color by (optional)", ["None"] + cat_cols, key="lc_c")
            fig = px.line(df.sort_values(x), x=x, y=y,
                          color=None if color=="None" else color,
                          title=f"{y} over {x}", markers=True,
                          color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Multi-line chart":
            x = st.selectbox("X axis", date_cols + num_cols or all_cols)
            y_cols = st.multiselect("Y columns (select multiple)", num_cols, default=num_cols[:3])
            if y_cols:
                fig = px.line(df.sort_values(x), x=x, y=y_cols,
                              title=f"Multi-line: {', '.join(y_cols)} over {x}",
                              markers=False, color_discrete_sequence=colors)
                show_chart(fig)

        elif chart_type == "Area chart":
            c1, c2 = st.columns(2)
            x = c1.selectbox("X axis", date_cols + num_cols or all_cols)
            y = c2.selectbox("Y axis", num_cols)
            color = st.selectbox("Color by", ["None"] + cat_cols, key="area_c")
            fig = px.area(df.sort_values(x), x=x, y=y,
                          color=None if color=="None" else color,
                          title=f"Area chart: {y} over {x}",
                          color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Stacked area chart":
            c1, c2 = st.columns(2)
            x     = c1.selectbox("X axis", date_cols + num_cols or all_cols)
            color = c2.selectbox("Stack by", cat_cols or all_cols)
            y     = st.selectbox("Value", num_cols)
            agg   = st.selectbox("Aggregation", ["sum","mean"])
            df_p  = df.groupby([x, color])[y].agg(agg).reset_index()
            fig   = px.area(df_p.sort_values(x), x=x, y=y, color=color,
                            title=f"Stacked area: {y} over {x}",
                            color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Step line chart":
            c1, c2 = st.columns(2)
            x = c1.selectbox("X axis (time/order)", date_cols + num_cols or all_cols, key="slx")
            y = c2.selectbox("Y axis", num_cols, key="sly")
            color = st.selectbox("Color by", ["None"] + cat_cols, key="slc")
            fig = px.line(df.sort_values(x), x=x, y=y,
                          color=None if color=="None" else color,
                          title=f"Step line: {y} over {x}",
                          line_shape="hv", color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Candlestick (OHLC)":
            st.info("Candlestick charts need Open, High, Low, Close columns.")
            c1, c2, c3, c4, c5 = st.columns(5)
            date_c = c1.selectbox("Date", all_cols)
            open_c = c2.selectbox("Open",  num_cols)
            high_c = c3.selectbox("High",  num_cols)
            low_c  = c4.selectbox("Low",   num_cols)
            close_c= c5.selectbox("Close", num_cols)
            df_s   = df.sort_values(date_c)
            fig    = go.Figure(go.Candlestick(
                x=df_s[date_c], open=df_s[open_c], high=df_s[high_c],
                low=df_s[low_c], close=df_s[close_c],
                increasing_line_color=colors[2], decreasing_line_color=colors[1]))
            fig.update_layout(title="Candlestick (OHLC) Chart", xaxis_rangeslider_visible=True)
            show_chart(fig)

        elif chart_type == "Time series with trend line":
            c1, c2 = st.columns(2)
            x = c1.selectbox("X (time)", date_cols + num_cols or all_cols)
            y = c2.selectbox("Y value",  num_cols)
            trendline = st.selectbox("Trend line", ["ols","lowess","none"])
            fig = px.scatter(df.sort_values(x), x=x, y=y,
                             trendline=None if trendline=="none" else trendline,
                             title=f"{y} over {x} with {trendline} trend",
                             color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "Area with range band":
            c1, c2 = st.columns(2)
            x = c1.selectbox("X axis", date_cols + num_cols or all_cols, key="arb_x")
            y = c2.selectbox("Y axis", num_cols, key="arb_y")
            window = st.slider("Rolling window", 2, 50, 7, key="arb_w")
            df_s = df.sort_values(x).copy()
            df_s["_mean"] = df_s[y].rolling(window=window, min_periods=1).mean()
            df_s["_std"]  = df_s[y].rolling(window=window, min_periods=1).std().fillna(0)
            df_s["_upper"] = df_s["_mean"] + df_s["_std"]
            df_s["_lower"] = df_s["_mean"] - df_s["_std"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_s[x], y=df_s["_upper"], mode="lines",
                line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df_s[x], y=df_s["_lower"], mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor="rgba(59,130,246,0.2)",
                name="±1 Std Dev"))
            fig.add_trace(go.Scatter(x=df_s[x], y=df_s["_mean"], mode="lines",
                line=dict(color=colors[0], width=2), name=f"Rolling mean ({window})"))
            fig.add_trace(go.Scatter(x=df_s[x], y=df_s[y], mode="markers",
                marker=dict(size=3, color=colors[1], opacity=0.4), name="Raw data"))
            fig.update_layout(title=f"Area with Range Band: {y}")
            show_chart(fig)

        elif chart_type == "Seasonal decomposition":
            col = st.selectbox("Value column", num_cols, key="sd_col")
            period = st.slider("Season period (e.g., 7=weekly, 12=monthly)", 2, 365, 12, key="sd_p")
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                data = df[col].dropna().reset_index(drop=True)
                if len(data) < period * 2:
                    st.warning(f"Need at least {period*2} data points for decomposition.")
                else:
                    result = seasonal_decompose(data, model="additive", period=period)
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
                                        vertical_spacing=0.05)
                    fig.add_trace(go.Scatter(y=result.observed, mode="lines", name="Observed",
                                            line=dict(color=colors[0])), row=1, col=1)
                    fig.add_trace(go.Scatter(y=result.trend, mode="lines", name="Trend",
                                            line=dict(color=colors[1], width=2)), row=2, col=1)
                    fig.add_trace(go.Scatter(y=result.seasonal, mode="lines", name="Seasonal",
                                            line=dict(color=colors[2])), row=3, col=1)
                    fig.add_trace(go.Scatter(y=result.resid, mode="markers", name="Residual",
                                            marker=dict(size=3, color=colors[3])), row=4, col=1)
                    fig.update_layout(title=f"Seasonal Decomposition: {col} (period={period})", height=700)
                    show_chart(fig, height=700)
            except ImportError:
                st.error("statsmodels required. Restart the app to auto-install.")

        # ── RELATIONSHIPS & CORRELATIONS ──────────────────────────────────────
        elif chart_type == "Scatter plot":
            c1, c2, c3, c4 = st.columns(4)
            x     = c1.selectbox("X axis", num_cols)
            y     = c2.selectbox("Y axis", num_cols[1:] if len(num_cols) > 1 else num_cols)
            color = c3.selectbox("Color",  ["None"] + cat_cols + num_cols)
            size  = c4.selectbox("Size",   ["None"] + num_cols)
            fig = px.scatter(df, x=x, y=y,
                             color=None if color=="None" else color,
                             size=None if size=="None" else size,
                             title=f"{x} vs {y}", opacity=0.7,
                             trendline="ols", hover_data=all_cols[:4],
                             color_discrete_sequence=colors)
            show_chart(fig)
            if len(num_cols) >= 2:
                corr_val = df[[x, y]].corr().iloc[0, 1]
                st.metric("Pearson correlation (r)", f"{corr_val:.3f}",
                          delta="Strong" if abs(corr_val) > 0.7 else "Moderate" if abs(corr_val) > 0.4 else "Weak")

        elif chart_type == "Bubble chart":
            c1, c2, c3, c4 = st.columns(4)
            x     = c1.selectbox("X",     num_cols)
            y     = c2.selectbox("Y",     num_cols[1:] if len(num_cols) > 1 else num_cols)
            size  = c3.selectbox("Size",  num_cols)
            color = c4.selectbox("Color", ["None"] + cat_cols)
            text  = st.selectbox("Label (optional)", ["None"] + cat_cols)
            fig = px.scatter(df, x=x, y=y, size=size,
                             color=None if color=="None" else color,
                             text=None if text=="None" else text,
                             title=f"Bubble chart: {x} vs {y}", size_max=60, opacity=0.7,
                             color_discrete_sequence=colors)
            show_chart(fig)

        elif chart_type == "3D scatter plot":
            if len(num_cols) < 3:
                st.warning("Need at least 3 numeric columns for a 3D scatter.")
            else:
                c1, c2, c3, c4 = st.columns(4)
                x     = c1.selectbox("X", num_cols)
                y     = c2.selectbox("Y", num_cols[1:] if len(num_cols) > 1 else num_cols)
                z     = c3.selectbox("Z", num_cols[2:] if len(num_cols) > 2 else num_cols)
                color = c4.selectbox("Color", ["None"] + cat_cols)
                fig = px.scatter_3d(df, x=x, y=y, z=z,
                                    color=None if color=="None" else color,
                                    title=f"3D Scatter: {x}, {y}, {z}",
                                    opacity=0.7, color_discrete_sequence=colors)
                show_chart(fig)

        elif chart_type == "3D surface plot":
            if len(num_cols) < 3:
                st.warning("Need at least 3 numeric columns for a 3D surface plot.")
            else:
                x = st.selectbox("X axis", num_cols, key="surface_x")
                y = st.selectbox("Y axis", [c for c in num_cols if c != x] or num_cols, key="surface_y")
                z = st.selectbox("Z axis", [c for c in num_cols if c not in [x, y]] or num_cols, key="surface_z")
                create_advanced_chart(df, "3d_surface", x_col=x, y_col=y, z_col=z)
                
                # Trigger workflow handoff for chart creation
                trigger_charts_created_handoff({
                    "chart_type": "3D surface plot",
                    "dataset": ds_name,
                    "columns": [x, y, z]
                })
                
                # Show handoff actions
                render_handoff_actions(
                    payload_type=HandoffType.CHART,
                    source_page="📊 Visualize",
                    title=f"3D Surface Plot",
                    description=f"3D surface plot of {x}, {y}, {z}",
                    data={
                        "chart_type": "3d_surface",
                        "dataset": ds_name,
                        "columns": [x, y, z]
                    }
                )
                
                if st.button("📌 Pin to Report", key="pin_3dsurface", width="stretch"):
                    state_mgr.pin_item("chart", {"chart_type": "3D surface plot", "dataset": ds_name}, f"3D surface plot — {ds_name}")
                    st.success("📌 Pinned to Report Studio!")

        elif chart_type == "Interactive dashboard":
            selected_columns = st.multiselect("Select columns for dashboard", num_cols + cat_cols, default=num_cols[:4])
            if selected_columns:
                create_advanced_chart(df, "interactive_dashboard", columns=selected_columns)
                if st.button("📌 Pin to Report", key="pin_idashboard", width="stretch"):
                    state_mgr.pin_item("chart", {"chart_type": "Interactive dashboard", "dataset": ds_name}, f"Interactive dashboard — {ds_name}")
                    st.success("📌 Pinned to Report Studio!")

        elif chart_type == "Geospatial map":
            lat_cols = [c for c in all_cols if any(k in c.lower() for k in ["lat","latitude","y_coord","y_pos","gps_lat","lat_deg"])]
            lon_cols = [c for c in all_cols if any(k in c.lower() for k in ["lon","long","longitude","x_coord","x_pos","gps_lon","xlon","lng","lon_deg"])]
            if not lat_cols or not lon_cols:
                st.warning("No latitude/longitude columns auto-detected. Please select manually.")
            lat = st.selectbox("Latitude", lat_cols or all_cols, key="geo_lat")
            lon = st.selectbox("Longitude", lon_cols or all_cols, key="geo_lon")
            popup = st.selectbox("Popup column", ["None"] + all_cols, key="geo_popup")
            color = st.selectbox("Color by", ["None"] + cat_cols, key="geo_color")
            create_advanced_chart(
                df,
                "geospatial_map",
                lat_col=lat,
                lon_col=lon,
                popup_col=None if popup == "None" else popup,
                color_col=None if color == "None" else color,
            )
            if st.button("📌 Pin to Report", key="pin_geomap", width="stretch"):
                state_mgr.pin_item("chart", {"chart_type": "Geospatial map", "dataset": ds_name}, f"Geospatial map — {ds_name}")
                st.success("📌 Pinned to Report Studio!")

        elif chart_type == "Network graph":
            source = st.selectbox("Source column", cat_cols + all_cols, key="net_src")
            target = st.selectbox("Target column", [c for c in (cat_cols + all_cols) if c != source] or all_cols, key="net_tgt")
            weight = st.selectbox("Weight column", ["None"] + num_cols, key="net_weight")
            edges_df = df[[source, target] + ([] if weight == "None" else [weight])].dropna()
            nodes_df = pd.DataFrame({"node": pd.concat([edges_df[source], edges_df[target]]).unique()})
            create_advanced_chart(
                nodes_df,
                "network_graph",
                edges_df=edges_df,
                source_col=source,
                target_col=target,
                weight_col=None if weight == "None" else weight,
            )
            if st.button("📌 Pin to Report", key="pin_network", width="stretch"):
                state_mgr.pin_item("chart", {"chart_type": "Network graph", "dataset": ds_name}, f"Network graph — {ds_name}")
                st.success("📌 Pinned to Report Studio!")

        elif chart_type == "Scatter matrix (pair plot)":
            selected = st.multiselect("Select columns (4-6 recommended)", num_cols,
                                      default=num_cols[:5])
            color = st.selectbox("Color by", ["None"] + cat_cols)
            if selected and len(selected) >= 2:
                fig = px.scatter_matrix(df, dimensions=selected,
                                        color=None if color=="None" else color,
                                        title="Scatter Matrix (Pair Plot)",
                                        opacity=0.5, color_discrete_sequence=colors)
                fig.update_traces(diagonal_visible=True, showupperhalf=False)
                show_chart(fig, height=700)

        elif chart_type == "Correlation heatmap":
            selected = st.multiselect("Select numeric columns", num_cols, default=num_cols[:12])
            method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], key="corr_m")
            if selected and len(selected) >= 2:
                corr = df[selected].corr(method=method).round(2)
                fig  = px.imshow(corr, color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1,
                                 title=f"Correlation Heatmap ({method.title()})", text_auto=True,
                                 aspect="auto")
                show_chart(fig, height=max(400, len(selected) * 40))
                strong = [(a, b, corr.loc[a, b]) for a in selected for b in selected
                          if a < b and abs(corr.loc[a, b]) > 0.6]
                if strong:
                    st.markdown("**Strong correlations (|r| > 0.6):**")
                    for a, b, v in sorted(strong, key=lambda t: abs(t[2]), reverse=True)[:8]:
                        bar = "🔴" if v > 0 else "🔵"
                        st.markdown(f"{bar} **{a}** ↔ **{b}**: {v:.2f}")

        elif chart_type == "Parallel coordinates":
            selected = st.multiselect("Select columns", num_cols, default=num_cols[:6])
            color    = st.selectbox("Color by", ["None"] + num_cols + cat_cols)
            if selected:
                color_col = None if color == "None" else color
                if color_col and color_col in cat_cols:
                    df2 = df.copy()
                    from sklearn.preprocessing import LabelEncoder
                    df2[color_col] = LabelEncoder().fit_transform(df2[color_col].astype(str))
                    color_col_plot = color_col
                else:
                    df2 = df
                    color_col_plot = color_col
                fig = px.parallel_coordinates(df2, dimensions=selected,
                                              color=color_col_plot,
                                              color_continuous_scale="Viridis",
                                              title="Parallel Coordinates Plot")
                show_chart(fig)

        elif chart_type == "Radar / spider chart":
            c1, c2 = st.columns(2)
            metrics = c1.multiselect("Metrics (numeric)", num_cols, default=num_cols[:6])
            group   = c2.selectbox("Group by", cat_cols or all_cols)
            top_n   = st.slider("Top N groups", 2, 10, 5, key="radar_n")
            if metrics:
                top_groups = df[group].value_counts().head(top_n).index.tolist()
                df_p = df[df[group].isin(top_groups)].groupby(group)[metrics].mean().reset_index()
                fig = go.Figure()
                for idx, row in df_p.iterrows():
                    vals = row[metrics].tolist()
                    fig.add_trace(go.Scatterpolar(r=vals + [vals[0]],
                                                   theta=metrics + [metrics[0]],
                                                   fill="toself", name=str(row[group]),
                                                   line=dict(color=colors[idx % len(colors)])))
                fig.update_layout(title=f"Radar Chart: {', '.join(metrics[:3])}... by {group}",
                                  polar=dict(radialaxis=dict(visible=True)))
                show_chart(fig)

        elif chart_type == "Hexbin plot":
            if len(num_cols) < 2:
                st.warning("Need at least 2 numeric columns.")
            else:
                c1, c2 = st.columns(2)
                x = c1.selectbox("X", num_cols, key="hex_x")
                y = c2.selectbox("Y", num_cols[1:] if len(num_cols) > 1 else num_cols, key="hex_y")
                nbins = st.slider("Bin resolution", 10, 80, 30, key="hex_b")
                fig = px.density_heatmap(df, x=x, y=y, nbinsx=nbins, nbinsy=nbins,
                                         color_continuous_scale="Viridis",
                                         title=f"Hexbin Density: {x} vs {y}")
                fig.update_traces(xgap=1, ygap=1)
                show_chart(fig)

        elif chart_type == "Regression with confidence bands":
            if len(num_cols) < 2:
                st.warning("Need at least 2 numeric columns.")
            else:
                c1, c2 = st.columns(2)
                x = c1.selectbox("X (predictor)", num_cols, key="reg_x")
                y = c2.selectbox("Y (response)", num_cols[1:] if len(num_cols) > 1 else num_cols, key="reg_y")
                from scipy import stats as sp_stats
                d = df[[x, y]].dropna()
                slope, intercept, r_value, p_value, std_err = sp_stats.linregress(d[x], d[y])
                x_line = np.linspace(d[x].min(), d[x].max(), 200)
                y_line = slope * x_line + intercept
                y_pred = slope * d[x] + intercept
                residuals = d[y] - y_pred
                se = np.sqrt(np.sum(residuals**2) / (len(d) - 2))
                ci = 1.96 * se * np.sqrt(1/len(d) + (x_line - d[x].mean())**2 / np.sum((d[x] - d[x].mean())**2))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=d[x], y=d[y], mode="markers",
                    marker=dict(size=5, color=colors[0], opacity=0.5), name="Data"))
                fig.add_trace(go.Scatter(x=x_line, y=y_line,
                    mode="lines", line=dict(color=colors[1], width=2),
                    name=f"OLS (R²={r_value**2:.3f})"))
                fig.add_trace(go.Scatter(x=np.concatenate([x_line, x_line[::-1]]),
                    y=np.concatenate([y_line + ci, (y_line - ci)[::-1]]),
                    fill="toself", fillcolor="rgba(239,68,68,0.15)",
                    line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
                fig.update_layout(title=f"Regression: {y} ~ {x} (R²={r_value**2:.3f}, p={p_value:.4f})")
                show_chart(fig)
                c1, c2, c3 = st.columns(3)
                c1.metric("R²", f"{r_value**2:.4f}")
                c2.metric("p-value", f"{p_value:.4f}")
                c3.metric("Slope", f"{slope:.4f}")

        # ── HIERARCHICAL & PART-TO-WHOLE ──────────────────────────────────────
        elif chart_type == "Pie chart":
            c1, c2 = st.columns(2)
            names  = c1.selectbox("Category", cat_cols or all_cols)
            values = c2.selectbox("Value",    num_cols or all_cols)
            top_n  = st.slider("Show top N slices", 3, 30, 10)
            df_p = df.groupby(names)[values].sum().nlargest(top_n).reset_index()
            fig = px.pie(df_p, names=names, values=values,
                         title=f"Pie chart: {values} by {names}",
                         color_discrete_sequence=colors)
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              hoverinfo="label+percent+value")
            show_chart(fig)

        elif chart_type == "Donut chart":
            c1, c2 = st.columns(2)
            names  = c1.selectbox("Category", cat_cols or all_cols, key="don_n")
            values = c2.selectbox("Value",    num_cols or all_cols, key="don_v")
            top_n  = st.slider("Top N", 3, 30, 10, key="don_t")
            df_p = df.groupby(names)[values].sum().nlargest(top_n).reset_index()
            fig = px.pie(df_p, names=names, values=values, hole=0.45,
                         title=f"Donut chart: {values} by {names}",
                         color_discrete_sequence=colors)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            show_chart(fig)

        elif chart_type == "Treemap":
            path_cols = st.multiselect("Hierarchy levels (broad → specific)",
                                       cat_cols or all_cols,
                                       default=cat_cols[:2] if len(cat_cols) >= 2 else cat_cols[:1])
            values = st.selectbox("Value (size)", num_cols or all_cols)
            if path_cols:
                fig = px.treemap(df, path=path_cols, values=values,
                                 title=f"Treemap: {values} by {' → '.join(path_cols)}",
                                 color=values, color_continuous_scale="Blues")
                show_chart(fig)

        elif chart_type == "Sunburst chart":
            path_cols = st.multiselect("Hierarchy levels",
                                       cat_cols or all_cols,
                                       default=cat_cols[:2] if len(cat_cols) >= 2 else cat_cols[:1],
                                       key="sun_p")
            values = st.selectbox("Value", num_cols or all_cols, key="sun_v")
            if path_cols:
                if values in num_cols:
                    df_s = df.groupby(path_cols)[values].sum().reset_index()
                    value_column = values
                else:
                    df_s = df.groupby(path_cols).size().reset_index(name="count")
                    value_column = "count"

                fig = px.sunburst(df_s, path=path_cols, values=value_column,
                                  title=f"Sunburst: {value_column} by {' → '.join(path_cols)}",
                                  color=value_column, color_continuous_scale="Viridis")
                show_chart(fig)

        elif chart_type == "Icicle chart":
            path_cols = st.multiselect("Hierarchy levels",
                                       cat_cols or all_cols,
                                       default=cat_cols[:2] if len(cat_cols) >= 2 else cat_cols[:1],
                                       key="ice_p")
            values = st.selectbox("Value", num_cols or all_cols, key="ice_v")
            if path_cols:
                fig = px.icicle(df, path=path_cols, values=values,
                                title=f"Icicle: {values} by {' → '.join(path_cols)}",
                                color=values, color_continuous_scale="Blues")
                show_chart(fig)

        elif chart_type == "Waterfall chart":
            c1, c2 = st.columns(2)
            x = c1.selectbox("Category (steps)", cat_cols or all_cols, key="wf_x")
            y = c2.selectbox("Value",            num_cols, key="wf_y")
            agg = st.selectbox("Aggregation", ["sum","mean"], key="wf_a")
            df_p = df.groupby(x)[y].agg(agg).reset_index().head(20)
            fig = go.Figure(go.Waterfall(
                name="", orientation="v",
                x=df_p[x].astype(str).tolist(),
                y=df_p[y].tolist(),
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                increasing={"marker": {"color": colors[2]}},
                decreasing={"marker": {"color": colors[1]}},
                totals={"marker": {"color": colors[0]}},
            ))
            fig.update_layout(title=f"Waterfall: {y} by {x}", showlegend=False)
            show_chart(fig)

        elif chart_type == "Sankey diagram":
            c1, c2, c3 = st.columns(3)
            source = c1.selectbox("Source",  cat_cols or all_cols, key="sk_s")
            target = c2.selectbox("Target",  [c for c in cat_cols if c != source] or cat_cols, key="sk_t")
            value  = c3.selectbox("Flow value", num_cols, key="sk_v")
            agg    = st.selectbox("Aggregation", ["sum","mean","count"], key="sk_a")
            
            # Zero-Loss Resilience: Flow limit slider
            n_flows = st.slider("Total flows to visualize", 5, 500, 50, key="sk_nf")
            df_p = df.groupby([source, target])[value].agg(agg).reset_index().sort_values(value, ascending=False).head(n_flows)
            all_nodes = list(pd.concat([df_p[source], df_p[target]]).unique())
            node_map  = {n: i for i, n in enumerate(all_nodes)}
            node_colors = [colors[i % len(colors)] for i in range(len(all_nodes))]
            fig = go.Figure(go.Sankey(
                node=dict(label=all_nodes, pad=15, thickness=20, color=node_colors),
                link=dict(
                    source=[node_map[s] for s in df_p[source]],
                    target=[node_map[t] for t in df_p[target]],
                    value=df_p[value].tolist(),
                )
            ))
            fig.update_layout(title=f"Sankey: flow of {value} from {source} to {target}")
            show_chart(fig)

        elif chart_type == "Marimekko chart":
            c1, c2, c3 = st.columns(3)
            x_cat = c1.selectbox("X Category", cat_cols or all_cols, key="mk_x")
            y_cat = c2.selectbox("Y Category", [c for c in cat_cols if c != x_cat] or cat_cols, key="mk_y")
            val   = c3.selectbox("Value", num_cols, key="mk_v")
            df_ct = df.groupby([x_cat, y_cat])[val].sum().reset_index()
            x_totals = df_ct.groupby(x_cat)[val].sum()
            widths = (x_totals / x_totals.sum()).to_dict()
            fig = go.Figure()
            y_cats = df_ct[y_cat].unique()
            for i, yc in enumerate(y_cats):
                sub = df_ct[df_ct[y_cat] == yc].set_index(x_cat)
                heights = []
                for xc in x_totals.index:
                    v = sub.loc[xc, val] if xc in sub.index else 0
                    heights.append(v / x_totals[xc] * 100)
                fig.add_trace(go.Bar(
                    x=list(x_totals.index), y=heights,
                    name=str(yc), width=[widths.get(xc, 0.1) for xc in x_totals.index],
                    marker_color=colors[i % len(colors)]))
            fig.update_layout(barmode="stack", title=f"Marimekko: {val} by {x_cat} / {y_cat}",
                              yaxis_title="Percentage (%)", xaxis_title=x_cat)
            show_chart(fig)

        # ── GEOGRAPHIC & DENSITY ──────────────────────────────────────────────
        elif chart_type == "Density heatmap (2D)":
            c1, c2 = st.columns(2)
            x = c1.selectbox("X axis", num_cols)
            y = c2.selectbox("Y axis", num_cols[1:] if len(num_cols) > 1 else num_cols)
            fig = px.density_heatmap(df, x=x, y=y, nbinsx=40, nbinsy=40,
                                     title=f"Density Heatmap: {x} vs {y}",
                                     color_continuous_scale="Hot", text_auto=True)
            show_chart(fig)

        elif chart_type == "Contour plot":
            c1, c2 = st.columns(2)
            x = c1.selectbox("X", num_cols, key="cnt_x")
            y = c2.selectbox("Y", num_cols[1:] if len(num_cols) > 1 else num_cols, key="cnt_y")
            fig = go.Figure(go.Histogram2dContour(x=df[x], y=df[y],
                                                   colorscale="Hot", reversescale=True))
            fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="markers",
                                     marker=dict(color="white", size=3, opacity=0.3)))
            fig.update_layout(title=f"Contour plot: {x} vs {y}")
            show_chart(fig)

        elif chart_type == "Geographic scatter map":
            lat_keywords = ["lat","latitude","y_coord","y_pos","gps_lat","ylat","lat_deg"]
            lon_keywords = ["lon","long","longitude","x_coord","x_pos","gps_lon","xlon","lng","lon_deg"]
            lat_cols = [c for c in num_cols if any(k in c.lower() for k in lat_keywords)]
            lon_cols = [c for c in num_cols if any(k in c.lower() for k in lon_keywords)]
            if not lat_cols or not lon_cols:
                st.warning("No lat/lon columns auto-detected. Select manually below.")
                lat_cols = num_cols; lon_cols = num_cols
            c1, c2, c3 = st.columns(3)
            lat   = c1.selectbox("Latitude",  lat_cols)
            lon   = c2.selectbox("Longitude", lon_cols)
            color = c3.selectbox("Color by",  ["None"] + cat_cols + num_cols, key="geo_c")
            size  = st.selectbox("Size by",   ["None"] + num_cols, key="geo_s")
            fig = px.scatter_mapbox(
                df.dropna(subset=[lat, lon]),
                lat=lat, lon=lon,
                color=None if color=="None" else color,
                size=None if size=="None" else size,
                mapbox_style="carto-darkmatter",
                zoom=3, title="Geographic Scatter Map",
                color_discrete_sequence=colors,
            )
            show_chart(fig)

        elif chart_type == "Choropleth map":
            st.markdown("**Choropleth maps** color geographic regions by value. Select a column with country/state codes.")
            c1, c2 = st.columns(2)
            location = c1.selectbox("Location column (ISO codes, country names, states)", cat_cols or all_cols, key="choro_loc")
            value_col = c2.selectbox("Value column", num_cols, key="choro_val")
            loc_mode = st.selectbox("Location type", ["country names", "ISO-3", "USA-states"], key="choro_mode")
            mode_map = {"country names": "country names", "ISO-3": "ISO-3", "USA-states": "USA-states"}
            agg = st.selectbox("Aggregation", ["sum","mean","count"], key="choro_agg")
            df_p = df.groupby(location)[value_col].agg(agg).reset_index()
            fig = px.choropleth(df_p, locations=location, locationmode=mode_map[loc_mode],
                                color=value_col, title=f"Choropleth: {value_col} by {location}",
                                color_continuous_scale="Viridis")
            fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
            show_chart(fig)

        elif chart_type == "Animated scatter map":
            lat_keywords = ["lat","latitude"]
            lon_keywords = ["lon","long","longitude","lng"]
            lat_cols_a = [c for c in num_cols if any(k in c.lower() for k in lat_keywords)] or num_cols
            lon_cols_a = [c for c in num_cols if any(k in c.lower() for k in lon_keywords)] or num_cols
            c1, c2, c3, c4 = st.columns(4)
            lat   = c1.selectbox("Latitude",  lat_cols_a, key="anim_lat")
            lon   = c2.selectbox("Longitude", lon_cols_a, key="anim_lon")
            anim  = c3.selectbox("Animation frame", cat_cols + date_cols or all_cols, key="anim_f")
            color = c4.selectbox("Color by", ["None"] + cat_cols, key="anim_c")
            fig = px.scatter_mapbox(
                df.dropna(subset=[lat, lon]).sort_values(anim),
                lat=lat, lon=lon,
                color=None if color=="None" else color,
                animation_frame=anim,
                mapbox_style="carto-darkmatter",
                zoom=2, title="Animated Scatter Map",
                color_discrete_sequence=colors,
            )
            show_chart(fig, height=600)

        # ── STATISTICAL & ADVANCED ────────────────────────────────────────────
        elif chart_type == "Q-Q plot (normality)":
            col = st.selectbox("Column to test normality", num_cols)
            from scipy import stats as scipy_stats
            data_clean = df[col].dropna()
            (osm, osr), (slope, intercept, r) = scipy_stats.probplot(data_clean)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(osm), y=list(osr), mode="markers",
                                     name="Data points", marker=dict(size=4, opacity=0.6, color=colors[0])))
            x_line = [min(osm), max(osm)]
            y_line = [slope * x_line[0] + intercept, slope * x_line[1] + intercept]
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                     name="Normal reference", line=dict(color=colors[1], dash="dash")))
            fig.update_layout(title=f"Q-Q Plot: {col} (R²={r**2:.3f})",
                              xaxis_title="Theoretical quantiles",
                              yaxis_title="Sample quantiles")
            show_chart(fig)
            if abs(r) > 0.98:
                st.success(f"✅ {col} appears approximately normally distributed (R²={r**2:.3f})")
            else:
                st.warning(f"⚠️ {col} deviates from normality (R²={r**2:.3f})")

        elif chart_type == "Cumulative distribution (CDF)":
            cols_sel = st.multiselect("Columns", num_cols, default=num_cols[:3])
            if cols_sel:
                fig = go.Figure()
                for i, col in enumerate(cols_sel):
                    d = df[col].dropna().sort_values()
                    cdf = np.arange(1, len(d) + 1) / len(d)
                    fig.add_trace(go.Scatter(x=d, y=cdf, mode="lines", name=col,
                                            line=dict(color=colors[i % len(colors)])))
                fig.update_layout(title="Cumulative Distribution Function (CDF)",
                                  xaxis_title="Value", yaxis_title="Cumulative probability")
                show_chart(fig)

        elif chart_type == "Density distribution (KDE)":
            cols_sel = st.multiselect("Columns", num_cols, default=num_cols[:3], key="kde_c")
            if cols_sel:
                from scipy.stats import gaussian_kde
                fig = go.Figure()
                for i, col in enumerate(cols_sel):
                    d = df[col].dropna()
                    if len(d) < 5: continue
                    kde = gaussian_kde(d)
                    x_range = np.linspace(d.min(), d.max(), 300)
                    fig.add_trace(go.Scatter(x=x_range, y=kde(x_range),
                                             mode="lines", name=col, fill="tozeroy",
                                             line=dict(color=colors[i % len(colors)])))
                fig.update_layout(title="Kernel Density Estimation (KDE)",
                                  xaxis_title="Value", yaxis_title="Density")
                show_chart(fig)

        elif chart_type == "Outlier comparison chart":
            col = st.selectbox("Column", num_cols, key="out_c")
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df_o = df.copy()
            df_o["Type"] = np.where((df_o[col] < lower) | (df_o[col] > upper), "Outlier", "Normal")
            fig = px.scatter(df_o.reset_index(), x="index", y=col, color="Type",
                             title=f"Outlier detection: {col}",
                             color_discrete_map={"Normal": colors[0], "Outlier": colors[1]})
            fig.add_hline(y=lower, line_dash="dash", line_color="orange",
                          annotation_text=f"Lower: {lower:.2f}")
            fig.add_hline(y=upper, line_dash="dash", line_color="orange",
                          annotation_text=f"Upper: {upper:.2f}")
            show_chart(fig)
            n_out = (df_o["Type"] == "Outlier").sum()
            st.metric("Outliers detected", f"{n_out:,}", delta=f"{n_out / len(df) * 100:.1f}% of data")

        elif chart_type == "Rolling average chart":
            c1, c2, c3 = st.columns(3)
            x      = c1.selectbox("X axis (time/order)", date_cols + num_cols + all_cols)
            y      = c2.selectbox("Y value", num_cols)
            window = c3.slider("Rolling window", 2, 50, 7)
            df_s   = df.sort_values(x).copy()
            df_s[f"{y}_rolling"] = df_s[y].rolling(window=window, min_periods=1).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_s[x], y=df_s[y], mode="lines",
                                     name="Raw", opacity=0.3, line=dict(color=colors[0])))
            fig.add_trace(go.Scatter(x=df_s[x], y=df_s[f"{y}_rolling"], mode="lines",
                                     name=f"{window}-period rolling avg",
                                     line=dict(color=colors[1], width=2.5)))
            fig.update_layout(title=f"Rolling Average ({window} periods): {y}")
            show_chart(fig)

        elif chart_type == "Pareto chart":
            c1, c2 = st.columns(2)
            x   = c1.selectbox("Category", cat_cols or all_cols, key="par_x")
            y   = c2.selectbox("Value",    num_cols, key="par_y")
            agg = st.selectbox("Aggregation", ["sum","count","mean"], key="par_a")
            df_p = df.groupby(x)[y].agg(agg).sort_values(ascending=False).head(30).reset_index()
            df_p["cumulative_pct"] = df_p[y].cumsum() / df_p[y].sum() * 100
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df_p[x], y=df_p[y], name=y,
                                 marker_color=colors[0]), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_p[x], y=df_p["cumulative_pct"],
                                     mode="lines+markers", name="Cumulative %",
                                     line=dict(color=colors[1])), secondary_y=True)
            fig.add_hline(y=80, secondary_y=True, line_dash="dash",
                          line_color="orange", annotation_text="80%")
            fig.update_layout(title=f"Pareto Chart: {y} by {x}")
            fig.update_yaxes(title_text=y, secondary_y=False)
            fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])
            show_chart(fig)

        elif chart_type == "Bland-Altman plot":
            if len(num_cols) < 2:
                st.warning("Need at least 2 numeric columns to compare methods.")
            else:
                c1, c2 = st.columns(2)
                m1 = c1.selectbox("Method 1 (column)", num_cols, key="ba_m1")
                m2 = c2.selectbox("Method 2 (column)", [c for c in num_cols if c != m1] or num_cols, key="ba_m2")
                d = df[[m1, m2]].dropna()
                mean_vals = (d[m1] + d[m2]) / 2
                diff_vals = d[m1] - d[m2]
                mean_diff = diff_vals.mean()
                std_diff  = diff_vals.std()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=mean_vals, y=diff_vals, mode="markers",
                    marker=dict(size=5, color=colors[0], opacity=0.6), name="Observations"))
                fig.add_hline(y=mean_diff, line_dash="solid", line_color=colors[2],
                              annotation_text=f"Mean diff: {mean_diff:.3f}")
                fig.add_hline(y=mean_diff + 1.96 * std_diff, line_dash="dash", line_color=colors[1],
                              annotation_text=f"+1.96 SD: {mean_diff + 1.96*std_diff:.3f}")
                fig.add_hline(y=mean_diff - 1.96 * std_diff, line_dash="dash", line_color=colors[1],
                              annotation_text=f"-1.96 SD: {mean_diff - 1.96*std_diff:.3f}")
                fig.update_layout(title=f"Bland-Altman: {m1} vs {m2}",
                                  xaxis_title=f"Mean of {m1} and {m2}",
                                  yaxis_title=f"Difference ({m1} − {m2})")
                show_chart(fig)

        elif chart_type == "Residual plot":
            if len(num_cols) < 2:
                st.warning("Need at least 2 numeric columns.")
            else:
                c1, c2 = st.columns(2)
                x = c1.selectbox("X (predictor)", num_cols, key="res_x")
                y = c2.selectbox("Y (response)", [c for c in num_cols if c != x] or num_cols, key="res_y")
                d = df[[x, y]].dropna()
                from scipy import stats as sp_stats2
                slope, intercept, r, p, se = sp_stats2.linregress(d[x], d[y])
                predicted = slope * d[x] + intercept
                residuals = d[y] - predicted
                fig = make_subplots(rows=1, cols=2, subplot_titles=["Residuals vs Fitted", "Residual Distribution"])
                fig.add_trace(go.Scatter(x=predicted, y=residuals, mode="markers",
                    marker=dict(size=4, color=colors[0], opacity=0.5), name="Residuals"), row=1, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color=colors[1], row=1, col=1)
                fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name="Distribution",
                    marker_color=colors[2]), row=1, col=2)
                fig.update_layout(title=f"Residual Analysis: {y} ~ {x}", showlegend=False)
                show_chart(fig)

        # ── COMPOSITION & FLOW ────────────────────────────────────────────────
        elif chart_type == "Gantt chart":
            st.markdown("**Gantt chart** requires start and end date/time columns.")
            c1, c2, c3 = st.columns(3)
            task_col  = c1.selectbox("Task/Label", cat_cols or all_cols, key="gantt_task")
            start_col = c2.selectbox("Start date",  all_cols, key="gantt_start")
            end_col   = c3.selectbox("End date",    all_cols, key="gantt_end")
            color_col = st.selectbox("Color by", ["None"] + cat_cols, key="gantt_color")
            try:
                df_g = df.copy()
                df_g[start_col] = pd.to_datetime(df_g[start_col], errors="coerce")
                df_g[end_col]   = pd.to_datetime(df_g[end_col], errors="coerce")
                df_g = df_g.dropna(subset=[start_col, end_col])
                if len(df_g) > 100:
                    df_g = df_g.sort_values(start_col).head(100)
                    st.info("Showing first 100 tasks by start date for the Gantt view, computed from the full dataset.")
                fig = px.timeline(df_g, x_start=start_col, x_end=end_col, y=task_col,
                                  color=None if color_col=="None" else color_col,
                                  title="Gantt Chart",
                                  color_discrete_sequence=colors)
                fig.update_yaxes(autorange="reversed")
                show_chart(fig)
            except Exception as e:
                st.error(f"Could not create Gantt chart: {e}. Make sure date columns are parseable.")

        elif chart_type == "General heatmap (matrix)":
            c1, c2, c3 = st.columns(3)
            x_cat = c1.selectbox("X (column)", cat_cols or all_cols, key="hm_x")
            y_cat = c2.selectbox("Y (row)",    [c for c in (cat_cols or all_cols) if c != x_cat] or (cat_cols or all_cols), key="hm_y")
            val   = c3.selectbox("Value",      num_cols, key="hm_v")
            agg = st.selectbox("Aggregation", ["mean","sum","count","max","min"], key="hm_a")
            
            # Zero-Loss Resilience: User-controlled sampling
            col1, col2 = st.columns(2)
            n_rows = col1.slider("Rows to show", 5, 200, 30, key="hm_nr")
            n_cols = col2.slider("Cols to show", 5, 100, 20, key="hm_nc")
            
            df_pivot = df.pivot_table(values=val, index=y_cat, columns=x_cat, aggfunc=agg).fillna(0)
            
            # Apply the limits
            display_df = df_pivot.iloc[:n_rows, :n_cols]
            
            fig = px.imshow(display_df, title=f"Heatmap: {agg}({val}) by {y_cat} × {x_cat}",
                            color_continuous_scale="Viridis", text_auto=".1f", aspect="auto")
            show_chart(fig, height=max(400, len(display_df) * 25))

        elif chart_type == "Parallel categories diagram":
            selected = st.multiselect("Select categorical columns", cat_cols, default=cat_cols[:4], key="pcat_cols")
            color_col = st.selectbox("Color by (numeric)", ["None"] + num_cols, key="pcat_c")
            
            if selected and len(selected) >= 2:
                # Cardinality check for Parallel Categories
                total_combos = 1
                for c in selected: total_combos *= df[c].nunique()
                
                df_pc = df.copy()
                if total_combos > 500:
                    st.warning(
                        f"⚠️ High cardinality detected ({total_combos:,} potential combinations). "
                        "Showing the top 1,000 most frequent combinations computed from full data."
                    )
                    top_combos = (
                        df.groupby(selected)
                          .size()
                          .reset_index(name="count")
                          .nlargest(1000, columns="count")[selected]
                    )
                    df_pc = df.merge(top_combos, on=selected, how="inner")
                
                # Optimization for Parallel Categories (Label density & Height)
                max_cardinality = max(df_pc[c].nunique() for c in selected)
                
                # Dynamic height & font size
                dynamic_height = max(550, max_cardinality * 12)
                f_size = 12
                if max_cardinality > 20: f_size = 10
                if max_cardinality > 40: f_size = 8
                
                fig = px.parallel_categories(df_pc, dimensions=selected,
                    color=None if color_col=="None" else color_col,
                    color_continuous_scale="Viridis",
                    title=f"Parallel Categories Diagram ({'Sampled' if len(df_pc) < len(df) else 'Full Data'})")
                
                fig.update_traces(labelfont=dict(size=f_size), 
                                 tickfont=dict(size=f_size))
                
                show_chart(fig, height=dynamic_height)

        # ── TABLES & SUMMARIES ────────────────────────────────────────────────
        elif chart_type == "Styled summary table":
            st.markdown("### 📊 Professional Summary Statistics")
            selected = st.multiselect("Select columns", num_cols, default=num_cols[:10], key="sst_cols")
            if selected:
                stats = df[selected].describe().T
                stats["null_count"] = df[selected].isnull().sum()
                stats["null_pct"] = (stats["null_count"] / len(df) * 100).round(1)
                stats["skewness"] = df[selected].skew().round(3)
                stats["kurtosis"] = df[selected].kurtosis().round(3)
                stats = stats.round(2)
                # Color-coded heatmap table
                fig = go.Figure(data=[go.Table(
                    header=dict(values=["Column"] + list(stats.columns),
                                fill_color=colors[0], font=dict(color="white", size=13),
                                align="center"),
                    cells=dict(values=[stats.index] + [stats[c] for c in stats.columns],
                               fill_color=[["#f8fafc","#f1f5f9"] * (len(stats) // 2 + 1)] * (len(stats.columns) + 1),
                               font=dict(size=12), align="center",
                               height=30))
                ])
                fig.update_layout(title="Summary Statistics Table", height=max(400, len(stats) * 35 + 100))
                st.plotly_chart(fig, width="stretch")
                add_history("Visualization", f"Summary table — {ds_name}")
                if st.button("📌 Pin to Report", key="pin_summary_table", width="stretch"):
                    try:
                        state_mgr.pin_item("chart", {"chart_type": "Styled summary table", "dataset": ds_name, "plotly_json": fig.to_json()}, f"Summary Table — {ds_name}")
                    except Exception:
                        state_mgr.pin_item("chart", {"chart_type": "Styled summary table", "dataset": ds_name}, f"Summary Table — {ds_name}")
                    st.success("📌 Pinned to Report Studio!")

        elif chart_type == "Pivot heatmap table":
            c1, c2, c3 = st.columns(3)
            row_col = c1.selectbox("Row", cat_cols or all_cols, key="pvt_r")
            col_col = c2.selectbox("Column", [c for c in (cat_cols or all_cols) if c != row_col] or (cat_cols or all_cols), key="pvt_c")
            val_col = c3.selectbox("Value", num_cols, key="pvt_v")
            agg = st.selectbox("Aggregation", ["mean","sum","count","max","min"], key="pvt_a")
            
            # Zero-Loss Resilience: Sliders for visual window
            tx1, tx2 = st.columns(2)
            n_rows = tx1.slider("Rows to include", 5, 500, 40, key="pv_nr")
            n_cols = tx2.slider("Cols to include", 5, 100, 25, key="pv_nc")
            
            try:
                # Pivot on FULL data
                pivot_full = df.pivot_table(values=val_col, index=row_col, columns=col_col,
                                           aggfunc=agg).fillna(0).round(2)
                
                # Sample for VISUALIZATION only
                pivot_viz = pivot_full.iloc[:n_rows, :n_cols]
                
                fig = px.imshow(pivot_viz, title=f"Pivot Heatmap: {agg}({val_col}) — {row_col} × {col_col}",
                                color_continuous_scale="Blues", text_auto=".1f", aspect="auto")
                show_chart(fig, height=max(400, pivot_viz.shape[0] * 30))
                
                st.info(f"💡 Showing {pivot_viz.shape[0]}x{pivot_viz.shape[1]} cells. Total table is {pivot_full.shape[0]}x{pivot_full.shape[1]}. Use sliders to expand.")
            except Exception as e:
                st.error(f"Could not create pivot table: {e}")

    except ImportError as e:
        st.error(f"Missing library: {e}. Restart the app to install it.")
    except Exception as e:
        st.error(f"Chart error: {e}. Try selecting different columns.")
