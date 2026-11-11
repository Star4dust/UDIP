"""
UDIP v3 - Visualization Studio.

Workflow:
1. Select dataset
2. Select chart category
3. Select chart type
4. Select axes
5. Render visualization
6. Optionally pin to report
"""

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.core import add_history
from modules.state_manager import get_state_manager
from modules.radio_interferometry import (
    compare_dirty_and_restored_images,
    compute_calibration_residuals,
    compute_closure_amplitude,
    compute_closure_phase,
    compute_delay_spectrum,
    compute_fringe_rate_spectrum,
    compute_point_spread_function,
    flag_bad_visibilities,
    apply_visibility_flags,
    summarize_visibility_flags,
    compute_stokes_parameters,
    dirty_image_from_visibilities,
    hogbom_clean,
    restore_clean_image,
    summarize_gain_stability,
    summarize_phase_drift,
)


def _sort_channel_columns(columns: list[str]) -> list[str]:
    def sort_key(col: str):
        text = str(col)
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            try:
                return (0, float(match.group(0)))
            except ValueError:
                pass
        return (1, text.lower())

    return sorted(columns, key=sort_key)


def _promote_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df where every column that can be numeric is converted.
    Handles HIRAX HDF5 data where columns may contain:
      - Python lists / numpy arrays (stored as object columns from ND arrays)
      - complex numbers (split to real, imag, amplitude)
      - structured arrays / tuples (unpack first numeric field)
      - byte strings (skip)
    """
    new_cols: dict[str, "pd.Series"] = {}
    for col in df.columns:
        series = df[col]
        # Already numeric — keep as-is
        if pd.api.types.is_numeric_dtype(series):
            new_cols[col] = series
            continue
        if pd.api.types.is_complex_dtype(series):
            new_cols[f"{col}_real"] = series.apply(lambda v: float(v.real) if isinstance(v, complex) else float(np.real(v)))
            new_cols[f"{col}_imag"] = series.apply(lambda v: float(v.imag) if isinstance(v, complex) else float(np.imag(v)))
            new_cols[f"{col}_amp"]  = series.apply(lambda v: float(abs(v)))
            continue
        # object dtype — try to extract a scalar number
        try:
            first = series.dropna().iloc[0] if not series.dropna().empty else None
        except Exception:
            first = None
        if first is None:
            continue
        if isinstance(first, complex):
            new_cols[f"{col}_real"] = pd.to_numeric(series.apply(lambda v: getattr(v, "real", np.nan)), errors="coerce")
            new_cols[f"{col}_imag"] = pd.to_numeric(series.apply(lambda v: getattr(v, "imag", np.nan)), errors="coerce")
            new_cols[f"{col}_amp"]  = pd.to_numeric(series.apply(lambda v: abs(v) if isinstance(v, (int, float, complex)) else np.nan), errors="coerce")
        elif isinstance(first, (list, np.ndarray)):
            try:
                arr = np.array(series.tolist())
                if np.issubdtype(arr.dtype, np.complexfloating):
                    arr = np.abs(arr)
                arr = arr.astype(float)
                if arr.ndim == 1:
                    new_cols[col] = pd.Series(arr, index=df.index)
                elif arr.ndim == 2:
                    for j in range(min(arr.shape[1], 32)):
                        new_cols[f"{col}_{j}"] = pd.Series(arr[:, j], index=df.index)
                else:
                    flat = arr.reshape(len(df), -1)
                    for j in range(min(flat.shape[1], 32)):
                        new_cols[f"{col}_{j}"] = pd.Series(flat[:, j], index=df.index)
            except Exception:
                pass
        elif isinstance(first, tuple):
            # Structured-array row tuple — take first numeric field
            try:
                vals = pd.to_numeric(series.apply(lambda v: v[0] if isinstance(v, tuple) and v else np.nan), errors="coerce")
                if vals.notna().any():
                    new_cols[col] = vals
            except Exception:
                pass
        else:
            # Try generic numeric conversion
            converted = pd.to_numeric(series, errors="coerce")
            if converted.notna().any():
                new_cols[col] = converted
    result = pd.DataFrame(new_cols, index=df.index)
    return result


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return all genuinely numeric column names after promotion."""
    promoted = _promote_to_numeric(df)
    return promoted.select_dtypes(include="number").columns.tolist()


def _complex_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if np.iscomplexobj(df[col].to_numpy())]


def _column_options(df: pd.DataFrame) -> list[str]:
    return df.columns.tolist()


def _template() -> str:
    return "plotly_dark" if st.session_state.get("dark_mode", False) else "plotly_white"


def _find_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lowered = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name in lowered:
            return lowered[name]
    for col in df.columns:
        col_l = str(col).lower()
        if any(name in col_l for name in names):
            return col
    return None


def _pin_chart(fig: go.Figure, chart_type: str, dataset_name: str, config: dict):
    state_mgr = get_state_manager()
    state_mgr.pin_item(
        "chart",
        {
            "chart_type": chart_type,
            "dataset": dataset_name,
            "config": config,
            "plotly_json": fig.to_json(),
        },
        f"{chart_type} - {dataset_name}",
    )


def _render_power_spectrum(df: pd.DataFrame, ds_name: str):
    numeric_cols = _numeric_columns(df)
    complex_cols = _complex_columns(df)
    cols = complex_cols + [c for c in numeric_cols if c not in complex_cols]
    if not cols:
        st.info("Power spectrum needs a numeric or complex-valued column.")
        return None, {}

    col = st.selectbox("Signal column", cols)
    sample_spacing = st.number_input("Sample spacing", min_value=1e-12, value=1.0, format="%.6f")
    series = df[col].dropna().to_numpy()
    if series.size < 2:
        st.info("At least two samples are required.")
        return None, {}

    if np.iscomplexobj(series):
        fft = np.fft.fft(series)
        freq = np.fft.fftfreq(series.size, d=sample_spacing)
    else:
        fft = np.fft.rfft(series)
        freq = np.fft.rfftfreq(series.size, d=sample_spacing)
    power = np.abs(fft) ** 2

    fig = go.Figure(go.Scatter(x=freq, y=power, mode="lines", name="Power"))
    fig.update_layout(
        title=f"Power Spectrum - {col}",
        xaxis_title="Frequency",
        yaxis_title="Power",
        template=_template(),
        height=520,
    )
    return fig, {"column": str(col), "sample_spacing": sample_spacing}


def _render_waterfall(df: pd.DataFrame, ds_name: str):
    numeric_cols = _numeric_columns(df)
    if not numeric_cols:
        st.info("Waterfall plots need numeric channel columns.")
        return None, {}

    time_col = st.selectbox("Time axis", ["Row index"] + _column_options(df))
    channel_cols = st.multiselect(
        "Frequency/channel columns",
        numeric_cols,
        default=numeric_cols[: min(64, len(numeric_cols))],
    )
    if not channel_cols:
        st.info("Select at least one channel column.")
        return None, {}

    channel_cols = _sort_channel_columns(channel_cols)
    display_mode = st.radio(
        "Waterfall representation",
        ["Amplitude", "Power (dB)", "Real part", "Imaginary part"],
        horizontal=True,
    )

    raw = df[channel_cols].to_numpy().T
    if np.iscomplexobj(raw):
        amp = np.abs(raw)
        real = raw.real
        imag = raw.imag
    else:
        amp = np.abs(raw)
        real = raw
        imag = np.zeros_like(raw)

    if display_mode == "Power (dB)":
        z = 10.0 * np.log10(np.maximum(amp ** 2, 1e-12))
    elif display_mode == "Real part":
        z = real
    elif display_mode == "Imaginary part":
        z = imag
    else:
        z = amp

    if np.issubdtype(np.asarray(z).dtype, np.number):
        finite = np.asarray(z)[np.isfinite(z)]
        if finite.size:
            lo, hi = np.nanpercentile(finite, [2, 98])
            if lo < hi:
                z = np.clip(z, lo, hi)

    x = np.arange(len(df)) if time_col == "Row index" else df[time_col].to_numpy()
    fig = go.Figure(go.Heatmap(x=x, y=channel_cols, z=z, colorscale="Viridis", colorbar=dict(title=display_mode)))
    fig.update_layout(
        title=f"Waterfall - {ds_name}",
        xaxis_title="Time sample" if time_col == "Row index" else str(time_col),
        yaxis_title="Frequency/channel",
        template=_template(),
        height=560,
    )
    return fig, {"time": str(time_col), "channels": [str(c) for c in channel_cols], "display_mode": display_mode}


def _render_uv_coverage(df: pd.DataFrame, ds_name: str):
    default_u = _find_column(df, ("u", "uv_u"))
    default_v = _find_column(df, ("v", "uv_v"))
    numeric_cols = _numeric_columns(df)

    if not numeric_cols:
        st.info("UV coverage needs numeric u and v coordinate columns.")
        return None, {}

    u_idx = numeric_cols.index(default_u) if default_u in numeric_cols else 0
    v_idx = numeric_cols.index(default_v) if default_v in numeric_cols else min(1, len(numeric_cols) - 1)
    u_col = st.selectbox("u coordinate", numeric_cols, index=u_idx)
    v_col = st.selectbox("v coordinate", numeric_cols, index=v_idx)
    mirror = st.checkbox("Mirror conjugate samples", value=True)

    if u_col == v_col:
        st.info("Choose distinct u and v coordinate columns.")
        return None, {}

    u = pd.to_numeric(df[u_col], errors="coerce").to_numpy()
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy()
    valid = np.isfinite(u) & np.isfinite(v)
    u = u[valid]
    v = v[valid]
    if u.size == 0:
        st.info("No finite uv coordinates were found.")
        return None, {}

    radius = np.hypot(u, v)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=u,
        y=v,
        mode="markers",
        marker=dict(size=6, color=radius, colorscale="Viridis", showscale=True, opacity=0.9),
        name="uv samples",
    ))
    if mirror:
        fig.add_trace(go.Scatter(
            x=-u,
            y=-v,
            mode="markers",
            marker=dict(size=5, color=radius, colorscale="Viridis", showscale=False, opacity=0.25),
            name="conjugate samples",
        ))
    fig.update_layout(
        title=f"UV Coverage - {ds_name}",
        xaxis_title=f"{u_col} (wavelengths)",
        yaxis_title=f"{v_col} (wavelengths)",
        template=_template(),
        height=560,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return fig, {"u": str(u_col), "v": str(v_col), "mirror": mirror}


def _render_visibility(df: pd.DataFrame, chart_type: str, ds_name: str):
    complex_cols = _complex_columns(df)
    numeric_cols = _numeric_columns(df)
    mode = st.radio("Visibility source", ["Complex column", "Real + imaginary columns"], horizontal=True)

    values = None
    config = {"source": mode}
    if mode == "Complex column":
        if not complex_cols:
            st.info("No complex-valued columns were found. Choose real + imaginary columns if available.")
            return None, {}
        col = st.selectbox("Complex visibility", complex_cols)
        values = df[col].dropna().to_numpy()
        config["column"] = str(col)
    else:
        if len(numeric_cols) < 2:
            st.info("Real + imaginary mode needs two numeric columns.")
            return None, {}
        real_col = st.selectbox("Real component", numeric_cols)
        imag_col = st.selectbox("Imaginary component", [c for c in numeric_cols if c != real_col] or numeric_cols)
        valid = df[[real_col, imag_col]].dropna()
        values = valid[real_col].to_numpy() + 1j * valid[imag_col].to_numpy()
        config.update({"real": str(real_col), "imag": str(imag_col)})

    y = np.abs(values) if chart_type == "Visibility Amplitude" else np.angle(values)
    y_title = "Amplitude" if chart_type == "Visibility Amplitude" else "Phase (radians)"
    fig = go.Figure(go.Scatter(x=np.arange(values.size), y=y, mode="lines", name=y_title))
    fig.update_layout(
        title=f"{chart_type} - {ds_name}",
        xaxis_title="Visibility sample",
        yaxis_title=y_title,
        template=_template(),
        height=520,
    )
    return fig, config


def _render_standard_chart(df: pd.DataFrame, chart_type: str, ds_name: str):
    promoted = _promote_to_numeric(df)
    numeric_cols = _numeric_columns(promoted)
    all_cols = _column_options(df)

    # If not enough numeric columns come from dtypes, try to coerce the string columns
    if len(numeric_cols) < 2:
        for col in df.columns:
            if col not in numeric_cols:
                try:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if s.notna().sum() >= 2:
                        promoted[col] = s
                        if col not in numeric_cols:
                            numeric_cols.append(col)
                except Exception:
                    pass

    if chart_type == "Histogram":
        if not numeric_cols:
            st.info("No numeric columns found. For HDF5 files, switch to the **Interferometry Diagnostics** or **Research Diagnostics** tab for tensor visualisations.")
            return None, {}
        col = st.selectbox("Column", numeric_cols)
        bins = st.slider("Bins", 10, 120, 50)
        fig = go.Figure(go.Histogram(x=promoted[col].dropna(), nbinsx=bins))
        fig.update_layout(title=f"Histogram - {col}", xaxis_title=str(col), yaxis_title="Count", template=_template(), height=520)
        return fig, {"column": str(col), "bins": bins}

    if chart_type == "Scatter":
        if len(numeric_cols) == 0:
            st.info(
                "No numeric columns found in this dataset preview. "
                "For HDF5 files like HIRAX data, use the **Interferometry Diagnostics** tab "
                "to visualise visibility tensors, or the **Research Diagnostics** tab for raw arrays."
            )
            return None, {}
        if len(numeric_cols) == 1:
            # One numeric column — scatter vs row index automatically
            y_col = numeric_cols[0]
            x = np.arange(len(promoted))
            fig = go.Figure(go.Scatter(x=x, y=promoted[y_col], mode="markers"))
            fig.update_layout(
                title=f"Scatter — {y_col} vs row index",
                xaxis_title="Row index",
                yaxis_title=str(y_col),
                template=_template(),
                height=520,
            )
            st.caption(
                f"Only one numeric column (`{y_col}`) was found in this dataset preview, so it is plotted "
                "against row index. For full HIRAX visibility tensor visualisation, use the "
                "**Interferometry Diagnostics** or **Research Diagnostics** tab."
            )
            return fig, {"x": "row_index", "y": str(y_col)}
        x_col = st.selectbox("X axis", numeric_cols)
        y_col = st.selectbox("Y axis", [c for c in numeric_cols if c != x_col] or numeric_cols)
        fig = go.Figure(go.Scatter(x=promoted[x_col], y=promoted[y_col], mode="markers"))
        fig.update_layout(title=f"Scatter - {x_col} vs {y_col}", xaxis_title=str(x_col), yaxis_title=str(y_col), template=_template(), height=520)
        return fig, {"x": str(x_col), "y": str(y_col)}

    if chart_type == "Line":
        if not numeric_cols:
            st.info("No numeric columns found. For HDF5 files use the **Interferometry Diagnostics** tab.")
            return None, {}
        x_col = st.selectbox("X axis", ["Row index"] + all_cols)
        y_col = st.selectbox("Y axis", numeric_cols)
        x = np.arange(len(promoted)) if x_col == "Row index" else promoted[x_col]
        fig = go.Figure(go.Scatter(x=x, y=promoted[y_col], mode="lines"))
        fig.update_layout(title=f"Line - {y_col}", xaxis_title=str(x_col), yaxis_title=str(y_col), template=_template(), height=520)
        return fig, {"x": str(x_col), "y": str(y_col)}

    if chart_type == "Correlation Heatmap":
        if len(numeric_cols) < 2:
            st.info("Correlation heatmap needs at least two numeric columns. For HDF5 files use the **Interferometry Diagnostics** tab.")
            return None, {}
        cols = st.multiselect("Columns", numeric_cols, default=numeric_cols[: min(12, len(numeric_cols))])
        corr = promoted[cols].corr() if cols else None
        if corr is None:
            return None, {}
        fig = go.Figure(go.Heatmap(z=corr, x=cols, y=cols, colorscale="RdBu", zmid=0))
        fig.update_layout(title=f"Correlation Heatmap - {ds_name}", template=_template(), height=560)
        return fig, {"columns": [str(c) for c in cols]}

    return None, {}


def render():
    st.markdown('<div class="main-header">📊 Visualize</div>', unsafe_allow_html=True)
    st.markdown("Select a dataset, choose a chart, and render a scientific visualization.")

    if not st.session_state.get("datasets"):
        st.warning("Upload a dataset first to start visualizing.")
        return

    ds_names = {k: v.get("name", k) for k, v in st.session_state.datasets.items()}
    chosen_id = st.selectbox("Select dataset", list(ds_names.keys()), format_func=lambda key: ds_names[key])
    df = st.session_state.datasets[chosen_id]["df"]
    ds_name = ds_names[chosen_id]

    st.caption(f"{ds_name} - {len(df):,} rows x {len(df.columns):,} columns")

    diag_tabs = st.tabs(["Chart Builder", "Calibration Diagnostics", "Interferometry Diagnostics", "Research Diagnostics", "Flagging / Masking"])

    with diag_tabs[0]:
        chart_categories = {
            "Standard": ["Scatter", "Line", "Histogram", "Correlation Heatmap"],
            "Time-Frequency": ["Waterfall"],
            "Spectral": ["Power Spectrum"],
            "Interferometry": ["UV Coverage", "Visibility Amplitude", "Visibility Phase"],
        }

        category = st.selectbox("Chart category", list(chart_categories.keys()))
        chart_type = st.selectbox("Chart type", chart_categories[category])

        fig = None
        config = {}
        if category == "Standard":
            fig, config = _render_standard_chart(df, chart_type, ds_name)
        elif category == "Time-Frequency":
            fig, config = _render_waterfall(df, ds_name)
        elif category == "Spectral":
            fig, config = _render_power_spectrum(df, ds_name)
        elif category == "Interferometry" and chart_type == "UV Coverage":
            fig, config = _render_uv_coverage(df, ds_name)
        elif category == "Interferometry" and chart_type in {"Visibility Amplitude", "Visibility Phase"}:
            fig, config = _render_visibility(df, chart_type, ds_name)

        if fig is not None:
            if st.button("Render visualization", type="primary"):
                st.session_state["_last_viz"] = {
                    "fig": fig,
                    "chart_type": chart_type,
                    "dataset": ds_name,
                    "config": config,
                }
                add_history("Visualization", f"{chart_type} - {ds_name}")

            last_viz = st.session_state.get("_last_viz")
            if last_viz:
                st.plotly_chart(last_viz["fig"], use_container_width=True)
                if st.button("Pin to Report"):
                    _pin_chart(last_viz["fig"], last_viz["chart_type"], last_viz["dataset"], last_viz["config"])
                    st.success("Chart pinned to Report Studio.")

    with diag_tabs[1]:
        st.caption("These panels help inspect calibration stability using real datasets when model visibilities or gain estimates are available.")
        numeric_cols = _numeric_columns(df)
        complex_cols = _complex_columns(df)
        if len(complex_cols) >= 1 and len(numeric_cols) >= 1:
            measured_col = st.selectbox("Measured complex column", complex_cols, key="cal_measured_col")
            model_col = st.selectbox("Model / reference column", [c for c in complex_cols if c != measured_col] or complex_cols, key="cal_model_col")
            measured = df[measured_col].dropna().to_numpy()
            model = df[model_col].dropna().to_numpy()
            n = min(measured.size, model.size)
            if n >= 2:
                residuals = compute_calibration_residuals(measured[:n], model[:n])
                st.metric("Residual RMS", f"{residuals['rms_amplitude_residual']:.4f}")
                st.metric("Phase RMS", f"{residuals['phase_rms']:.4f}")
                st.plotly_chart(go.Figure(go.Scatter(y=residuals["amplitude_residual"], mode="lines" )).update_layout(title="Amplitude Residuals", template=_template(), height=420), use_container_width=True)
                st.plotly_chart(go.Figure(go.Scatter(y=residuals["phase_residual"], mode="lines" )).update_layout(title="Phase Residuals", template=_template(), height=420), use_container_width=True)
            else:
                st.info("Need at least two aligned complex samples to compute residuals.")
        elif len(numeric_cols) >= 2:
            gain_col = st.selectbox("Gain column", numeric_cols, key="gain_col")
            arr = df[gain_col].dropna().to_numpy()
            stability = summarize_gain_stability(arr)
            st.metric("Mean amplitude", f"{stability['mean_amplitude']:.4f}")
            st.metric("Amplitude std", f"{stability['std_amplitude']:.4f}")
            st.plotly_chart(go.Figure(go.Scatter(y=np.asarray(stability["amplitude_mean"]).ravel(), mode="lines" )).update_layout(title="Gain Amplitude Mean", template=_template(), height=360), use_container_width=True)
        else:
            st.info("Add complex visibilities or gain columns to inspect calibration diagnostics.")

    with diag_tabs[2]:
        st.caption("Inspect phase drift and visibility structure without changing the scientific data.")
        if complex_cols:
            vis_col = st.selectbox("Visibility column", complex_cols, key="phase_vis_col")
            vis = df[vis_col].dropna().to_numpy()
            if vis.size >= 2:
                drift = summarize_phase_drift(vis)
                st.metric("Mean phase slope", f"{drift['mean_slope']:.4f}")
                st.metric("RMS phase slope", f"{drift['rms_slope']:.4f}")
                st.plotly_chart(go.Figure(go.Scatter(y=np.asarray(drift["phase_slope"]).ravel(), mode="lines" )).update_layout(title="Phase Drift", template=_template(), height=360), use_container_width=True)
        elif len(numeric_cols) >= 2:
            u_col = st.selectbox("u column", numeric_cols, key="uv_u_col")
            v_col = st.selectbox("v column", [c for c in numeric_cols if c != u_col] or numeric_cols, key="uv_v_col")
            if u_col != v_col:
                st.plotly_chart(go.Figure(go.Scatter(x=df[u_col], y=df[v_col], mode="markers", marker=dict(size=6, color=np.hypot(df[u_col], df[v_col]), colorscale="Viridis", showscale=True))).update_layout(title="UV Coverage", template=_template(), height=480), use_container_width=True)

        st.markdown("##### Polarization diagnostics")
        pol_cols = complex_cols if complex_cols else numeric_cols
        if len(pol_cols) >= 2:
            xx_col = st.selectbox("XX / I proxy", pol_cols, key="pol_xx_col")
            yy_col = st.selectbox("YY / Q proxy", [c for c in pol_cols if c != xx_col] or pol_cols, key="pol_yy_col")
            extra_cols = [c for c in pol_cols if c not in {xx_col, yy_col}]
            xy_col = st.selectbox("XY optional", ["(none)"] + extra_cols, key="pol_xy_col")
            yx_col = st.selectbox("YX optional", ["(none)"] + [c for c in extra_cols if c != xy_col], key="pol_yx_col")

            xx = df[xx_col].dropna().to_numpy()
            yy = df[yy_col].dropna().to_numpy()
            xy = df[xy_col].dropna().to_numpy() if xy_col != "(none)" else None
            yx = df[yx_col].dropna().to_numpy() if yx_col != "(none)" else None
            n = min(xx.size, yy.size, *(arr.size for arr in [xy, yx] if arr is not None)) if (xy is not None or yx is not None) else min(xx.size, yy.size)
            if n >= 2:
                stokes = compute_stokes_parameters(
                    xx[:n],
                    yy[:n],
                    xy[:n] if xy is not None else None,
                    yx[:n] if yx is not None else None,
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean linear pol.", f"{stokes['mean_linear_fraction']:.4f}")
                c2.metric("Mean circular pol.", f"{stokes['mean_circular_fraction']:.4f}")
                c3.metric("Samples", f"{stokes['n_samples']}")
                pol_fig = go.Figure()
                for label, series in [("I", stokes["I"]), ("Q", stokes["Q"]), ("U", stokes["U"]), ("V", stokes["V"])]:
                    pol_fig.add_trace(go.Scatter(y=np.abs(series), mode="lines", name=label))
                pol_fig.update_layout(title="Stokes / polarization diagnostics", template=_template(), height=420, yaxis_title="|Stokes|")
                st.plotly_chart(pol_fig, use_container_width=True)
        else:
            st.info("No visibility or uv-like columns found for interferometry diagnostics.")

    with diag_tabs[4]:
        st.caption("Flag real-world outliers and inspect the resulting masked visibility set.")
        complex_cols = _complex_columns(df)
        numeric_cols = _numeric_columns(df)
        mode = st.radio("Flagging source", ["Complex column", "Real + imaginary columns"], horizontal=True, key="flagging_source")
        if mode == "Complex column" and complex_cols:
            col = st.selectbox("Complex visibility", complex_cols, key="flag_complex_col")
            series = df[col].dropna().to_numpy()
            if series.size:
                threshold = st.slider("Robust sigma threshold", 1.0, 12.0, 5.0, 0.5, key="flag_threshold")
                mask = flag_bad_visibilities(series, threshold=threshold)
                summary = summarize_visibility_flags(mask)
                c1, c2, c3 = st.columns(3)
                c1.metric("Flagged samples", f"{summary['n_flagged']}")
                c2.metric("Flagged fraction", f"{summary['flagged_fraction']:.2%}")
                c3.metric("Total samples", f"{summary['n_total']}")
                masked = apply_visibility_flags(series, mask)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=np.abs(series), mode="lines", name="Amplitude"))
                fig.add_trace(go.Scatter(y=np.where(mask, np.abs(series), np.nan), mode="markers", name="Flagged", marker=dict(color="red", size=7)))
                fig.update_layout(title="Flagging diagnostics", template=_template(), height=360)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Masked values are preserved as NaN / NaN+1jNaN in the output series.")
        elif mode == "Real + imaginary columns" and len(numeric_cols) >= 2:
            real_col = st.selectbox("Real component", numeric_cols, key="flag_real_col")
            imag_col = st.selectbox("Imaginary component", [c for c in numeric_cols if c != real_col] or numeric_cols, key="flag_imag_col")
            series = df[[real_col, imag_col]].dropna()
            if not series.empty:
                vis = series[real_col].to_numpy() + 1j * series[imag_col].to_numpy()
                threshold = st.slider("Robust sigma threshold", 1.0, 12.0, 5.0, 0.5, key="flag_threshold_pair")
                mask = flag_bad_visibilities(vis, threshold=threshold)
                summary = summarize_visibility_flags(mask)
                c1, c2, c3 = st.columns(3)
                c1.metric("Flagged samples", f"{summary['n_flagged']}")
                c2.metric("Flagged fraction", f"{summary['flagged_fraction']:.2%}")
                c3.metric("Total samples", f"{summary['n_total']}")
                masked = apply_visibility_flags(vis, mask)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=np.abs(vis), mode="lines", name="Amplitude"))
                fig.add_trace(go.Scatter(y=np.where(mask, np.abs(vis), np.nan), mode="markers", name="Flagged", marker=dict(color="red", size=7)))
                fig.update_layout(title="Flagging diagnostics", template=_template(), height=360)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Masked values are preserved as NaN / NaN+1jNaN in the output series.")
        else:
            st.info("Flagging diagnostics need a complex visibility series or paired real/imag columns.")

    with diag_tabs[3]:
        st.caption("Minimal research-grade diagnostics for closure and spectral-delay behavior.")
        complex_cols = _complex_columns(df)
        numeric_cols = _numeric_columns(df)
        if complex_cols:
            vis_col = st.selectbox("Complex visibility column", complex_cols, key="research_vis_col")
            vis = df[vis_col].dropna().to_numpy()
        elif len(numeric_cols) >= 2:
            real_col = st.selectbox("Real column", numeric_cols, key="research_real_col")
            imag_col = st.selectbox("Imaginary column", [c for c in numeric_cols if c != real_col] or numeric_cols, key="research_imag_col")
            valid = df[[real_col, imag_col]].dropna()
            vis = valid[real_col].to_numpy() + 1j * valid[imag_col].to_numpy()
        else:
            vis = np.array([])

        if vis.size >= 3:
            tri = vis[: (vis.size // 3) * 3]
            if tri.size >= 3:
                a = tri[0::3]
                b = tri[1::3]
                c = tri[2::3]
                closure = compute_closure_phase(a, b, c)
                st.metric("Mean closure phase", f"{closure['mean_closure_phase']:.4f}")
                st.plotly_chart(go.Figure(go.Scatter(y=closure["closure_phase"], mode="lines")).update_layout(title="Closure Phase", template=_template(), height=320), use_container_width=True)

        if vis.size >= 4:
            quad = vis[: (vis.size // 4) * 4]
            if quad.size >= 4:
                ab = quad[0::4]
                cd = quad[1::4]
                ac = quad[2::4]
                bd = quad[3::4]
                closure_amp = compute_closure_amplitude(ab, cd, ac, bd)
                st.metric("Mean closure amplitude", f"{closure_amp['mean_closure_amplitude']:.4f}")
                st.plotly_chart(go.Figure(go.Scatter(y=closure_amp["closure_amplitude"], mode="lines")).update_layout(title="Closure Amplitude", template=_template(), height=320), use_container_width=True)

        if vis.size >= 2:
            sample_spacing = st.number_input("Delay / fringe sample spacing", min_value=1e-12, value=1.0, format="%.6f", key="research_sample_spacing")
            delay = compute_delay_spectrum(vis, sample_spacing_hz=sample_spacing)
            fringe = compute_fringe_rate_spectrum(vis, sample_spacing_s=sample_spacing)
            c1, c2 = st.columns(2)
            c1.metric("Peak delay", f"{delay['peak_delay']:.4e}")
            c2.metric("Peak fringe-rate", f"{fringe['peak_fringe_rate']:.4e}")
            st.plotly_chart(go.Figure(go.Scatter(x=delay["delay_axis"], y=np.asarray(delay["average_power"]).ravel(), mode="lines")).update_layout(title="Delay Spectrum", template=_template(), height=320), use_container_width=True)
            st.plotly_chart(go.Figure(go.Scatter(x=fringe["fringe_rate_axis"], y=np.asarray(fringe["average_power"]).ravel(), mode="lines")).update_layout(title="Fringe-Rate Spectrum", template=_template(), height=320), use_container_width=True)

        if vis.size >= 2 and len(numeric_cols) >= 2:
            st.markdown("##### Dirty-beam / restored-beam comparison")
            u_col = st.selectbox("u column", numeric_cols, key="research_dirty_u_col")
            v_col = st.selectbox("v column", [c for c in numeric_cols if c != u_col] or numeric_cols, key="research_dirty_v_col")
            if u_col != v_col:
                u = pd.to_numeric(df[u_col], errors="coerce").to_numpy()
                v = pd.to_numeric(df[v_col], errors="coerce").to_numpy()
                n = min(vis.size, u.size, v.size)
                if n >= 2:
                    dirty = dirty_image_from_visibilities(vis[:n], u[:n], v[:n], image_size=64)
                    psf = compute_point_spread_function(u[:n], v[:n], image_size=64)
                    clean = hogbom_clean(dirty, psf, max_iterations=50)
                    restored = restore_clean_image(clean["components"], clean["residual"], psf)
                    compare = compare_dirty_and_restored_images(dirty, restored["restored_image"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Dirty RMS", f"{compare['dirty_rms']:.4f}")
                    c2.metric("Restored RMS", f"{compare['restored_rms']:.4f}")
                    c3.metric("Difference RMS", f"{compare['difference_rms']:.4f}")
                    st.plotly_chart(go.Figure(go.Heatmap(z=dirty, colorscale="Viridis")).update_layout(title="Dirty image", template=_template(), height=320), use_container_width=True)
                    st.plotly_chart(go.Figure(go.Heatmap(z=clean["residual"], colorscale="Viridis")).update_layout(title="Residual image", template=_template(), height=320), use_container_width=True)
                    st.plotly_chart(go.Figure(go.Heatmap(z=restored["restored_image"], colorscale="Viridis")).update_layout(title="Restored image", template=_template(), height=320), use_container_width=True)
        else:
            st.info("Research diagnostics need at least one complex or real+imag visibility series.")
