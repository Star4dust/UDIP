"""Foundational 21cm cosmology workflows for UDIP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules.core import add_history, sanitize_varname, save_dataset

REST_FREQUENCY_MHZ = 1420.40575177


def frequency_to_redshift(frequency_mhz, rest_frequency_mhz: float = REST_FREQUENCY_MHZ):
    """Convert observed frequency in MHz to 21cm redshift."""
    freq = np.asarray(frequency_mhz, dtype=float)
    if np.any(freq <= 0):
        raise ValueError("Observed frequencies must be positive.")
    return (rest_frequency_mhz / freq) - 1.0


def _coerce_numeric_frame(df: pd.DataFrame, channel_columns: list[str]) -> pd.DataFrame:
    frame = df[channel_columns].apply(pd.to_numeric, errors="coerce")
    if frame.empty:
        raise ValueError("No usable numeric channels were found for 21cm analysis.")
    column_means = frame.mean(axis=0, skipna=True)
    frame = frame.fillna(column_means)
    if frame.isna().any().any():
        frame = frame.fillna(0.0)
    return frame


def _prepare_channel_inputs(columns: list[str], fallback_start: float, channel_width: float) -> dict:
    frequencies = _parse_frequency_columns(columns, fallback_start, channel_width)
    order = np.argsort(frequencies, kind="stable")
    ordered_columns = [columns[i] for i in order]
    ordered_frequencies = frequencies[order]
    rounded = np.round(ordered_frequencies.astype(float), 9)
    has_duplicates = len(np.unique(rounded)) != len(rounded)
    is_monotonic = bool(np.all(np.diff(ordered_frequencies) >= 0))
    return {
        "columns": ordered_columns,
        "frequencies_mhz": ordered_frequencies,
        "redshift": frequency_to_redshift(ordered_frequencies),
        "has_duplicates": has_duplicates,
        "is_monotonic": is_monotonic,
    }


def construct_hydrogen_cube(df: pd.DataFrame, channel_columns: list[str]) -> np.ndarray:
    """Construct a minimal brightness cube from selected frequency-channel columns."""
    if not channel_columns:
        raise ValueError("At least one frequency channel column is required.")
    frame = _coerce_numeric_frame(df, channel_columns)
    data = frame.to_numpy(dtype=float)
    if data.size == 0:
        raise ValueError("Selected 21cm channels are empty after numeric coercion.")
    return data[:, :, np.newaxis]


def pca_foreground_clean(cube: np.ndarray, n_components: int = 3) -> dict:
    """Remove smooth foreground modes with SVD/PCA along frequency channels."""
    if cube.ndim != 3:
        raise ValueError("Expected cube with shape (time, frequency, line_of_sight).")

    matrix = np.asarray(cube, dtype=float).reshape(cube.shape[0], -1)
    if not np.isfinite(matrix).all():
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean
    n_components = max(0, min(int(n_components), min(centered.shape)))

    if centered.size == 0:
        return {
            "cleaned_cube": cube.copy(),
            "foreground_model": np.zeros_like(cube),
            "singular_values": np.asarray([]),
            "explained_variance_ratio": np.asarray([]),
            "n_components": 0,
            "svd_status": "empty",
        }

    try:
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 0.0, None)
        vt = eigvecs[:, order].T
        s = np.sqrt(eigvals)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = centered @ vt.T / np.maximum(s, 1e-12)
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        vt = np.nan_to_num(vt, nan=0.0, posinf=0.0, neginf=0.0)
        svd_status = "fallback_eigh"
    else:
        svd_status = "svd"

    power = np.square(np.asarray(s, dtype=float))
    power_total = float(power.sum()) if power.size else 0.0
    explained_variance_ratio = power / power_total if power_total > 0 else np.zeros_like(power)

    foreground = (
        (u[:, :n_components] * s[:n_components]) @ vt[:n_components, :]
        if n_components
        else np.zeros_like(centered)
    )
    cleaned = centered - foreground
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    foreground = np.nan_to_num(foreground, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "cleaned_cube": cleaned.reshape(cube.shape),
        "foreground_model": np.asarray(foreground).reshape(cube.shape),
        "singular_values": np.asarray(s),
        "explained_variance_ratio": np.asarray(explained_variance_ratio),
        "n_components": n_components,
        "svd_status": svd_status,
    }


def estimate_power_spectrum(cube: np.ndarray) -> dict:
    """Estimate an isotropic 1D power spectrum from a cube with FFT amplitudes."""
    fft_cube = np.fft.fftn(cube)
    power_cube = np.abs(fft_cube) ** 2
    shape = cube.shape
    grids = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    k_radius = np.sqrt(sum(g ** 2 for g in grids)).ravel()
    power = power_cube.ravel()

    if k_radius.size == 0:
        return {"k": np.asarray([]), "power": np.asarray([]), "power_cube": power_cube}

    k_max = float(k_radius.max())
    if k_max <= 0:
        return {"k": np.asarray([0.0]), "power": np.asarray([float(power.mean()) if power.size else 0.0]), "power_cube": power_cube}

    bins = np.linspace(0, k_max, min(40, max(8, cube.size // 4)))
    which = np.digitize(k_radius, bins) - 1
    k_vals = []
    p_vals = []
    for idx in range(len(bins) - 1):
        mask = which == idx
        if np.any(mask):
            k_vals.append(float(k_radius[mask].mean()))
            p_vals.append(float(power[mask].mean()))

    if not k_vals:
        return {"k": np.asarray([k_max]), "power": np.asarray([float(power.mean()) if power.size else 0.0]), "power_cube": power_cube}

    return {"k": np.asarray(k_vals), "power": np.asarray(p_vals), "power_cube": power_cube}


def _parse_frequency_columns(columns: list[str], fallback_start: float, channel_width: float) -> np.ndarray:
    parsed = []
    for idx, col in enumerate(columns):
        text = str(col).lower().replace("mhz", "").replace("freq", "").replace("frequency", "")
        try:
            parsed.append(float(text.strip("_ -")))
        except ValueError:
            parsed.append(fallback_start + idx * channel_width)
    return np.asarray(parsed, dtype=float)


def render():
    st.markdown('<div class="main-header">21cm Cosmology</div>', unsafe_allow_html=True)

    if not st.session_state.get("datasets"):
        st.warning("Upload a dataset before running cosmology analysis.")
        return

    ds_names = {k: v.get("name", k) for k, v in st.session_state.datasets.items()}
    ds_id = st.selectbox("Select dataset", list(ds_names.keys()), format_func=lambda key: ds_names[key])
    df = st.session_state.datasets[ds_id]["df"]
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.info("21cm workflows need multiple numeric frequency-channel columns.")
        return

    selected_cols = st.multiselect(
        "Frequency channel columns",
        numeric_cols,
        default=numeric_cols[: min(64, len(numeric_cols))],
    )
    if not selected_cols:
        return

    c1, c2, c3 = st.columns(3)
    start_freq = c1.number_input("Fallback first frequency (MHz)", value=800.0)
    channel_width = c2.number_input("Fallback channel width (MHz)", value=0.39)
    n_components = c3.slider("Foreground PCA modes", 0, min(10, len(selected_cols)), min(3, len(selected_cols)))

    channel_inputs = _prepare_channel_inputs(selected_cols, start_freq, channel_width)
    if not channel_inputs["is_monotonic"]:
        st.info("Selected channels were reordered into ascending frequency for analysis.")
    if channel_inputs["has_duplicates"]:
        st.warning("Duplicate or nearly duplicate channel frequencies were detected.")

    selected_cols = channel_inputs["columns"]
    frequencies = channel_inputs["frequencies_mhz"]
    redshifts = channel_inputs["redshift"]

    z_table = pd.DataFrame({
        "channel": selected_cols,
        "frequency_mhz": frequencies,
        "redshift": redshifts,
    })
    st.dataframe(z_table, use_container_width=True)

    if st.button("Run 21cm Analysis", type="primary"):
        try:
            cube = construct_hydrogen_cube(df, selected_cols)
            cleaning = pca_foreground_clean(cube, n_components=n_components)
            ps = estimate_power_spectrum(cleaning["cleaned_cube"])
        except Exception as exc:
            st.error(f"21cm analysis could not complete safely: {exc}")
            return
        st.session_state["cosmology_21cm_last"] = {
            "cube": cube,
            "cleaned_cube": cleaning["cleaned_cube"],
            "singular_values": cleaning["singular_values"],
            "explained_variance_ratio": cleaning.get("explained_variance_ratio", np.asarray([])),
            "power_spectrum": ps,
            "columns": selected_cols,
            "frequencies": frequencies,
            "svd_status": cleaning.get("svd_status", "unknown"),
        }
        add_history("21cm Cosmology", f"{ds_names[ds_id]} with {len(selected_cols)} channels")

    result = st.session_state.get("cosmology_21cm_last")
    if not result:
        return

    tabs = st.tabs(["Foreground Modes", "Cleaned Waterfall", "Power Spectrum", "Export"])

    with tabs[0]:
        fig = go.Figure(go.Scatter(y=result["singular_values"], mode="lines+markers"))
        fig.update_layout(title="Foreground Mode Singular Values", xaxis_title="Mode", yaxis_title="Singular value")
        st.plotly_chart(fig, use_container_width=True)
        if result.get("explained_variance_ratio") is not None and len(result["explained_variance_ratio"]):
            st.caption(f"svd mode: {result.get('svd_status', 'unknown')} | first mode variance fraction: {result['explained_variance_ratio'][0]:.2%}")

    with tabs[1]:
        cleaned = result["cleaned_cube"][:, :, 0]
        fig = go.Figure(go.Heatmap(z=cleaned.T, y=result["columns"], colorscale="RdBu", zmid=0))
        fig.update_layout(title="PCA-cleaned Brightness Cube Slice", xaxis_title="Time sample", yaxis_title="Frequency channel")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        ps = result["power_spectrum"]
        fig = go.Figure(go.Scatter(x=ps["k"], y=ps["power"], mode="lines+markers"))
        fig.update_layout(title="Estimated 1D Power Spectrum", xaxis_title="k (normalized)", yaxis_title="Power")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        if st.button("Save Cleaned 21cm Cube Slice"):
            out_name = f"{ds_names[ds_id]}_21cm_pca_cleaned"
            out_id = sanitize_varname(out_name)
            out_df = pd.DataFrame(result["cleaned_cube"][:, :, 0], columns=result["columns"])
            entry = {
                "name": out_name,
                "df": out_df,
                "format": "21cm_cleaned",
                "source": ds_names[ds_id],
                "metadata": {
                    "frequencies_mhz": result["frequencies"].tolist(),
                    "rest_frequency_mhz": REST_FREQUENCY_MHZ,
                    "method": "pca_svd_foreground_cleaning",
                    "svd_status": result.get("svd_status", "unknown"),
                },
            }
            save_dataset(out_id, entry)
            st.session_state.datasets[out_id] = entry
            st.success(f"Saved {out_name}")
